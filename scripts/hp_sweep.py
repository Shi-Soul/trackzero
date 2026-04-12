#!/usr/bin/env python3
"""Hyperparameter sweep for a given training dataset.

Tests multiple (hidden_dim, n_hidden, epochs, lr, batch_size) combos.
Designed to run one GPU per invocation. Use the launcher script
to dispatch across multiple GPUs.

Usage:
  python scripts/hp_sweep.py --data-dir outputs/stage1c_maxent_rl \
    --gpu 0 --tag maxent_rl
"""

import argparse
import json
import itertools
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from trackzero.config import load_config
from trackzero.data.dataset import TrajectoryDataset
from trackzero.eval.harness import EvalHarness
from trackzero.policy.mlp import InverseDynamicsMLP, MLPPolicy


HP_GRID = {
    "hidden_dim": [512, 1024, 2048],
    "n_hidden": [4, 6],
    "epochs": [100],
    "lr": [1e-3, 3e-4],
    "batch_size": [65536],
}


def load_training_data(data_dir: str):
    """Load training data. Handles both trajectory (.h5) and pair (.npy) formats."""
    data_dir = Path(data_dir)

    # Try .npy pair format (maxent_rl, reachability, etc.)
    states_path = data_dir / "explorer_states.npy"
    actions_path = data_dir / "explorer_actions.npy"
    if states_path.exists() and actions_path.exists():
        states = np.load(states_path)  # (N, T+1, 4)
        actions = np.load(actions_path)  # (N, T, 2)
        return states, actions, "npy_trajectories"

    # Try .h5 trajectory format
    h5_files = list(data_dir.glob("*.h5"))
    if h5_files:
        ds = TrajectoryDataset(str(h5_files[0]))
        s, a = ds.get_all_states(), ds.get_all_actions()
        ds.close()
        return s, a, "h5_trajectories"

    raise FileNotFoundError(f"No training data found in {data_dir}")


def extract_pairs(states, actions, tau_max=5.0):
    """Extract (s_t, s_{t+1}, u_t) pairs from trajectory arrays."""
    N, Tp1, d = states.shape
    s_t = states[:, :-1].reshape(-1, d)
    s_tp1 = states[:, 1:].reshape(-1, d)
    u_t = actions.reshape(-1, actions.shape[-1])
    X = np.concatenate([s_t, s_tp1], axis=-1).astype(np.float32)
    Y = (u_t / tau_max).astype(np.float32)
    return X, Y


def train_single(X_train, Y_train, X_val, Y_val, hp, device):
    """Train one model with given hyperparameters."""
    model = InverseDynamicsMLP(
        state_dim=4, action_dim=2,
        hidden_dim=hp["hidden_dim"], n_hidden=hp["n_hidden"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train)),
        batch_size=hp["batch_size"], shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)),
        batch_size=hp["batch_size"], shuffle=False,
        num_workers=0, pin_memory=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp["lr"], weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hp["epochs"]
    )
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None
    log = []

    t0 = time.time()
    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        epoch_loss, nb = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            nb += 1

        model.eval()
        vl, nvb = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                vl += loss_fn(model(xb), yb).item()
                nvb += 1

        t_avg = epoch_loss / max(nb, 1)
        v_avg = vl / max(nvb, 1)
        scheduler.step()

        if v_avg < best_val:
            best_val = v_avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0 or epoch <= 5:
            log.append({"epoch": epoch, "train": t_avg, "val": v_avg})

    elapsed = time.time() - t0

    if best_state:
        model.load_state_dict(best_state)
    model.to(device)

    return model, best_val, n_params, elapsed, log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Directory with training data")
    parser.add_argument("--data-h5", default=None,
                        help="Direct path to .h5 training data")
    parser.add_argument("--config", default="configs/medium.yaml")
    parser.add_argument("--val-data", default="data/medium/test.h5")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tag", default="sweep")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-configs", type=int, default=None,
                        help="Limit number of configs to test")
    parser.add_argument("--eval-traj", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = f"cuda:{args.gpu}"

    cfg = load_config(args.config)
    tau_max = cfg.pendulum.tau_max

    output_dir = Path(args.output_dir or f"outputs/sweep_{args.tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    if args.data_h5:
        ds = TrajectoryDataset(args.data_h5)
        train_s, train_a = ds.get_all_states(), ds.get_all_actions()
        ds.close()
        data_fmt = "h5"
    else:
        train_s, train_a, data_fmt = load_training_data(args.data_dir)
    print(f"Training data: {train_s.shape} ({data_fmt})")

    X_train, Y_train = extract_pairs(train_s, train_a, tau_max)
    print(f"Training pairs: {len(X_train)}")

    # Load validation data
    val_ds = TrajectoryDataset(args.val_data)
    val_s, val_a = val_ds.get_all_states(), val_ds.get_all_actions()
    val_ds.close()
    X_val, Y_val = extract_pairs(val_s, val_a, tau_max)

    # Generate HP configs
    keys = sorted(HP_GRID.keys())
    all_combos = list(itertools.product(*(HP_GRID[k] for k in keys)))
    configs = [dict(zip(keys, combo)) for combo in all_combos]

    if args.max_configs:
        configs = configs[:args.max_configs]
    print(f"\nSweeping {len(configs)} configurations on GPU {args.gpu}")

    # Eval harness
    harness = EvalHarness(cfg)
    results = []

    for i, hp in enumerate(configs):
        hp_str = " ".join(f"{k}={v}" for k, v in sorted(hp.items()))
        print(f"\n[{i+1}/{len(configs)}] {hp_str}")

        model, best_val, n_params, elapsed, log = train_single(
            X_train, Y_train, X_val, Y_val, hp, device
        )
        print(f"  val_loss={best_val:.6f}  params={n_params:,}  time={elapsed:.0f}s")

        # Closed-loop evaluation
        policy = MLPPolicy(model, tau_max=tau_max, device=device)
        summary = harness.evaluate_policy(
            policy, val_s, val_a, max_trajectories=args.eval_traj
        )
        mse = float(summary.mean_mse_total)
        ratio = mse / 1.185e-4
        print(f"  mean_mse={mse:.4e} ({ratio:.1f}x random baseline)")

        entry = {
            **hp,
            "n_params": n_params,
            "best_val_loss": float(best_val),
            "mean_mse_total": mse,
            "max_mse_total": float(summary.max_mse_total),
            "median_mse_total": float(summary.median_mse_total),
            "ratio_vs_random": ratio,
            "train_time_s": elapsed,
            "log": log,
        }
        results.append(entry)

        # Save best model for this config
        if i == 0 or mse < min(r["mean_mse_total"] for r in results[:-1]):
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  *** NEW BEST ***")

    # Save all results
    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump({
            "tag": args.tag,
            "data_dir": str(args.data_dir) if not args.data_h5 else args.data_h5,
            "n_train_pairs": len(X_train),
            "configs": results,
        }, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print(f"SWEEP RESULTS: {args.tag}")
    print("=" * 80)
    sorted_results = sorted(results, key=lambda x: x["mean_mse_total"])
    print(f"{'Config':50s} {'MSE':>12s} {'Ratio':>8s} {'Params':>10s}")
    print("-" * 80)
    for r in sorted_results:
        cfg_str = f"h={r['hidden_dim']} n={r['n_hidden']} e={r['epochs']} lr={r['lr']}"
        print(f"{cfg_str:50s} {r['mean_mse_total']:12.4e} {r['ratio_vs_random']:7.1f}x {r['n_params']:>10,}")

    best = sorted_results[0]
    print(f"\nBest: h={best['hidden_dim']} n={best['n_hidden']} "
          f"e={best['epochs']} lr={best['lr']} → MSE={best['mean_mse_total']:.4e}")


if __name__ == "__main__":
    main()
