#!/usr/bin/env python3
"""Targeted capacity sweep: train one config per invocation.

Usage:
  python scripts/train_single_config.py \
    --data-dir outputs/stage1c_maxent_rl \
    --hidden-dim 1024 --n-hidden 6 --epochs 100 --lr 3e-4 \
    --output-dir outputs/sweep_maxent_1024x6 \
    --device cuda:0
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from trackzero.config import load_config
from trackzero.data.dataset import TrajectoryDataset
from trackzero.data.ood_references import (
    OOD_ACTION_GENERATORS,
    generate_ood_reference_data,
)
from trackzero.eval.harness import EvalHarness
from trackzero.policy.mlp import InverseDynamicsMLP, MLPPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--data-h5", default=None)
    parser.add_argument("--config", default="configs/medium.yaml")
    parser.add_argument("--val-data", default="data/medium/test.h5")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-hidden", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ood", action="store_true",
                        help="Also evaluate on OOD references")
    parser.add_argument("--n-ood", type=int, default=500)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    tau_max = cfg.pendulum.tau_max

    # Load training data
    if args.data_h5:
        ds = TrajectoryDataset(args.data_h5)
        train_s, train_a = ds.get_all_states(), ds.get_all_actions()
        ds.close()
    elif args.data_dir:
        data_dir = Path(args.data_dir)
        npy_s = data_dir / "explorer_states.npy"
        npy_a = data_dir / "explorer_actions.npy"
        if npy_s.exists():
            train_s = np.load(npy_s)
            train_a = np.load(npy_a)
        else:
            h5_files = list(data_dir.glob("*.h5"))
            ds = TrajectoryDataset(str(h5_files[0]))
            train_s, train_a = ds.get_all_states(), ds.get_all_actions()
            ds.close()
    else:
        raise ValueError("Provide --data-dir or --data-h5")

    # Extract pairs
    N, Tp1, d = train_s.shape
    s_t = train_s[:, :-1].reshape(-1, d)
    s_tp1 = train_s[:, 1:].reshape(-1, d)
    u_t = train_a.reshape(-1, train_a.shape[-1])
    X_train = np.concatenate([s_t, s_tp1], axis=-1).astype(np.float32)
    Y_train = (u_t / tau_max).astype(np.float32)
    print(f"Training pairs: {len(X_train)}")

    # Load validation data
    val_ds = TrajectoryDataset(args.val_data)
    val_s, val_a = val_ds.get_all_states(), val_ds.get_all_actions()
    val_ds.close()
    v_st = val_s[:, :-1].reshape(-1, 4)
    v_stp1 = val_s[:, 1:].reshape(-1, 4)
    v_ut = val_a.reshape(-1, 2)
    X_val = np.concatenate([v_st, v_stp1], axis=-1).astype(np.float32)
    Y_val = (v_ut / tau_max).astype(np.float32)

    # Model
    model = InverseDynamicsMLP(
        state_dim=4, action_dim=2,
        hidden_dim=args.hidden_dim, n_hidden=args.n_hidden,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.hidden_dim}x{args.n_hidden} ({n_params:,} params)")

    # DataLoaders
    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train)),
        batch_size=args.batch_size, shuffle=True, pin_memory=True,
    )
    val_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)),
        batch_size=args.batch_size, shuffle=False, pin_memory=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None
    log = []

    print(f"\nTraining for {args.epochs} epochs...")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, nb = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(args.device), yb.to(args.device)
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
                xb, yb = xb.to(args.device), yb.to(args.device)
                vl += loss_fn(model(xb), yb).item()
                nvb += 1

        t_avg = epoch_loss / max(nb, 1)
        v_avg = vl / max(nvb, 1)
        scheduler.step()

        if v_avg < best_val:
            best_val = v_avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch <= 3:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"train={t_avg:.6f}  val={v_avg:.6f}  "
                  f"best_val={best_val:.6f}  time={elapsed:.0f}s")
            log.append({
                "epoch": epoch, "train": t_avg,
                "val": v_avg, "best_val": best_val
            })

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.0f}s")
    print(f"Best val loss: {best_val:.6f}")

    if best_state:
        model.load_state_dict(best_state)
    model.to(args.device)

    # Save model
    torch.save({
        "model_state_dict": model.state_dict(),
        "state_dim": 4,
        "action_dim": 2,
    }, out / "best_model.pt")

    # ID evaluation
    print("\nEvaluating on ID test set...")
    policy = MLPPolicy(model, tau_max=tau_max, device=args.device)
    harness = EvalHarness(cfg)
    id_summary = harness.evaluate_policy(
        policy, val_s, val_a, max_trajectories=200
    )
    id_mse = float(id_summary.mean_mse_total)
    id_ratio = id_mse / 1.185e-4
    print(f"  ID mean_mse={id_mse:.4e} ({id_ratio:.1f}x random baseline)")

    results = {
        "config": {
            "hidden_dim": args.hidden_dim,
            "n_hidden": args.n_hidden,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
        },
        "n_params": n_params,
        "n_train_pairs": len(X_train),
        "best_val_loss": float(best_val),
        "train_time_s": elapsed,
        "id_eval": {
            "mean_mse_total": id_mse,
            "max_mse_total": float(id_summary.max_mse_total),
            "median_mse_total": float(id_summary.median_mse_total),
            "mean_mse_q": float(id_summary.mean_mse_q),
            "mean_mse_v": float(id_summary.mean_mse_v),
            "ratio_vs_random": id_ratio,
        },
        "training_log": log,
    }

    # OOD evaluation
    if args.eval_ood:
        print("\nEvaluating on OOD references...")
        ood_results = {}
        for ood_type in OOD_ACTION_GENERATORS:
            s, a = generate_ood_reference_data(
                cfg, ood_type, n_trajectories=args.n_ood
            )
            ood_summary = harness.evaluate_policy(
                policy, s, a, max_trajectories=200
            )
            ood_mse = float(ood_summary.mean_mse_total)
            ood_results[ood_type] = {
                "mean_mse_total": ood_mse,
                "max_mse_total": float(ood_summary.max_mse_total),
                "median_mse_total": float(ood_summary.median_mse_total),
            }
            print(f"  {ood_type}: mean_mse={ood_mse:.4e}")
        results["ood_eval"] = ood_results

    # Save
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    id_summary.to_json(out / "eval_results.json")

    print(f"\nDone! Results in {out}")
    print(f"  ID MSE: {id_mse:.4e} ({id_ratio:.1f}x)")


if __name__ == "__main__":
    main()
