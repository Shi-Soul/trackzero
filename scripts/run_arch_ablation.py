#!/usr/bin/env python3
"""Architecture ablation: test different model sizes on same 10K mixed data.

Tests hypothesis: is the oracle gap driven by overfitting (too-large model)
or by data distribution?
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from trackzero.config import load_config
from trackzero.data.random_rollout import generate_random_rollout_data
from trackzero.data.dataset import TrajectoryDataset
from trackzero.data.ood_references import generate_ood_reference_data
from trackzero.policy.mlp import InverseDynamicsMLP, MLPPolicy
from trackzero.eval.harness import EvalHarness

BENCHMARK_SEED = 12345
N_PER_FAMILY = 100


def generate_benchmark_data(cfg):
    families = {}
    ds = TrajectoryDataset("data/medium/test.h5")
    all_s, all_a = ds.get_all_states(), ds.get_all_actions()
    ds.close()
    families["multisine"] = (all_s[:N_PER_FAMILY], all_a[:N_PER_FAMILY])
    for name in ["chirp", "step", "random_walk", "sawtooth", "pulse"]:
        s, a = generate_ood_reference_data(cfg, N_PER_FAMILY, action_type=name, seed=BENCHMARK_SEED)
        families[name] = (s, a)
    return families


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--n-hidden", type=int, required=True)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--loss-fn", choices=["mse", "huber"], default="mse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-traj", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg = load_config()
    device = torch.device(args.device)
    tau_max = cfg.pendulum.tau_max
    arch_tag = f"{args.hidden_dim}x{args.n_hidden}"
    extras = []
    if args.dropout > 0:
        extras.append(f"drop{args.dropout}")
    if args.weight_decay > 0:
        extras.append(f"wd{args.weight_decay}")
    if args.lr_schedule != "constant":
        extras.append(f"lr_{args.lr_schedule}")
    if args.loss_fn != "mse":
        extras.append(args.loss_fn)
    if args.seed != 42:
        extras.append(f"s{args.seed}")
    tag = f"{arch_tag}{'_' + '_'.join(extras) if extras else ''}"
    out_dir = Path(f"outputs/arch_{tag}_{args.n_traj // 1000}k")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"ARCHITECTURE ABLATION: {tag}")
    print(f"  Data: {args.n_traj // 1000}K mixed trajectories")
    print(f"  Model: {arch_tag} (dropout={args.dropout}, wd={args.weight_decay}, "
          f"lr={args.lr_schedule}, loss={args.loss_fn})")
    print(f"{'='*60}")

    # Generate data (same seed as baseline)
    print(f"\nGenerating {args.n_traj // 1000}K mixed trajectories...")
    t0 = time.time()
    train_s, train_a = generate_random_rollout_data(
        cfg, args.n_traj, action_type="mixed", seed=args.seed,
        use_gpu=True, gpu_device=args.device
    )
    print(f"  Generated in {time.time() - t0:.0f}s")

    # Prepare training pairs
    s_t = train_s[:, :-1].reshape(-1, 4)
    s_tp1 = train_s[:, 1:].reshape(-1, 4)
    u_t = train_a.reshape(-1, 2)
    X = np.concatenate([s_t, s_tp1], axis=-1).astype(np.float32)
    Y = u_t.astype(np.float32)

    # Validation
    val_ds = TrajectoryDataset("data/medium/test.h5")
    val_s, val_a = val_ds.get_all_states(), val_ds.get_all_actions()
    val_ds.close()
    vs_t = val_s[:, :-1].reshape(-1, 4)
    vs_tp1 = val_s[:, 1:].reshape(-1, 4)
    vu_t = val_a.reshape(-1, 2)
    X_val = np.concatenate([vs_t, vs_tp1], axis=-1).astype(np.float32)
    Y_val = vu_t.astype(np.float32)

    # Build model
    model = InverseDynamicsMLP(
        4, 2, args.hidden_dim, args.n_hidden, dropout=args.dropout
    ).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"  Model: {nparams:,} params")

    # Data to GPU
    X_train = torch.from_numpy(X).to(device)
    Y_train = torch.from_numpy(Y).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    Y_val_t = torch.from_numpy(Y_val).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=3e-4, weight_decay=args.weight_decay
    )
    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )

    loss_fn = torch.nn.functional.mse_loss
    if args.loss_fn == "huber":
        loss_fn = torch.nn.functional.smooth_l1_loss

    batch_size = 4096
    n_batches = len(X_train) // batch_size
    best_val_loss = float("inf")

    print(f"\nTraining {args.epochs} epochs, {len(X_train):,} pairs, batch={batch_size}")
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(X_train), device=device)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            pred = model(X_train[idx])
            loss = loss_fn(pred, Y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Batched validation
        model.eval()
        with torch.no_grad():
            val_total, val_n = 0.0, 0
            for vb in range(0, len(X_val_t), batch_size):
                xv = X_val_t[vb:vb + batch_size]
                yv = Y_val_t[vb:vb + batch_size]
                val_total += torch.nn.functional.mse_loss(model(xv), yv).item() * len(xv)
                val_n += len(xv)
            val_loss = val_total / val_n

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "val_loss": val_loss},
                       out_dir / "best_model.pt")

        if epoch % 50 == 0 or epoch == 1:
            elapsed = time.time() - train_start
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}: train={epoch_loss/n_batches:.6f}, "
                  f"val={val_loss:.6f}, best={best_val_loss:.6f} "
                  f"lr={lr_now:.2e} [{elapsed:.0f}s]")

        if scheduler is not None:
            scheduler.step()

    train_time = time.time() - train_start
    print(f"\nTraining done in {train_time:.0f}s. Best val_loss: {best_val_loss:.6f}")

    # Benchmark
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    policy = MLPPolicy(model, tau_max=tau_max)
    harness = EvalHarness(cfg)
    families = generate_benchmark_data(cfg)

    print(f"\nRunning standard benchmark...")
    benchmark = {}
    all_mse = []
    for fname, (ref_s, ref_a) in families.items():
        summary = harness.evaluate_policy(policy, ref_s, ref_a, max_trajectories=N_PER_FAMILY)
        mse = float(summary.mean_mse_total)
        benchmark[fname] = {"mean_mse": mse, "median_mse": float(summary.median_mse_total)}
        all_mse.extend([r.mse_total for r in summary.results])
        print(f"  {fname:<15s}: mean_mse={mse:.4e}")

    agg_mse = float(np.mean(all_mse))
    benchmark["_aggregate"] = {"mean_mse": agg_mse, "n_trajectories": len(all_mse)}
    print(f"  {'_aggregate':<15s}: mean_mse={agg_mse:.4e}")

    # Save
    results = {
        "architecture": arch_tag,
        "n_params": nparams,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "lr_schedule": args.lr_schedule,
        "loss_fn": args.loss_fn,
        "seed": args.seed,
        "n_trajectories": args.n_traj,
        "best_val_loss": float(best_val_loss),
        "train_time_s": train_time,
        "benchmark_results": benchmark,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
