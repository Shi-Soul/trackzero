#!/usr/bin/env python3
"""Importance-weighted training: resample random data to match benchmark distribution.

Tests hypothesis: if we had the same 10k random trajectories but reweighted
to match benchmark visitation frequency, would performance improve?

Strategy:
  1. Bin training data and benchmark data into 4D cells
  2. Compute importance weight = p_bench(cell) / p_train(cell) for each cell
  3. Resample training pairs using importance weights
  4. Train model on resampled data
  5. Evaluate on standard benchmark

This isolates "distribution mismatch" from "coverage gap".
"""

import argparse
import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from trackzero.config import load_config
from trackzero.data.dataset import TrajectoryDataset
from trackzero.data.ood_references import generate_ood_reference_data
from trackzero.eval.harness import EvalHarness
from trackzero.policy.mlp import InverseDynamicsMLP, MLPPolicy


N_BINS = 15


def compute_cell_indices(states, q_edges, v_edges):
    """Map states to 4D cell indices."""
    flat = states.reshape(-1, 4)
    idx = np.zeros((len(flat), 4), dtype=np.int32)
    idx[:, 0] = np.clip(np.digitize(flat[:, 0], q_edges) - 1, 0, N_BINS - 1)
    idx[:, 1] = np.clip(np.digitize(flat[:, 1], q_edges) - 1, 0, N_BINS - 1)
    idx[:, 2] = np.clip(np.digitize(flat[:, 2], v_edges) - 1, 0, N_BINS - 1)
    idx[:, 3] = np.clip(np.digitize(flat[:, 3], v_edges) - 1, 0, N_BINS - 1)
    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-hidden", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="outputs/resampled_training")
    parser.add_argument("--resample-size", type=int, default=5_000_000,
                        help="Number of training pairs after resampling")
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    tau_max = cfg.pendulum.tau_max

    # Load training data
    print("Loading training data...")
    ds = TrajectoryDataset("data/medium/train.h5")
    train_s, train_a = ds.get_all_states(), ds.get_all_actions()
    ds.close()

    # Load benchmark data (same as standard_benchmark.py)
    print("Loading benchmark data...")
    bench_states = []
    ds = TrajectoryDataset("data/medium/test.h5")
    bench_states.append(ds.get_all_states()[:100])
    ds.close()
    for name in ["chirp", "step", "random_walk", "sawtooth", "pulse"]:
        s, _ = generate_ood_reference_data(
            cfg, 100, action_type=name, seed=12345
        )
        bench_states.append(s)
    bench_flat = np.concatenate([s.reshape(-1, 4) for s in bench_states])

    # Compute bin edges from combined data
    combined = np.concatenate([train_s.reshape(-1, 4), bench_flat])
    q_lo, q_hi = combined[:, :2].min() - 0.01, combined[:, :2].max() + 0.01
    v_lo, v_hi = combined[:, 2:].min() - 0.01, combined[:, 2:].max() + 0.01
    q_edges = np.linspace(q_lo, q_hi, N_BINS + 1)
    v_edges = np.linspace(v_lo, v_hi, N_BINS + 1)

    # Compute cell distributions
    train_flat = train_s.reshape(-1, 4)
    train_cell_idx = compute_cell_indices(train_s, q_edges, v_edges)
    bench_cell_idx = compute_cell_indices(
        np.concatenate(bench_states), q_edges, v_edges
    )

    train_cells = [tuple(c) for c in train_cell_idx.tolist()]
    bench_cells = [tuple(c) for c in bench_cell_idx.tolist()]
    train_counts = Counter(train_cells)
    bench_counts = Counter(bench_cells)
    total_train = sum(train_counts.values())
    total_bench = sum(bench_counts.values())

    # Compute importance weights for each training sample
    # weight_i = p_bench(cell_i) / p_train(cell_i)
    print("Computing importance weights...")
    n_train_states = len(train_flat)
    # Each training pair is (state_t, state_{t+1}) -> action_t
    # We use the state_t cell for weighting
    n_traj, T_plus_1, _ = train_s.shape
    T = T_plus_1 - 1
    n_pairs = n_traj * T

    # Flatten to pairs
    X_s = train_s[:, :-1, :].reshape(n_pairs, 4)  # state_t
    X_s1 = train_s[:, 1:, :].reshape(n_pairs, 4)   # state_{t+1}
    Y = train_a.reshape(n_pairs, 2)                 # action_t
    X = np.concatenate([X_s, X_s1], axis=1)  # (n_pairs, 8)

    pair_cell_idx = compute_cell_indices(
        X_s.reshape(-1, 1, 4), q_edges, v_edges
    )
    pair_cells = [tuple(c) for c in pair_cell_idx.tolist()]

    weights = np.zeros(n_pairs, dtype=np.float64)
    for i, cell in enumerate(pair_cells):
        p_train = train_counts.get(cell, 0) / total_train
        p_bench = bench_counts.get(cell, 0) / total_bench
        if p_train > 0:
            weights[i] = p_bench / p_train
        else:
            weights[i] = 0.0

    # Normalize weights
    weights /= weights.sum()

    # Report weight statistics
    nonzero = weights > 0
    print(f"  Pairs with nonzero weight: {nonzero.sum()} / {n_pairs}")
    print(f"  Max weight: {weights.max():.6f}")
    print(f"  Effective sample size: {1.0 / (weights[nonzero]**2).sum():.0f}")

    # Resample
    print(f"Resampling {args.resample_size} pairs...")
    resample_idx = np.random.choice(
        n_pairs, size=args.resample_size, replace=True, p=weights
    )
    X_resamp = X[resample_idx]
    Y_resamp = Y[resample_idx]

    # Training
    print(f"Training {args.hidden_dim}x{args.n_hidden} on resampled data...")
    device = torch.device(args.device)
    X_t = torch.tensor(X_resamp, dtype=torch.float32)
    Y_t = torch.tensor(Y_resamp, dtype=torch.float32)
    dataset = TensorDataset(X_t, Y_t)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    model = InverseDynamicsMLP(
        state_dim=4, action_dim=2,
        hidden_dim=args.hidden_dim, n_hidden=args.n_hidden,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Validation set (original test set)
    val_ds = TrajectoryDataset("data/medium/test.h5")
    val_s, val_a = val_ds.get_all_states(), val_ds.get_all_actions()
    val_ds.close()
    n_val = val_s.shape[0]
    X_val = np.concatenate(
        [val_s[:, :-1, :].reshape(-1, 4), val_s[:, 1:, :].reshape(-1, 4)],
        axis=1
    )
    Y_val = val_a.reshape(-1, 2)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32, device=device)

    best_val = float("inf")
    t0 = time.time()
    log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        train_loss = epoch_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = torch.nn.functional.mse_loss(val_pred, Y_val_t).item()

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "state_dim": 4,
                "action_dim": 2,
            }, out / "best_model.pt")

        log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val": best_val,
        })

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:>4d}/{args.epochs}"
                f"  train={train_loss:.6f}"
                f"  val={val_loss:.6f}"
                f"  best={best_val:.6f}"
                f"  time={elapsed:.0f}s"
            )

    # Save results
    results = {
        "config": vars(args),
        "best_val_loss": best_val,
        "effective_sample_size": float(1.0 / (weights[nonzero]**2).sum()),
        "training_log": log,
    }
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone! Results in {out}")


if __name__ == "__main__":
    main()
