#!/usr/bin/env python3
"""Stage 1D: Data Generation Strategy Ablation.

Controlled experiment testing whether SMARTER data generation can close the
oracle gap more efficiently than brute-force random scaling.

Design: 2×2 factorial {torque_type} × {v_range} at fixed 10K trajectory budget
  - Torque type: mixed (baseline) vs bangbang (max velocity excitation)
  - Initial velocity: narrow [-3,3] (default) vs wide [-15,15]

Key insight from coverage analysis:
  - bangbang produces 7.2% high-velocity states (|v|>10) vs 1.8% for mixed
  - Benchmark step/random_walk families need ~1.7% high-velocity states
  - But during TRACKING, errors push the system to even higher velocities
  - So training coverage at high-v is critical for error recovery

Controlled variables:
  - Architecture: 1024×6 (10.5M params)
  - Data budget: 10K trajectories × 500 steps = 5M transition pairs
  - Training: 200 epochs, lr=3e-4, batch=4096
  - Evaluation: standard benchmark (600 trajectories, 6 families)
  - Seed: 42

Usage:
    # Run one condition
    python scripts/run_datagen_ablation.py --strategy bangbang --v-range wide --device cuda:0
    
    # Or run all conditions (see launch commands at bottom)
"""
import argparse
import json
import time
import copy

import numpy as np
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from trackzero.config import load_config
from trackzero.data.random_rollout import generate_random_rollout_data
from trackzero.data.dataset import TrajectoryDataset
from trackzero.policy.mlp import InverseDynamicsMLP, MLPPolicy
from trackzero.eval.harness import EvalHarness
from trackzero.data.ood_references import generate_ood_reference_data


BENCHMARK_SEED = 12345
N_PER_FAMILY = 100


def generate_benchmark_data(cfg):
    """Generate the fixed standard benchmark dataset."""
    families = {}
    ds = TrajectoryDataset("data/medium/test.h5")
    all_s, all_a = ds.get_all_states(), ds.get_all_actions()
    ds.close()
    families["multisine"] = (all_s[:N_PER_FAMILY], all_a[:N_PER_FAMILY])
    for name in ["chirp", "step", "random_walk", "sawtooth", "pulse"]:
        s, a = generate_ood_reference_data(cfg, N_PER_FAMILY, action_type=name, seed=BENCHMARK_SEED)
        families[name] = (s, a)
    return families


def train_and_evaluate(args):
    cfg = load_config()
    device = torch.device(args.device)
    tau_max = cfg.pendulum.tau_max

    # Strategy name for output
    strategy_name = f"{args.strategy}_{args.v_range}v"
    out_dir = Path(f"outputs/ablation_{strategy_name}_10k_1024x6")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"STRATEGY: {strategy_name}")
    print(f"  Torque type: {args.strategy}")
    print(f"  V₀ range: {args.v_range} ({'[-3,3]' if args.v_range == 'narrow' else '[-15,15]'})")
    print(f"  Device: {args.device}")
    print(f"  Output: {out_dir}")
    print(f"{'='*70}")

    # Override v_range if wide
    if args.v_range == "wide":
        cfg.dataset.initial_state.v_range = [-15.0, 15.0]
        print(f"  → Overriding v_range to [-15, 15]")

    # Generate training data
    print(f"\nGenerating 10K {args.strategy} trajectories...")
    t0 = time.time()
    train_s, train_a = generate_random_rollout_data(
        cfg, 10000, action_type=args.strategy, seed=42,
        use_gpu=True, gpu_device=args.device
    )
    data_time = time.time() - t0
    print(f"  Generated in {data_time:.0f}s: states={train_s.shape}, actions={train_a.shape}")

    # Coverage statistics
    v = train_s[:, :, 2:].reshape(-1, 2)
    print(f"  Velocity stats: range=[{v.min():.1f}, {v.max():.1f}], std={v.std():.2f}")
    print(f"  |v|>10: {(np.abs(v)>10).mean()*100:.1f}%, |v|>20: {(np.abs(v)>20).mean()*100:.1f}%")

    # Prepare training pairs
    s_t = train_s[:, :-1].reshape(-1, 4)
    s_tp1 = train_s[:, 1:].reshape(-1, 4)
    u_t = train_a.reshape(-1, 2)
    X = np.concatenate([s_t, s_tp1], axis=-1).astype(np.float32)
    Y = u_t.astype(np.float32)
    print(f"  Training pairs: {len(X):,}")

    # Validation data
    val_ds = TrajectoryDataset("data/medium/test.h5")
    val_s, val_a = val_ds.get_all_states(), val_ds.get_all_actions()
    val_ds.close()
    vs_t = val_s[:, :-1].reshape(-1, 4)
    vs_tp1 = val_s[:, 1:].reshape(-1, 4)
    vu_t = val_a.reshape(-1, 2)
    X_val = np.concatenate([vs_t, vs_tp1], axis=-1).astype(np.float32)
    Y_val = vu_t.astype(np.float32)

    # Model
    hidden_dim, n_hidden = 1024, 6
    model = InverseDynamicsMLP(4, 2, hidden_dim, n_hidden).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {hidden_dim}x{n_hidden} ({nparams:,} params)")

    # Move data to GPU
    X_train = torch.from_numpy(X).to(device)
    Y_train = torch.from_numpy(Y).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    Y_val_t = torch.from_numpy(Y_val).to(device)

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    batch_size = 4096
    n_batches = len(X_train) // batch_size
    best_val_loss = float("inf")
    epochs = 200

    print(f"\nTraining for {epochs} epochs, batch_size={batch_size}, n_batches={n_batches}")
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_train), device=device)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            xb, yb = X_train[idx], Y_train[idx]
            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation (batched to avoid OOM)
        model.eval()
        with torch.no_grad():
            val_total, val_n = 0.0, 0
            for vb in range(0, len(X_val_t), batch_size):
                xv = X_val_t[vb:vb+batch_size]
                yv = Y_val_t[vb:vb+batch_size]
                val_total += torch.nn.functional.mse_loss(model(xv), yv).item() * len(xv)
                val_n += len(xv)
            val_loss = val_total / val_n

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "strategy": strategy_name,
            }, out_dir / "best_model.pt")

        if epoch % 20 == 0 or epoch == 1:
            elapsed = time.time() - train_start
            eta = elapsed / epoch * (epochs - epoch)
            print(f"  Epoch {epoch:3d}/{epochs}: train_loss={epoch_loss/n_batches:.6f}, "
                  f"val_loss={val_loss:.6f}, best={best_val_loss:.6f} "
                  f"[{elapsed:.0f}s elapsed, ETA {eta:.0f}s]")

    train_time = time.time() - train_start
    print(f"\nTraining complete in {train_time:.0f}s ({train_time/60:.1f} min)")
    print(f"Best val_loss: {best_val_loss:.6f}")

    # Standard benchmark evaluation
    print(f"\nRunning standard benchmark...")
    harness = EvalHarness(cfg)
    families = generate_benchmark_data(cfg)

    # Load best model
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    policy = MLPPolicy(model, tau_max=tau_max)

    results = {}
    all_mse = []
    for fname, (ref_s, ref_a) in families.items():
        summary = harness.evaluate_policy(policy, ref_s, ref_a, max_trajectories=N_PER_FAMILY)
        mse = float(summary.mean_mse_total)
        results[fname] = {"mean_mse": mse, "median_mse": float(summary.median_mse_total)}
        all_mse.extend([r.mse_total for r in summary.results])
        print(f"  {fname:12s}: mean_mse={mse:.4e}")

    agg_mse = float(np.mean(all_mse))
    results["_aggregate"] = {
        "mean_mse": agg_mse,
        "median_mse": float(np.median(all_mse)),
        "n_trajectories": len(all_mse),
    }
    oracle_mse = 1.85e-4
    print(f"\n  AGGREGATE: {agg_mse:.4e} ({agg_mse/oracle_mse:.2f}× oracle)")

    # Save results
    total_time = time.time() - t0
    meta = {
        "strategy": strategy_name,
        "torque_type": args.strategy,
        "v_range": args.v_range,
        "n_traj": 10000,
        "architecture": f"{hidden_dim}x{n_hidden}",
        "epochs": epochs,
        "best_val_loss": best_val_loss,
        "data_gen_time_s": data_time,
        "train_time_s": train_time,
        "total_time_s": total_time,
        "total_time_min": total_time / 60,
        "benchmark_results": results,
        "velocity_coverage": {
            "v_range": [float(v.min()), float(v.max())],
            "v_std": float(v.std()),
            "pct_above_10": float((np.abs(v) > 10).mean() * 100),
            "pct_above_20": float((np.abs(v) > 20).mean() * 100),
        },
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Results saved to {out_dir}/results.json")

    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Generation Strategy Ablation")
    parser.add_argument("--strategy", choices=["mixed", "bangbang", "uniform", "ou", "gaussian"],
                        default="mixed", help="Torque distribution type")
    parser.add_argument("--v-range", choices=["narrow", "wide"], default="narrow",
                        help="Initial velocity range: narrow=[-3,3], wide=[-15,15]")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    train_and_evaluate(args)
