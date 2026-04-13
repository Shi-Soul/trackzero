#!/usr/bin/env python3
"""Stage 1D: Coverage-Weighted Training (no extra data).

Algorithm: use the SAME 10K random trajectories but weight each training pair
by the inverse density of its state. This upweights rare (high-velocity) states
and downweights common (low-velocity) ones.

Key insight: the model fails on step/random_walk families because high-velocity
states are underrepresented in training data (1.8% of states have |v|>10).
Instead of generating MORE data, we REWEIGHT existing data for uniform coverage.

This is importance sampling: w(x) = 1/p(x), where p(x) is the empirical density.
Approximated by binning the state space and setting w = 1/bin_count.

Comparison: same data as baseline (10K random mixed), same architecture (1024×6),
only the LOSS WEIGHTS change. If this works, it's a pure algorithmic improvement
with zero additional compute for data generation.
"""
import argparse
import json
import time
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


def compute_coverage_weights(states_flat, n_bins=20, temp=1.0):
    """Compute inverse-density weights for uniform state-space coverage.
    
    Args:
        states_flat: (N, 4) state vectors
        n_bins: bins per dimension for density estimation
        temp: temperature for weight sharpness (higher = more uniform weights)
    
    Returns:
        weights: (N,) importance weights, normalized to mean=1
    """
    N, D = states_flat.shape
    
    # Compute bin edges from data
    mins = states_flat.min(axis=0)
    maxs = states_flat.max(axis=0)
    ranges = maxs - mins + 1e-8
    
    # Assign each state to a bin
    normed = (states_flat - mins) / ranges
    bin_idx = np.minimum((normed * n_bins).astype(int), n_bins - 1)
    
    # Count occupancy per bin (flattened index)
    flat_idx = np.zeros(N, dtype=int)
    for d in range(D):
        flat_idx = flat_idx * n_bins + bin_idx[:, d]
    
    # Count
    unique, counts = np.unique(flat_idx, return_counts=True)
    count_map = dict(zip(unique, counts))
    bin_counts = np.array([count_map.get(flat_idx[i], 1) for i in range(N)], dtype=np.float32)
    
    # Weight = 1/count^temp (temp controls sharpness)
    weights = 1.0 / (bin_counts ** temp)
    # Clip extreme weights to prevent gradient explosion
    max_weight = np.percentile(weights, 99)
    weights = np.minimum(weights, max_weight)
    weights = weights / weights.mean()  # normalize to mean=1
    
    return weights


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
    parser.add_argument("--n-bins", type=int, default=15,
                        help="Bins per dimension for density estimation")
    parser.add_argument("--temp", type=float, default=1.0,
                        help="Weight temperature (1.0=inverse density, 0.5=sqrt)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    
    cfg = load_config()
    device = torch.device(args.device)
    tau_max = cfg.pendulum.tau_max
    
    tag = f"bins{args.n_bins}_temp{args.temp}"
    out_dir = Path(f"outputs/weighted_{tag}_10k_1024x6")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*70}")
    print(f"COVERAGE-WEIGHTED TRAINING: {tag}")
    print(f"{'='*70}")
    
    # Generate same 10K data as baseline
    print(f"\nGenerating 10K mixed trajectories (same as baseline)...")
    t0 = time.time()
    train_s, train_a = generate_random_rollout_data(
        cfg, 10000, action_type="mixed", seed=42,
        use_gpu=True, gpu_device=args.device
    )
    gen_time = time.time() - t0
    print(f"  Generated in {gen_time:.0f}s")
    
    # Prepare training pairs
    s_t = train_s[:, :-1].reshape(-1, 4)
    s_tp1 = train_s[:, 1:].reshape(-1, 4)
    u_t = train_a.reshape(-1, 2)
    X = np.concatenate([s_t, s_tp1], axis=-1).astype(np.float32)
    Y = u_t.astype(np.float32)
    print(f"  Training pairs: {len(X):,}")
    
    # Compute coverage weights on s_t (current state)
    print(f"\nComputing coverage weights (n_bins={args.n_bins}, temp={args.temp})...")
    weights = compute_coverage_weights(s_t, n_bins=args.n_bins, temp=args.temp)
    print(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"  Weight std: {weights.std():.4f}")
    print(f"  Effective sample size: {(weights.sum()**2 / (weights**2).sum()):.0f} / {len(weights)}")
    
    # High velocity state weights
    v_mag = np.sqrt((s_t[:, 2:]**2).sum(axis=1))
    hi_v_mask = v_mag > 10
    print(f"  High-v states (|v|>10): {hi_v_mask.mean()*100:.1f}% of data")
    if hi_v_mask.any():
        print(f"    Unweighted contribution: {hi_v_mask.mean()*100:.1f}%")
        weighted_contribution = (weights[hi_v_mask] * hi_v_mask.sum()).sum() / weights.sum() / len(weights) * 100
        print(f"    Weighted contribution: ~{weights[hi_v_mask].mean():.2f}× average weight")
    
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
    
    # Move to GPU
    X_train = torch.from_numpy(X).to(device)
    Y_train = torch.from_numpy(Y).to(device)
    W_train = torch.from_numpy(weights).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    Y_val_t = torch.from_numpy(Y_val).to(device)
    
    # Training with weighted loss
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    batch_size = 4096
    n_batches = len(X_train) // batch_size
    best_val_loss = float("inf")
    epochs = 200
    
    print(f"\nTraining {epochs} epochs with WEIGHTED MSE loss...")
    train_start = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_train), device=device)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            xb, yb, wb = X_train[idx], Y_train[idx], W_train[idx]
            pred = model(xb)
            # Weighted MSE: mean(w * (pred - target)^2)
            per_sample_loss = ((pred - yb) ** 2).mean(dim=1)  # (batch,)
            loss = (wb * per_sample_loss).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Unweighted validation loss (for fair comparison), batched to avoid OOM
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
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "val_loss": val_loss},
                       out_dir / "best_model.pt")
        
        if epoch % 50 == 0 or epoch == 1:
            elapsed = time.time() - train_start
            print(f"  Epoch {epoch:3d}: weighted_loss={epoch_loss/n_batches:.6f}, "
                  f"val_loss={val_loss:.6f}, best={best_val_loss:.6f} [{elapsed:.0f}s]")
    
    train_time = time.time() - train_start
    print(f"\nTraining done in {train_time:.0f}s. Best val_loss: {best_val_loss:.6f}")
    
    # Benchmark
    print(f"\nStandard benchmark...")
    harness = EvalHarness(cfg)
    families = generate_benchmark_data(cfg)
    
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    policy = MLPPolicy(model, tau_max=tau_max)
    
    results = {}
    all_mse = []
    for fname, (ref_s, ref_a) in families.items():
        summary = harness.evaluate_policy(policy, ref_s, ref_a, max_trajectories=N_PER_FAMILY)
        mse = float(summary.mean_mse_total)
        results[fname] = {"mean_mse": mse}
        all_mse.extend([r.mse_total for r in summary.results])
        print(f"  {fname:12s}: {mse:.4e}")
    
    agg = float(np.mean(all_mse))
    results["_aggregate"] = {"mean_mse": agg}
    print(f"  AGGREGATE:    {agg:.4e}")
    
    # Save
    total_time = time.time() - t0
    meta = {
        "method": "coverage_weighted",
        "n_bins": args.n_bins,
        "temp": args.temp,
        "n_traj": 10000,
        "weight_range": [float(weights.min()), float(weights.max())],
        "weight_std": float(weights.std()),
        "effective_sample_size": float((weights.sum()**2 / (weights**2).sum())),
        "gen_time_s": gen_time,
        "train_time_s": train_time,
        "total_time_s": total_time,
        "total_time_min": total_time / 60,
        "best_val_loss": best_val_loss,
        "benchmark_results": results,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nTotal: {total_time:.0f}s ({total_time/60:.1f}min). "
          f"Budget: {'✅' if total_time < 7200 else '❌'}")


if __name__ == "__main__":
    main()
