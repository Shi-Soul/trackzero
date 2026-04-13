#!/usr/bin/env python3
"""Stage 1D: Maximum-Coverage Data Selection.

Algorithm: instead of generating SMARTER data, CURATE random data for coverage.
  1. Generate 100K random trajectories on GPU (fast, <60s)
  2. Compute state-space coverage per trajectory
  3. Greedily select 10K trajectories that maximize total coverage
  4. Train on selected subset

This tests: can intelligent DATA SELECTION from a large random pool match
more data? If 10K selected ≈ 50K random, that's a 5× efficiency gain.

Hypothesis: the step/random_walk benchmark failure is caused by poor
high-velocity coverage. Selecting trajectories with diverse velocity
profiles should specifically improve these families.

Selection strategies:
  - greedy_coverage: bin the state space, greedily pick trajectories
    that fill the most empty bins
  - max_velocity: pick trajectories with highest peak velocity
    (directly targets the high-velocity gap)
  - stratified: uniform sampling across velocity strata
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


def select_greedy_coverage(states, n_select, n_bins=20):
    """Greedily select trajectories maximizing state-space bin coverage.
    
    Bins the 4D state space (q1, q2, v1, v2) into n_bins^4 cells.
    Iteratively picks the trajectory that fills the most new bins.
    """
    N, T, D = states.shape  # (N_traj, T+1, 4)
    
    # Compute per-dimension ranges for binning
    flat = states.reshape(-1, D)
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    ranges = maxs - mins + 1e-8
    
    # Precompute bin indices for each trajectory
    print("  Precomputing bin indices...")
    traj_bins = []
    for i in range(N):
        # Bin each state in the trajectory
        normed = (states[i] - mins) / ranges  # (T+1, 4), values in [0, 1)
        bin_idx = np.minimum((normed * n_bins).astype(int), n_bins - 1)
        # Convert to unique bin hash
        hashes = set()
        for t in range(T):
            h = bin_idx[t, 0] * n_bins**3 + bin_idx[t, 1] * n_bins**2 + \
                bin_idx[t, 2] * n_bins + bin_idx[t, 3]
            hashes.add(h)
        traj_bins.append(hashes)
    
    # Greedy selection
    print(f"  Greedy selection of {n_select} from {N}...")
    selected = []
    covered = set()
    for step in range(n_select):
        best_i = -1
        best_new = -1
        for i in range(N):
            if i in set(selected):
                continue
            new_bins = len(traj_bins[i] - covered)
            if new_bins > best_new:
                best_new = new_bins
                best_i = i
        selected.append(best_i)
        covered.update(traj_bins[best_i])
        if step % 1000 == 0:
            print(f"    Step {step}: {len(covered)} bins covered, last added {best_new} new")
    
    return np.array(selected), len(covered)


def select_max_velocity(states, n_select):
    """Select trajectories with highest peak velocity magnitude."""
    N = states.shape[0]
    # Compute max velocity magnitude per trajectory
    v = states[:, :, 2:]  # (N, T+1, 2)
    v_mag = np.sqrt((v**2).sum(axis=-1))  # (N, T+1)
    peak_v = v_mag.max(axis=1)  # (N,)
    # Select top n_select by peak velocity
    selected = np.argsort(peak_v)[-n_select:]
    return selected, float(peak_v[selected].mean())


def select_stratified_velocity(states, n_select, n_strata=10):
    """Stratified sampling: uniform across velocity strata."""
    N = states.shape[0]
    v = states[:, :, 2:]
    v_mag = np.sqrt((v**2).sum(axis=-1))
    peak_v = v_mag.max(axis=1)
    
    # Assign each trajectory to a stratum
    v_min, v_max = peak_v.min(), peak_v.max()
    boundaries = np.linspace(v_min, v_max, n_strata + 1)
    
    per_stratum = n_select // n_strata
    selected = []
    for s in range(n_strata):
        mask = (peak_v >= boundaries[s]) & (peak_v < boundaries[s+1] + 1e-8)
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            continue
        # Random sample from stratum
        n_pick = min(per_stratum, len(candidates))
        picked = np.random.choice(candidates, n_pick, replace=False)
        selected.extend(picked.tolist())
    
    # Fill remaining
    remaining = n_select - len(selected)
    if remaining > 0:
        all_idx = set(range(N)) - set(selected)
        extra = np.random.choice(list(all_idx), remaining, replace=False)
        selected.extend(extra.tolist())
    
    return np.array(selected[:n_select]), n_strata


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


def train_and_benchmark(X, Y, X_val, Y_val, families, cfg, device, out_dir):
    """Train 1024×6 model and run standard benchmark."""
    tau_max = cfg.pendulum.tau_max
    hidden_dim, n_hidden = 1024, 6
    model = InverseDynamicsMLP(4, 2, hidden_dim, n_hidden).to(device)
    
    X_train = torch.from_numpy(X).to(device)
    Y_train = torch.from_numpy(Y).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    Y_val_t = torch.from_numpy(Y_val).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    batch_size = 4096
    n_batches = len(X_train) // batch_size
    best_val_loss = float("inf")
    epochs = 200
    
    print(f"\nTraining: {epochs} epochs, {len(X_train):,} pairs, batch={batch_size}")
    train_start = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_train), device=device)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            pred = model(X_train[idx])
            loss = torch.nn.functional.mse_loss(pred, Y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            # Batch val to avoid OOM on large val sets with wide models
            val_total = 0.0
            val_n = 0
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
            print(f"  Epoch {epoch:3d}: train={epoch_loss/n_batches:.6f}, "
                  f"val={val_loss:.6f}, best={best_val_loss:.6f} [{elapsed:.0f}s]")
    
    train_time = time.time() - train_start
    print(f"Training done in {train_time:.0f}s. Best val_loss: {best_val_loss:.6f}")
    
    # Benchmark
    harness = EvalHarness(cfg)
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
    print(f"  AGGREGATE:    {agg:.4e} ({agg/3.72e-8:.0f}× oracle)")
    
    return results, best_val_loss, train_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["greedy_coverage", "max_velocity", "stratified"],
                        default="stratified")
    parser.add_argument("--pool-size", type=int, default=100000,
                        help="Size of random trajectory pool to select from")
    parser.add_argument("--select-size", type=int, default=10000,
                        help="Number of trajectories to select")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    
    cfg = load_config()
    device = torch.device(args.device)
    out_dir = Path(f"outputs/selection_{args.method}_10k_1024x6")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*70}")
    print(f"MAX-COVERAGE DATA SELECTION: {args.method}")
    print(f"  Pool: {args.pool_size:,} trajectories → Select: {args.select_size:,}")
    print(f"{'='*70}")
    
    # Step 1: Generate large pool on GPU
    print(f"\nGenerating {args.pool_size:,} trajectory pool...")
    t0 = time.time()
    pool_s, pool_a = generate_random_rollout_data(
        cfg, args.pool_size, action_type="mixed", seed=42,
        use_gpu=True, gpu_device=args.device
    )
    gen_time = time.time() - t0
    print(f"  Generated in {gen_time:.0f}s: {pool_s.shape}")
    
    # Coverage stats of pool
    v = pool_s[:, :, 2:].reshape(-1, 2)
    print(f"  Pool velocity: range=[{v.min():.1f}, {v.max():.1f}], "
          f"std={v.std():.2f}, |v|>10: {(np.abs(v)>10).mean()*100:.1f}%")
    
    # Step 2: Select subset
    print(f"\nSelecting {args.select_size:,} trajectories via {args.method}...")
    t1 = time.time()
    
    if args.method == "greedy_coverage":
        selected_idx, n_bins = select_greedy_coverage(
            pool_s, args.select_size, n_bins=15)
        sel_meta = {"n_bins_covered": n_bins}
    elif args.method == "max_velocity":
        selected_idx, mean_peak_v = select_max_velocity(pool_s, args.select_size)
        sel_meta = {"mean_peak_velocity": mean_peak_v}
    elif args.method == "stratified":
        selected_idx, n_strata = select_stratified_velocity(
            pool_s, args.select_size, n_strata=20)
        sel_meta = {"n_strata": n_strata}
    
    sel_time = time.time() - t1
    print(f"  Selected in {sel_time:.0f}s")
    
    # Coverage of selected subset
    sel_s = pool_s[selected_idx]
    sel_a = pool_a[selected_idx]
    v_sel = sel_s[:, :, 2:].reshape(-1, 2)
    print(f"  Selected velocity: range=[{v_sel.min():.1f}, {v_sel.max():.1f}], "
          f"std={v_sel.std():.2f}, |v|>10: {(np.abs(v_sel)>10).mean()*100:.1f}%")
    
    # Step 3: Prepare training data
    s_t = sel_s[:, :-1].reshape(-1, 4)
    s_tp1 = sel_s[:, 1:].reshape(-1, 4)
    u_t = sel_a.reshape(-1, 2)
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
    
    # Step 4: Train and benchmark
    families = generate_benchmark_data(cfg)
    results, best_val, train_time = train_and_benchmark(
        X, Y, X_val, Y_val, families, cfg, device, out_dir)
    
    # Save
    total_time = time.time() - t0
    meta = {
        "method": args.method,
        "pool_size": args.pool_size,
        "select_size": args.select_size,
        "selection_meta": sel_meta,
        "gen_time_s": gen_time,
        "sel_time_s": sel_time,
        "train_time_s": train_time,
        "total_time_s": total_time,
        "total_time_min": total_time / 60,
        "best_val_loss": best_val,
        "benchmark_results": results,
        "velocity_coverage": {
            "pool_v_std": float(v.std()),
            "pool_pct_above_10": float((np.abs(v)>10).mean()*100),
            "selected_v_std": float(v_sel.std()),
            "selected_pct_above_10": float((np.abs(v_sel)>10).mean()*100),
        },
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Budget: {'✅ PASS' if total_time < 7200 else '❌ OVER'}")


if __name__ == "__main__":
    main()
