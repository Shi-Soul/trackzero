#!/usr/bin/env python3
"""Auto-benchmark: checks if models have improved since last benchmark, re-evaluates if so.
Run periodically or after training completes."""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from trackzero.config import load_config
from trackzero.data.dataset import TrajectoryDataset
from trackzero.data.ood_references import generate_ood_reference_data
from trackzero.eval.harness import EvalHarness
from trackzero.policy.mlp import InverseDynamicsMLP, MLPPolicy

BENCHMARK_SEED = 12345
N_PER_FAMILY = 100

MODELS = {
    # Key baselines (already benchmarked in v4, for comparison)
    "active_10k_512x4": ("outputs/stage1c_active_full/best_model.pt", 512, 4),
    "random_10k_512x4": ("outputs/stage1c_random_matched/best_model.pt", 512, 4),
    # Data scaling
    "random_20k_1024x6": ("outputs/random_20k_1024x6/best_model.pt", 1024, 6),
    "random_50k_1024x6": ("outputs/random_50k_1024x6/best_model.pt", 1024, 6),
    "random_100k_1024x6": ("outputs/random_100k_1024x6/best_model.pt", 1024, 6),
    "random_20k_512x4": ("outputs/random_20k_512x4/best_model.pt", 512, 4),
    # Coverage test
    "bangbang_aug_512x4": ("outputs/bangbang_augmented_512x4/best_model.pt", 512, 4),
}

RESULTS_FILE = "outputs/scaling_benchmark_v2.json"
TIMESTAMP_FILE = "outputs/scaling_benchmark_timestamps.json"

def get_model_timestamp(path):
    if os.path.exists(path):
        return os.path.getmtime(path)
    return 0

def needs_update(model_name, model_path, timestamps):
    current_ts = get_model_timestamp(model_path)
    last_ts = timestamps.get(model_name, 0)
    return current_ts > last_ts

def evaluate_single(model_path, hd, nh, harness, families, tau_max, device):
    model = InverseDynamicsMLP(state_dim=4, action_dim=2, hidden_dim=hd, n_hidden=nh).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    policy = MLPPolicy(model, tau_max=tau_max)
    
    results = {}
    all_mse = []
    for fname, (ref_s, ref_a) in families.items():
        summary = harness.evaluate_policy(policy, ref_s, ref_a, max_trajectories=N_PER_FAMILY)
        mse = float(summary.mean_mse_total)
        median = float(summary.median_mse_total)
        worst = float(summary.max_mse_total)
        results[fname] = {"mean_mse": mse, "median_mse": median, "max_mse": worst}
        all_mse.extend([r.mse_total for r in summary.results])
    
    results["_aggregate"] = {
        "mean_mse": float(np.mean(all_mse)),
        "median_mse": float(np.median(all_mse)),
        "max_mse": float(np.max(all_mse)),
    }
    return results

def main():
    device = torch.device("cpu")
    cfg = load_config()
    tau_max = cfg.pendulum.tau_max
    harness = EvalHarness(cfg)
    oracle_agg = 1.85e-4
    
    # Load previous results and timestamps
    prev_results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            prev_results = json.load(f)
    
    timestamps = {}
    if os.path.exists(TIMESTAMP_FILE):
        with open(TIMESTAMP_FILE) as f:
            timestamps = json.load(f)
    
    # Check which models need updating
    models_to_eval = {}
    for name, (path, hd, nh) in MODELS.items():
        if not os.path.exists(path):
            continue
        if needs_update(name, path, timestamps):
            models_to_eval[name] = (path, hd, nh)
            print(f"  {name}: checkpoint updated, will re-evaluate")
        else:
            print(f"  {name}: unchanged, skipping")
    
    if not models_to_eval:
        print("\nNo models need updating. Current ranking:")
    else:
        # Generate benchmark data
        print(f"\nEvaluating {len(models_to_eval)} updated models...")
        families = {}
        ds = TrajectoryDataset("data/medium/test.h5")
        families["multisine"] = (ds.get_all_states()[:N_PER_FAMILY], ds.get_all_actions()[:N_PER_FAMILY])
        ds.close()
        for name in ["chirp", "step", "random_walk", "sawtooth", "pulse"]:
            s, a = generate_ood_reference_data(cfg, N_PER_FAMILY, action_type=name, seed=BENCHMARK_SEED)
            families[name] = (s, a)
        
        for name, (path, hd, nh) in models_to_eval.items():
            print(f"\n  Evaluating {name} ({hd}x{nh})...")
            t0 = time.time()
            results = evaluate_single(path, hd, nh, harness, families, tau_max, device)
            agg = results["_aggregate"]
            ratio = agg["mean_mse"] / oracle_agg
            dt = time.time() - t0
            print(f"    Aggregate: {agg['mean_mse']:.4e} ({ratio:.1f}x oracle) [{dt:.0f}s]")
            prev_results[name] = results
            timestamps[name] = get_model_timestamp(path)
        
        with open(RESULTS_FILE, "w") as f:
            json.dump(prev_results, f, indent=2)
        with open(TIMESTAMP_FILE, "w") as f:
            json.dump(timestamps, f, indent=2)
    
    # Print ranking
    print(f"\n{'='*65}")
    print("SCALING BENCHMARK RANKING")
    print(f"{'='*65}")
    ranked = sorted(prev_results.items(), key=lambda x: x[1]["_aggregate"]["mean_mse"])
    print(f"  {'#':>2s} {'Model':30s} {'Agg MSE':>10s} {'xOracle':>8s} {'Step':>10s} {'RW':>10s}")
    for i, (name, res) in enumerate(ranked, 1):
        agg = res["_aggregate"]
        step_mse = res.get("step", {}).get("mean_mse", 0)
        rw_mse = res.get("random_walk", {}).get("mean_mse", 0)
        print(f"  {i:2d} {name:30s} {agg['mean_mse']:10.4e} {agg['mean_mse']/oracle_agg:8.1f}x "
              f"{step_mse:10.4e} {rw_mse:10.4e}")

if __name__ == "__main__":
    main()
