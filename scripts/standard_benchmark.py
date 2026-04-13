#!/usr/bin/env python3
"""Standard Tracking Benchmark for TRACK-ZERO.

Design principles:
  - Fixed, method-agnostic: same trajectories for ALL models
  - Diverse signal families: multisine, chirp, step, random_walk, sawtooth, pulse
  - The benchmark represents the TASK, not any training distribution
  - Single metric: mean tracking MSE averaged across all families
  - Per-family breakdown for diagnostics

The benchmark generates 100 trajectories × 6 signal families = 600 total tracking tasks.
All trajectories are deterministically seeded for reproducibility.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from trackzero.config import load_config
from trackzero.data.dataset import TrajectoryDataset
from trackzero.data.ood_references import OOD_ACTION_GENERATORS, generate_ood_reference_data
from trackzero.eval.harness import EvalHarness
from trackzero.policy.mlp import InverseDynamicsMLP, MLPPolicy


BENCHMARK_SEED = 12345
N_PER_FAMILY = 100  # trajectories per signal family


def generate_benchmark_data(cfg):
    """Generate the fixed standard benchmark dataset."""
    families = {}

    # Family 1: multisine — use held-out test set (deterministic, fixed)
    print("  Loading multisine references from test set...")
    ds = TrajectoryDataset("data/medium/test.h5")
    all_s, all_a = ds.get_all_states(), ds.get_all_actions()
    ds.close()
    # Take first N_PER_FAMILY trajectories (deterministic subset)
    families["multisine"] = (all_s[:N_PER_FAMILY], all_a[:N_PER_FAMILY])

    # Families 2-6: chirp, step, random_walk, sawtooth, pulse
    for name in ["chirp", "step", "random_walk", "sawtooth", "pulse"]:
        print(f"  Generating {name} references...")
        s, a = generate_ood_reference_data(
            cfg, N_PER_FAMILY, action_type=name, seed=BENCHMARK_SEED
        )
        families[name] = (s, a)

    return families


def evaluate_model(model_path, hidden_dim, n_hidden, harness, families,
                   device, tau_max, label=""):
    """Evaluate one model on the full benchmark."""
    model = InverseDynamicsMLP(
        state_dim=4, action_dim=2, hidden_dim=hidden_dim, n_hidden=n_hidden
    ).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    policy = MLPPolicy(model, tau_max=tau_max)

    results = {}
    all_mse = []
    for fname, (ref_s, ref_a) in families.items():
        summary = harness.evaluate_policy(policy, ref_s, ref_a,
                                          max_trajectories=N_PER_FAMILY)
        mse = float(summary.mean_mse_total)
        median = float(summary.median_mse_total)
        worst = float(summary.max_mse_total)
        results[fname] = {
            "mean_mse": mse,
            "median_mse": median,
            "max_mse": worst,
        }
        all_mse.extend([r.mse_total for r in summary.results])
        if label:
            print(f"  {label:30s} | {fname:12s}: mean={mse:.4e}  med={median:.4e}  max={worst:.4e}")

    # Aggregate: average across ALL trajectories (not per-family average)
    results["_aggregate"] = {
        "mean_mse": float(np.mean(all_mse)),
        "median_mse": float(np.median(all_mse)),
        "max_mse": float(np.max(all_mse)),
        "n_trajectories": len(all_mse),
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="TRACK-ZERO Standard Benchmark")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="outputs/standard_benchmark.json")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_config()
    tau_max = cfg.pendulum.tau_max
    harness = EvalHarness(cfg)

    # Generate benchmark data (deterministic)
    print("Generating standard benchmark trajectories...")
    t0 = time.time()
    families = generate_benchmark_data(cfg)
    print(f"  Generated in {time.time()-t0:.1f}s")
    for name, (s, a) in families.items():
        print(f"    {name}: {s.shape[0]} trajectories, T={s.shape[1]-1}")

    # All models to evaluate
    models = {
        # Stage 1A baseline
        "supervised_1a": ("outputs/stage1a_scaled/best_model.pt", 512, 4),
        # Stage 1C methods (all 512x4)
        "random": ("outputs/stage1c_random_matched/best_model.pt", 512, 4),
        "hybrid_select": ("outputs/stage1c_hybrid_full/best_model.pt", 512, 4),
        "density": ("outputs/stage1c_density_full/best_model.pt", 512, 4),
        "active": ("outputs/stage1c_active_full/best_model.pt", 512, 4),
        "adversarial": ("outputs/stage1c_adversarial_full/best_model.pt", 512, 4),
        "rebalance": ("outputs/stage1c_rebalance_full/best_model.pt", 512, 4),
        "hindsight": ("outputs/stage1c_hindsight_full/best_model.pt", 512, 4),
        "maxent_rl": ("outputs/stage1c_maxent_rl/best_model.pt", 512, 4),
        # Hybrid data strategies
        "hybrid_curriculum": ("outputs/hybrid_curriculum_512x4/phase2_random_best.pt", 512, 4),
        "hybrid_weighted": ("outputs/hybrid_weighted_512x4/weighted_best.pt", 512, 4),
        # HP sweep: random data, larger models
        "hp_random_1024x6": ("outputs/hp_random_1024x6_lr3e4/best_model.pt", 1024, 6),
        "hp_random_2048x3": ("outputs/hp_random_2048x3_lr3e4/best_model.pt", 2048, 3),
        "hp_random_512x4_wd": ("outputs/hp_random_512x4_lr1e4_wd/best_model.pt", 512, 4),
        "hp_random_1024x4_wd": ("outputs/hp_random_1024x4_lr3e4_wd/best_model.pt", 1024, 4),
        # HP sweep: maxent data, larger models
        "hp_maxent_1024x6": ("outputs/hp_maxent_1024x6_lr3e4/best_model.pt", 1024, 6),
        "hp_maxent_2048x4": ("outputs/hp_maxent_2048x4_lr1e4/best_model.pt", 2048, 4),
        # DAgger: iterative benchmark-focused training
        "dagger_512x4": ("outputs/dagger_benchmark_512x4/best_model.pt", 512, 4),
        "dagger_1024x4": ("outputs/dagger_benchmark_1024x4/best_model.pt", 1024, 4),
        # DAgger iteration checkpoints
        "dagger_1024x4_iter0": ("outputs/dagger_benchmark_1024x4/iter0_best.pt", 1024, 4),
        "dagger_1024x4_iter1": ("outputs/dagger_benchmark_1024x4/iter1_best.pt", 1024, 4),
        "dagger_512x4_iter0": ("outputs/dagger_benchmark_512x4/iter0_best.pt", 512, 4),
        "dagger_512x4_iter1": ("outputs/dagger_benchmark_512x4/iter1_best.pt", 512, 4),
        "dagger_512x4_iter2": ("outputs/dagger_benchmark_512x4/iter2_best.pt", 512, 4),
        # Data scaling experiments
        "random_20k_1024x6": ("outputs/random_20k_1024x6/best_model.pt", 1024, 6),
        "random_50k_1024x6": ("outputs/random_50k_1024x6/best_model.pt", 1024, 6),
        "random_50k_1024x6_v2": ("outputs/random_50k_1024x6_v2/best_model.pt", 1024, 6),
        "random_100k_1024x6": ("outputs/random_100k_1024x6/best_model.pt", 1024, 6),
        "random_20k_512x4": ("outputs/random_20k_512x4/best_model.pt", 512, 4),
        # Targeted exploration
        "bangbang_augmented_512x4": ("outputs/bangbang_augmented_512x4/best_model.pt", 512, 4),
    }

    # Filter to existing models
    models = {k: v for k, v in models.items() if os.path.exists(v[0])}
    print(f"\nEvaluating {len(models)} models on benchmark...")

    all_results = {}
    for name, (path, hd, nh) in models.items():
        print(f"\n{'='*70}")
        print(f"Model: {name}")
        print(f"{'='*70}")
        results = evaluate_model(path, hd, nh, harness, families,
                                 device, tau_max, label=name)
        agg = results["_aggregate"]
        print(f"  {'AGGREGATE':30s} | mean={agg['mean_mse']:.4e}  "
              f"med={agg['median_mse']:.4e}  max={agg['max_mse']:.4e}")
        all_results[name] = results

    # Print final ranking
    print("\n" + "=" * 90)
    print("STANDARD BENCHMARK RANKING (by aggregate mean MSE)")
    print("=" * 90)
    ranked = sorted(all_results.items(),
                    key=lambda x: x[1]["_aggregate"]["mean_mse"])

    fnames = [f for f in list(families.keys())]
    header = f"{'Rank':>4} {'Method':<25} {'Aggregate':>12}"
    for f in fnames:
        header += f" {f:>12}"
    print(header)
    print("-" * len(header))

    for rank, (name, res) in enumerate(ranked, 1):
        row = f"{rank:>4} {name:<25} {res['_aggregate']['mean_mse']:>12.4e}"
        for f in fnames:
            row += f" {res[f]['mean_mse']:>12.4e}"
        print(row)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
