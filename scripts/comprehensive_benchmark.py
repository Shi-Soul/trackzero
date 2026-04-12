#!/usr/bin/env python3
"""Comprehensive benchmark: evaluate all methods on ID + OOD settings.

Evaluates every available model checkpoint on:
  - ID: multisine test set
  - OOD: chirp, step, random_walk, sawtooth, pulse

Produces a complete comparison table.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from trackzero.config import load_config
from trackzero.data.dataset import TrajectoryDataset
from trackzero.data.ood_references import (
    OOD_ACTION_GENERATORS,
    generate_ood_reference_data,
)
from trackzero.eval.harness import EvalHarness
from trackzero.oracle.inverse_dynamics import InverseDynamicsOracle
from trackzero.policy.mlp import MLPPolicy, load_checkpoint


def eval_model(model_path, harness, ref_s, ref_a, tau_max, max_traj=200):
    """Evaluate a model on given references."""
    model = load_checkpoint(model_path)
    policy = MLPPolicy(model, tau_max=tau_max)
    summary = harness.evaluate_policy(policy, ref_s, ref_a, max_trajectories=max_traj)
    return {
        "mean_mse_total": float(summary.mean_mse_total),
        "median_mse_total": float(summary.median_mse_total),
        "max_mse_total": float(summary.max_mse_total),
        "mean_mse_q": float(summary.mean_mse_q),
        "mean_mse_v": float(summary.mean_mse_v),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medium.yaml")
    parser.add_argument("--val-data", default="data/medium/test.h5")
    parser.add_argument("--output", default="outputs/comprehensive_benchmark.json")
    parser.add_argument("--n-ood", type=int, default=500,
                        help="OOD references per type")
    parser.add_argument("--max-traj", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    cfg = load_config(args.config)
    tau_max = cfg.pendulum.tau_max
    harness = EvalHarness(cfg)

    # Discover all models
    models = {}
    for d in sorted(os.listdir("outputs")):
        for fname in ["best_model.pt", "final_model.pt"]:
            path = f"outputs/{d}/{fname}"
            if os.path.exists(path):
                key = d if fname == "best_model.pt" else f"{d}_final"
                models[key] = path
                break  # prefer best_model.pt

    print(f"Found {len(models)} models:")
    for k in models:
        print(f"  {k}")

    # Load ID test data
    print("\nLoading ID test data...")
    val_ds = TrajectoryDataset(args.val_data)
    val_s, val_a = val_ds.get_all_states(), val_ds.get_all_actions()
    val_ds.close()

    # Generate OOD references
    print(f"Generating OOD references ({args.n_ood} per type)...")
    ood_data = {}
    for ood_type in OOD_ACTION_GENERATORS:
        s, a = generate_ood_reference_data(cfg, ood_type, n_trajectories=args.n_ood)
        ood_data[ood_type] = (s, a)
        print(f"  {ood_type}: {s.shape}")

    # Evaluate all models on all settings
    eval_settings = {"id_multisine": (val_s, val_a)}
    eval_settings.update({f"ood_{k}": v for k, v in ood_data.items()})

    results = {}
    total = len(models) * len(eval_settings)
    done = 0

    for model_name, model_path in models.items():
        results[model_name] = {}
        for setting_name, (ref_s, ref_a) in eval_settings.items():
            done += 1
            print(f"[{done}/{total}] {model_name} on {setting_name}...", end=" ", flush=True)
            try:
                r = eval_model(model_path, harness, ref_s, ref_a, tau_max, args.max_traj)
                results[model_name][setting_name] = r
                print(f"mse={r['mean_mse_total']:.4e}")
            except Exception as e:
                print(f"ERROR: {e}")
                results[model_name][setting_name] = {"error": str(e)}

    # Save full results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 120)
    print("COMPREHENSIVE BENCHMARK RESULTS")
    print("=" * 120)

    settings = list(eval_settings.keys())
    header = f"{'Method':40s}" + "".join(f"{s:>14s}" for s in settings)
    print(header)
    print("-" * len(header))

    baseline_mse = {}
    if "stage1c_random_matched" in results:
        for s in settings:
            r = results["stage1c_random_matched"].get(s, {})
            baseline_mse[s] = r.get("mean_mse_total", 1.0)

    for model_name in sorted(results.keys()):
        row = f"{model_name:40s}"
        for s in settings:
            r = results[model_name].get(s, {})
            mse = r.get("mean_mse_total")
            if mse is not None:
                base = baseline_mse.get(s, mse)
                ratio = mse / base if base > 0 else 0
                row += f"  {mse:.2e}({ratio:4.1f}x)"
            else:
                row += f"{'ERROR':>14s}"
        print(row)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
