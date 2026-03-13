#!/usr/bin/env python3
"""Evaluate policies on out-of-distribution reference trajectories.

Compares Stage 1A (supervised) and Stage 1B (random rollout) on OOD references
to test Hypothesis 6: "The gap between TRACK-ZERO and the supervised baseline
is larger on out-of-distribution references than on in-distribution references."
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from trackzero.config import load_config
from trackzero.data.ood_references import (
    OOD_ACTION_GENERATORS,
    generate_ood_reference_data,
)
from trackzero.eval.harness import EvalHarness
from trackzero.policy.mlp import MLPPolicy, load_checkpoint


def evaluate_model_on_refs(
    model_path: str,
    tau_max: float,
    ref_states: np.ndarray,
    ref_actions: np.ndarray,
    harness: EvalHarness,
    max_trajectories: int = 200,
    label: str = "",
) -> dict:
    """Evaluate a single model on given reference data."""
    model = load_checkpoint(model_path)
    policy = MLPPolicy(model, tau_max=tau_max)

    t0 = time.time()
    summary = harness.evaluate_policy(
        policy, ref_states, ref_actions,
        max_trajectories=max_trajectories,
    )
    elapsed = time.time() - t0

    result = {
        "label": label,
        "model_path": str(model_path),
        "n_trajectories": summary.n_trajectories,
        "mean_mse_q": summary.mean_mse_q,
        "mean_mse_v": summary.mean_mse_v,
        "mean_mse_total": summary.mean_mse_total,
        "median_mse_total": summary.median_mse_total,
        "max_mse_total": summary.max_mse_total,
        "mean_max_error_q": summary.mean_max_error_q,
        "mean_pct95_error_q": summary.mean_pct95_error_q,
        "elapsed_s": elapsed,
    }

    print(f"  {label}:")
    print(f"    MSE_q:     {summary.mean_mse_q:.6e}")
    print(f"    MSE_v:     {summary.mean_mse_v:.6e}")
    print(f"    MSE_total: {summary.mean_mse_total:.6e}")
    print(f"    Max total: {summary.max_mse_total:.6e}")
    print(f"    Max_err_q: {summary.mean_max_error_q:.6e}")
    print(f"    ({elapsed:.1f}s)")

    return result, summary


def main():
    parser = argparse.ArgumentParser(description="OOD evaluation of tracking policies")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model-1a", type=str, default="outputs/stage1a/best_model.pt",
                        help="Stage 1A (supervised) model checkpoint")
    parser.add_argument("--model-1b", type=str, default="outputs/stage1b_mixed_80k/best_model.pt",
                        help="Stage 1B (random rollout) model checkpoint")
    parser.add_argument("--n-trajectories", type=int, default=500,
                        help="OOD trajectories per action type")
    parser.add_argument("--eval-trajectories", type=int, default=200,
                        help="Max trajectories to evaluate per type")
    parser.add_argument("--output-dir", type=str, default="outputs/ood_eval")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    harness = EvalHarness(cfg)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = {}
    if Path(args.model_1a).exists():
        models["1A_supervised"] = args.model_1a
    else:
        print(f"WARNING: 1A model not found at {args.model_1a}")
    if Path(args.model_1b).exists():
        models["1B_random_rollout"] = args.model_1b
    else:
        print(f"WARNING: 1B model not found at {args.model_1b}")

    if not models:
        print("No models found. Exiting.")
        return

    all_results = {}

    # Evaluate on each OOD action type separately
    ood_types = list(OOD_ACTION_GENERATORS.keys()) + ["mixed_ood"]

    for ood_type in ood_types:
        print(f"\n{'='*60}")
        print(f"OOD Type: {ood_type}")
        print(f"{'='*60}")

        print(f"Generating {args.n_trajectories} {ood_type} reference trajectories...")
        t0 = time.time()
        ref_states, ref_actions = generate_ood_reference_data(
            cfg, args.n_trajectories,
            action_type=ood_type,
            seed=args.seed,
        )
        print(f"  Done in {time.time() - t0:.1f}s")

        type_results = {}
        for model_name, model_path in models.items():
            result, summary = evaluate_model_on_refs(
                model_path, cfg.pendulum.tau_max,
                ref_states, ref_actions,
                harness,
                max_trajectories=args.eval_trajectories,
                label=f"{model_name} on {ood_type}",
            )
            type_results[model_name] = result

            # Save per-type detailed results
            summary.to_json(output_dir / f"{ood_type}_{model_name}_details.json")

        all_results[ood_type] = type_results

        # Print comparison
        if len(models) == 2:
            names = list(type_results.keys())
            r0, r1 = type_results[names[0]], type_results[names[1]]
            ratio = r0["mean_mse_total"] / max(r1["mean_mse_total"], 1e-20)
            print(f"\n  Ratio ({names[0]} / {names[1]}): {ratio:.2f}x")

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'OOD Type':<15}", end="")
    for name in models:
        print(f"  {name:>20s}", end="")
    if len(models) == 2:
        print(f"  {'Ratio (1A/1B)':>15s}", end="")
    print()
    print("-" * 80)

    for ood_type in ood_types:
        print(f"{ood_type:<15}", end="")
        vals = []
        for name in models:
            v = all_results[ood_type][name]["mean_mse_total"]
            vals.append(v)
            print(f"  {v:>20.6e}", end="")
        if len(vals) == 2:
            ratio = vals[0] / max(vals[1], 1e-20)
            print(f"  {ratio:>15.2f}x", end="")
        print()

    # Save all results
    results_path = output_dir / "ood_comparison.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
