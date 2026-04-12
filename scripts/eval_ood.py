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
from trackzero.oracle.inverse_dynamics import InverseDynamicsOracle
from trackzero.policy.mlp import MLPPolicy, load_checkpoint


def evaluate_model_on_refs(
    model_path,
    tau_max: float,
    ref_states: np.ndarray,
    ref_actions: np.ndarray,
    harness: EvalHarness,
    max_trajectories: int = 200,
    label: str = "",
    policy_override=None,
) -> dict:
    """Evaluate a single model (or direct policy) on given reference data.

    Args:
        model_path: path to checkpoint, or None if policy_override is given.
        policy_override: if provided, use this callable directly instead of loading from checkpoint.
    """
    if policy_override is not None:
        policy = policy_override
    else:
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
        "model_path": str(model_path) if model_path else "oracle",
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
    parser.add_argument("--model-best", type=str, default=None,
                        help="Optional: best model trained with optimal action distribution")
    parser.add_argument("--n-trajectories", type=int, default=500,
                        help="OOD trajectories per action type")
    parser.add_argument("--eval-trajectories", type=int, default=200,
                        help="Max trajectories to evaluate per type")
    parser.add_argument("--output-dir", type=str, default="outputs/ood_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-oracle", action="store_true",
                        help="Also evaluate the inverse-dynamics oracle (shooting) as upper bound")
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
    if args.model_best and Path(args.model_best).exists():
        models["1B_best_dist"] = args.model_best
    elif args.model_best:
        print(f"WARNING: best model not found at {args.model_best}")

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

        # Optionally evaluate oracle as upper bound
        if args.include_oracle:
            oracle = InverseDynamicsOracle(cfg)
            oracle_policy = oracle.as_policy(mode="shooting")
            result, summary = evaluate_model_on_refs(
                None, cfg.pendulum.tau_max,
                ref_states, ref_actions,
                harness,
                max_trajectories=args.eval_trajectories,
                label=f"oracle on {ood_type}",
                policy_override=oracle_policy,
            )
            type_results["oracle"] = result
            summary.to_json(output_dir / f"{ood_type}_oracle_details.json")

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

        # Print live comparison
        names_all = list(type_results.keys())
        for i, n0 in enumerate(names_all):
            for n1 in names_all[i+1:]:
                v0 = type_results[n0]["mean_mse_total"]
                v1 = type_results[n1]["mean_mse_total"]
                ratio = v0 / max(v1, 1e-20)
                print(f"\n  {n0}/{n1} ratio: {ratio:.2f}x")

    # Summary table
    all_names = []
    for ood_type in ood_types:
        for n in all_results[ood_type]:
            if n not in all_names:
                all_names.append(n)

    print(f"\n{'='*100}")
    print("SUMMARY TABLE (mean MSE_total)")
    print(f"{'='*100}")
    col_w = 18
    print(f"{'OOD Type':<15}", end="")
    for name in all_names:
        print(f"  {name:>{col_w}s}", end="")
    # ratio columns: 1A/oracle, 1B/oracle, 1A/1B
    if "oracle" in all_names and "1A_supervised" in all_names:
        print(f"  {'1A/oracle':>12s}", end="")
    if "oracle" in all_names and "1B_random_rollout" in all_names:
        print(f"  {'1B/oracle':>12s}", end="")
    if "1A_supervised" in all_names and "1B_random_rollout" in all_names:
        print(f"  {'1A/1B':>10s}", end="")
    print()
    print("-" * 100)

    for ood_type in ood_types:
        tr = all_results[ood_type]
        print(f"{ood_type:<15}", end="")
        for name in all_names:
            v = tr.get(name, {}).get("mean_mse_total", float("nan"))
            print(f"  {v:>{col_w}.4e}", end="")
        if "oracle" in tr and "1A_supervised" in tr:
            r = tr["1A_supervised"]["mean_mse_total"] / max(tr["oracle"]["mean_mse_total"], 1e-20)
            print(f"  {r:>12.2f}x", end="")
        if "oracle" in tr and "1B_random_rollout" in tr:
            r = tr["1B_random_rollout"]["mean_mse_total"] / max(tr["oracle"]["mean_mse_total"], 1e-20)
            print(f"  {r:>12.2f}x", end="")
        if "1A_supervised" in tr and "1B_random_rollout" in tr:
            r = tr["1A_supervised"]["mean_mse_total"] / max(tr["1B_random_rollout"]["mean_mse_total"], 1e-20)
            print(f"  {r:>10.2f}x", end="")
        print()

    # Save all results
    results_path = output_dir / "ood_comparison.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
