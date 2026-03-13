#!/usr/bin/env python3
"""Evaluate a policy on the reference dataset."""

import argparse
import time

import numpy as np

from trackzero.config import load_config
from trackzero.data.dataset import TrajectoryDataset
from trackzero.eval.harness import EvalHarness
from trackzero.oracle.inverse_dynamics import InverseDynamicsOracle


def main():
    parser = argparse.ArgumentParser(description="Evaluate a tracking policy")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default="data/test.h5")
    parser.add_argument("--policy", choices=["oracle", "zero", "supervised"], default="oracle")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (for --policy supervised)")
    parser.add_argument("--oracle-mode", choices=["shooting", "finite_difference"],
                        default="shooting")
    parser.add_argument("--openloop", action="store_true",
                        help="Run open-loop replay sanity check (ignores --policy)")
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    ds = TrajectoryDataset(args.dataset)
    states = ds.get_all_states()
    actions = ds.get_all_actions()

    # Open-loop replay mode
    if args.openloop:
        print("Mode: open-loop replay (sanity check)")
        harness = EvalHarness(cfg)
        N = len(states)
        if args.max_trajectories:
            N = min(N, args.max_trajectories)
        results = []
        for i in range(N):
            results.append(harness.evaluate_trajectory_openloop(states[i], actions[i], i))
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{N} trajectories", flush=True)
        max_err = max(r.mse_total for r in results)
        print(f"\nOpen-loop replay: max MSE_total = {max_err:.2e}")
        if max_err < 1e-15:
            print("  PASS: Simulator is deterministic")
        ds.close()
        return

    # Select policy
    if args.policy == "oracle":
        oracle = InverseDynamicsOracle(cfg)
        policy = oracle.as_policy(mode=args.oracle_mode)
        print(f"Policy: inverse dynamics oracle ({args.oracle_mode})")
    elif args.policy == "supervised":
        from trackzero.policy.mlp import MLPPolicy, load_checkpoint
        if not args.checkpoint:
            args.checkpoint = "outputs/stage1a/best_model.pt"
        model = load_checkpoint(args.checkpoint)
        policy = MLPPolicy(model, tau_max=cfg.pendulum.tau_max)
        print(f"Policy: supervised MLP ({args.checkpoint})")
    elif args.policy == "zero":
        policy = lambda s, ns: np.zeros(2)
        print("Policy: zero torque (baseline)")

    # Evaluate
    harness = EvalHarness(cfg)

    def progress(done, total):
        print(f"  {done}/{total} trajectories", flush=True)

    print("Evaluating...")
    t0 = time.time()
    summary = harness.evaluate_policy(
        policy, states, actions,
        max_trajectories=args.max_trajectories,
        progress_callback=progress,
    )
    elapsed = time.time() - t0

    print(f"\nResults ({summary.n_trajectories} trajectories, {elapsed:.1f}s):")
    print(f"  Mean MSE_q:     {summary.mean_mse_q:.6e}")
    print(f"  Mean MSE_v:     {summary.mean_mse_v:.6e}")
    print(f"  Mean MSE_total: {summary.mean_mse_total:.6e}")
    print(f"  Median MSE_total: {summary.median_mse_total:.6e}")
    print(f"  Max MSE_total:  {summary.max_mse_total:.6e}")
    print(f"  Mean max_err_q: {summary.mean_max_error_q:.6e}")
    print(f"  Mean 95th_q:    {summary.mean_pct95_error_q:.6e}")

    if args.output:
        summary.to_json(args.output)
        print(f"  Results saved to {args.output}")

    ds.close()


if __name__ == "__main__":
    main()
