#!/usr/bin/env python3
"""Verify inverse dynamics oracle on dataset trajectories."""

import argparse
import time

import numpy as np

from trackzero.config import load_config
from trackzero.data.dataset import TrajectoryDataset
from trackzero.oracle.inverse_dynamics import InverseDynamicsOracle
from trackzero.sim.simulator import Simulator


def main():
    parser = argparse.ArgumentParser(description="Verify inverse dynamics oracle")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default="data/test.h5")
    parser.add_argument("--n-trajectories", type=int, default=1000)
    parser.add_argument("--save-plot", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    oracle = InverseDynamicsOracle(cfg)
    sim = Simulator(cfg)

    print(f"Loading dataset: {args.dataset}")
    ds = TrajectoryDataset(args.dataset)
    N = min(args.n_trajectories, len(ds))
    print(f"Verifying oracle on {N} trajectories...")

    all_errors = []
    t0 = time.time()

    for i in range(N):
        ref_states, ref_actions = ds[i]

        # Re-simulate to get qacc
        states, qaccs = sim.rollout_with_qacc(
            ref_actions,
            q0=ref_states[0, :2],
            v0=ref_states[0, 2:],
        )

        # Recover actions via oracle
        recovered = oracle.recover_actions(states, qaccs)

        # Compare
        errors = recovered - ref_actions
        all_errors.append(errors)

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{N} done")

    elapsed = time.time() - t0
    all_errors = np.concatenate(all_errors, axis=0)

    max_err = np.max(np.abs(all_errors))
    mean_err = np.mean(np.abs(all_errors))
    print(f"\nOracle verification ({N} trajectories, {elapsed:.1f}s):")
    print(f"  Max |error|:  {max_err:.2e}")
    print(f"  Mean |error|: {mean_err:.2e}")

    if max_err < 1e-10:
        print("  PASS: Oracle achieves machine-epsilon accuracy")
    elif max_err < 1e-6:
        print("  PASS: Oracle achieves near-zero error")
    else:
        print(f"  WARN: Max error {max_err:.2e} exceeds threshold")

    if args.save_plot:
        from trackzero.viz.plots import plot_oracle_verification
        plot_oracle_verification(all_errors, save_path=args.save_plot)
        print(f"  Plot saved to {args.save_plot}")

    ds.close()


if __name__ == "__main__":
    main()
