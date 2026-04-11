#!/usr/bin/env python3
"""Generate visualizations from dataset and evaluation results."""

import argparse
from pathlib import Path

import numpy as np

from trackzero.config import load_config
from trackzero.data.dataset import TrajectoryDataset
from trackzero.eval.harness import EvalSummary
from trackzero.viz.plots import (
    plot_coverage,
    plot_error_histogram,
    plot_error_cdf,
    plot_trajectory_diagnostics,
    plot_policy_comparison,
)
from trackzero.viz.playback import animate_pendulum


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--eval-results", type=str, default=None)
    parser.add_argument("--compare-policies", type=str, nargs="+", metavar="LABEL:PATH",
                        help="Compare multiple eval JSONs, e.g. Oracle:outputs/oracle_eval.json")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--animate-trajectory", type=int, default=None,
                        help="Index of trajectory to animate")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Coverage plots
    if args.dataset:
        print(f"Loading dataset: {args.dataset}")
        ds = TrajectoryDataset(args.dataset)

        print("Computing coverage...")
        cov = ds.compute_coverage()
        print(f"  Position coverage: {cov['q_coverage']:.1%}")
        print(f"  Velocity coverage: {cov['v_coverage']:.1%}")

        plot_coverage(
            cov["q_hist"], cov["v_hist"],
            cov["q_edges"], cov["v_edges"],
            cov["q_coverage"], cov["v_coverage"],
            save_path=output_dir / "coverage.png",
        )
        print(f"  Coverage plot saved to {output_dir / 'coverage.png'}")

        # Animate a trajectory
        if args.animate_trajectory is not None:
            idx = args.animate_trajectory
            states, actions = ds[idx]
            animate_pendulum(
                states, cfg.simulation.control_dt,
                link_length=cfg.pendulum.link_length,
                save_path=output_dir / f"trajectory_{idx}.gif",
                skip=2,
            )
            print(f"  Animation saved to {output_dir / f'trajectory_{idx}.gif'}")

        ds.close()

    # Multi-policy comparison
    if args.compare_policies:
        summaries = []
        for spec in args.compare_policies:
            label, path = spec.split(":", 1)
            summaries.append((label, EvalSummary.from_json(path)))
            print(f"  Loaded: {label} ({path})")
        out = output_dir / "policy_comparison.png"
        plot_policy_comparison(summaries, save_path=out)
        print(f"  Comparison plot saved to {out}")

    # Evaluation plots
    if args.eval_results:
        print(f"Loading eval results: {args.eval_results}")
        summary = EvalSummary.from_json(args.eval_results)

        plot_error_histogram(summary, save_path=output_dir / "error_histogram.png")
        print(f"  Error histogram saved to {output_dir / 'error_histogram.png'}")

        plot_error_cdf(summary, save_path=output_dir / "error_cdf.png")
        print(f"  Error CDF saved to {output_dir / 'error_cdf.png'}")


if __name__ == "__main__":
    main()
