#!/usr/bin/env python3
"""Stage 1B: 4D joint state-space coverage analysis.

Computes coverage and entropy over the full 4D state space (q1, q2, v1, v2)
treated as a single vector — not split into separate q and v projections.
Uses numpy.histogramdd with 10 bins per dimension (10^4 = 10,000 total bins).

Usage:
    python scripts/analyze_coverage.py \\
        --config configs/medium.yaml \\
        --ref-data data/medium/train.h5 \\
        --output-dir outputs/coverage_analysis_4d \\
        --n-trajectories 2000

Outputs:
    coverage_summary_4d.json  — coverage + entropy per action type
    coverage_summary_4d.png   — bar chart comparison
    coverage_marginals_4d.png — 2D marginal projections (6 pairs) per type
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trackzero.config import load_config
from trackzero.data.dataset import TrajectoryDataset
from trackzero.data.random_rollout import generate_random_rollout_data


ACTION_TYPES = ["uniform", "gaussian", "ou", "bangbang", "multisine", "mixed"]

# State bounds: [q1, q2, v1, v2]
STATE_RANGES = [(-np.pi, np.pi), (-np.pi, np.pi), (-8.0, 8.0), (-8.0, 8.0)]
BINS_PER_DIM = 10  # 10^4 = 10,000 total 4D bins


def compute_4d_coverage(states: np.ndarray) -> dict:
    """Compute 4D joint state-space coverage and entropy.

    Args:
        states: (N, T+1, 4) array — [q1, q2, v1, v2]

    Returns:
        dict with keys:
            coverage: fraction of 10^4 bins that contain at least one sample
            entropy:  normalised Shannon entropy (0 = single bin, 1 = perfectly uniform)
            n_samples: total number of state samples
            n_bins: total number of bins (BINS_PER_DIM^4)
    """
    flat = states.reshape(-1, states.shape[-1])  # (N*(T+1), 4)
    flat_clipped = np.clip(
        flat,
        [r[0] for r in STATE_RANGES],
        [r[1] for r in STATE_RANGES],
    )
    hist, _ = np.histogramdd(
        flat_clipped,
        bins=BINS_PER_DIM,
        range=STATE_RANGES,
    )

    n_bins = hist.size
    occupied = (hist > 0).sum()
    coverage = occupied / n_bins

    # Normalised Shannon entropy over occupied bins
    flat_hist = hist.flatten().astype(float)
    flat_hist = flat_hist[flat_hist > 0]
    p = flat_hist / flat_hist.sum()
    raw_entropy = -np.sum(p * np.log(p))
    # Normalise by log(n_bins) — max possible entropy if all bins uniformly filled
    max_entropy = np.log(n_bins)
    entropy = raw_entropy / max_entropy

    return {
        "coverage": float(coverage),
        "entropy": float(entropy),
        "n_occupied_bins": int(occupied),
        "n_bins": int(n_bins),
        "n_samples": int(flat.shape[0]),
    }


def compute_2d_marginals(states: np.ndarray, bins: int = 30) -> dict:
    """Compute all 6 pairwise 2D marginal histograms for visualisation."""
    flat = states.reshape(-1, states.shape[-1])
    dim_names = ["q₁", "q₂", "v₁", "v₂"]
    marginals = {}
    for i in range(4):
        for j in range(i + 1, 4):
            ri, rj = STATE_RANGES[i], STATE_RANGES[j]
            h, xe, ye = np.histogram2d(
                np.clip(flat[:, i], ri[0], ri[1]),
                np.clip(flat[:, j], rj[0], rj[1]),
                bins=bins,
                range=[ri, rj],
            )
            key = f"{dim_names[i]}_vs_{dim_names[j]}"
            marginals[key] = {"hist": h, "xedges": xe, "yedges": ye,
                              "xlabel": dim_names[i], "ylabel": dim_names[j]}
    return marginals


def plot_coverage_marginals(all_marginals: dict, output_dir: Path):
    """Plot 2D marginal projections for each action type side-by-side."""
    pairs = ["q₁_vs_q₂", "q₁_vs_v₁", "q₁_vs_v₂", "q₂_vs_v₁", "q₂_vs_v₂", "v₁_vs_v₂"]
    n_types = len(all_marginals)
    n_pairs = len(pairs)

    fig, axes = plt.subplots(n_types, n_pairs, figsize=(n_pairs * 2.5, n_types * 2.5))
    if n_types == 1:
        axes = axes[np.newaxis, :]

    for row, (label, marginals) in enumerate(all_marginals.items()):
        for col, pair in enumerate(pairs):
            ax = axes[row, col]
            m = marginals[pair]
            ax.pcolormesh(m["xedges"], m["yedges"], np.log1p(m["hist"].T), cmap="viridis")
            if row == 0:
                ax.set_title(pair.replace("_vs_", "\nvs\n"), fontsize=8)
            if col == 0:
                clean_label = label.replace("\n", " ")
                ax.set_ylabel(clean_label, fontsize=8, rotation=0, ha="right", labelpad=60)
            ax.tick_params(labelsize=6)

    fig.suptitle("4D Coverage: 2D Marginal Projections (log(1+count))", fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = output_dir / "coverage_marginals_4d.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved marginals grid to {out}")


def plot_coverage_summary_4d(summary: dict, output_dir: Path):
    """Bar chart comparing 4D coverage and entropy."""
    labels = [k.replace("\n", " ") for k in summary.keys()]
    covs = [summary[k]["coverage"] for k in summary]
    ents = [summary[k]["entropy"] for k in summary]
    scores = [0.5 * c + 0.5 * e for c, e in zip(covs, ents)]

    x = np.arange(len(labels))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax1, ax2, ax3 = axes

    for ax, vals, title, color in [
        (ax1, covs, "4D Coverage\n(fraction of 10⁴ bins occupied)", "#4C72B0"),
        (ax2, ents, "4D Entropy\n(normalised Shannon, higher = more uniform)", "#DD8452"),
        (ax3, scores, "Composite Score\n(0.5×coverage + 0.5×entropy)", "#55A868"),
    ]:
        bars = ax.bar(x, vals, width=0.6, color=color, alpha=0.85)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.4)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        f"Stage 1B — 4D Joint Coverage Analysis\n"
        f"State: (q₁,q₂,v₁,v₂), bins: {BINS_PER_DIM}×{BINS_PER_DIM}×{BINS_PER_DIM}×{BINS_PER_DIM} = {BINS_PER_DIM**4:,} total",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    out = output_dir / "coverage_summary_4d.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 4D coverage summary to {out}")


def main():
    parser = argparse.ArgumentParser(description="Stage 1B 4D coverage analysis")
    parser.add_argument("--config", type=str, default="configs/medium.yaml")
    parser.add_argument("--ref-data", type=str, default="data/medium/train.h5",
                        help="Reference (multisine) HDF5 dataset for comparison")
    parser.add_argument("--output-dir", type=str, default="outputs/coverage_analysis_4d")
    parser.add_argument("--n-trajectories", type=int, default=2000,
                        help="Number of random rollout trajectories per action type")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    all_marginals = {}

    # 1. Reference dataset (multisine)
    print(f"Loading reference data: {args.ref_data}")
    ref_ds = TrajectoryDataset(args.ref_data)
    ref_states = ref_ds.get_all_states()
    ref_ds.close()
    print(f"  Reference: {ref_states.shape[0]} trajectories, {ref_states.shape[1]} steps")

    metrics = compute_4d_coverage(ref_states)
    summary["multisine (reference)"] = metrics
    all_marginals["multisine (reference)"] = compute_2d_marginals(ref_states)
    print(f"  multisine (reference): cov4d={metrics['coverage']:.3f}, H4d={metrics['entropy']:.3f}, "
          f"bins={metrics['n_occupied_bins']}/{metrics['n_bins']}")

    # 2. Random rollout datasets (each action type)
    for atype in ACTION_TYPES:
        print(f"Generating {args.n_trajectories} {atype} rollouts...")
        states, _ = generate_random_rollout_data(
            cfg, args.n_trajectories, action_type=atype, seed=args.seed
        )
        metrics = compute_4d_coverage(states)
        summary[atype] = metrics
        all_marginals[atype] = compute_2d_marginals(states)
        print(f"  {atype}: cov4d={metrics['coverage']:.3f}, H4d={metrics['entropy']:.3f}, "
              f"bins={metrics['n_occupied_bins']}/{metrics['n_bins']}, "
              f"score={0.5*metrics['coverage'] + 0.5*metrics['entropy']:.3f}")

    # Save plots
    plot_coverage_marginals(all_marginals, output_dir)
    plot_coverage_summary_4d(summary, output_dir)

    # Save JSON
    with open(output_dir / "coverage_summary_4d.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nCoverage summary saved to {output_dir}/coverage_summary_4d.json")

    # Print ranked table
    print(f"\n{'Action Type':<30} {'cov4d':>7} {'H4d':>7} {'score':>7} {'n_bins':>10}")
    print("-" * 65)
    ranked = sorted(summary.items(), key=lambda x: -(0.5*x[1]["coverage"] + 0.5*x[1]["entropy"]))
    for label, m in ranked:
        score = 0.5 * m["coverage"] + 0.5 * m["entropy"]
        print(f"{label:<30} {m['coverage']:>7.3f} {m['entropy']:>7.3f} {score:>7.3f} "
              f"{m['n_occupied_bins']:>5}/{m['n_bins']:>5}")


if __name__ == "__main__":
    main()

