#!/usr/bin/env python3
"""Stage 1B: Hyperparameter sweep to find the optimal action distribution.

Sweeps over action type × hyperparameter combinations and ranks them by
4D joint state-space coverage and normalised entropy. The best-scoring
config is used to train the final Stage 1B model.

Metric:
    composite_score = 0.5 * coverage_4d + 0.5 * entropy_4d
    where coverage_4d = fraction of 10^4 bins (10 per dim) occupied
          entropy_4d  = normalised Shannon entropy over those bins

Usage:
    python scripts/search_action_distribution.py \\
        --config configs/medium.yaml \\
        --output-dir outputs/action_sweep \\
        --n-trajectories 1000 \\
        --seed 42

Outputs:
    outputs/action_sweep/sweep_results.json  — full results for all configs
    outputs/action_sweep/sweep_ranking.png   — composite score bar chart
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trackzero.config import load_config
from trackzero.data.random_rollout import generate_random_rollout_data

# State bounds: [q1, q2, v1, v2]
STATE_RANGES = [(-np.pi, np.pi), (-np.pi, np.pi), (-8.0, 8.0), (-8.0, 8.0)]
BINS_PER_DIM = 10  # 10^4 = 10,000 total bins


def compute_4d_coverage(states: np.ndarray) -> dict:
    flat = states.reshape(-1, states.shape[-1])
    flat_clipped = np.clip(flat, [r[0] for r in STATE_RANGES], [r[1] for r in STATE_RANGES])
    hist, _ = np.histogramdd(flat_clipped, bins=BINS_PER_DIM, range=STATE_RANGES)
    n_bins = hist.size
    occupied = (hist > 0).sum()
    coverage = occupied / n_bins
    flat_hist = hist.flatten().astype(float)
    flat_hist = flat_hist[flat_hist > 0]
    p = flat_hist / flat_hist.sum()
    entropy = -np.sum(p * np.log(p)) / np.log(n_bins)
    return {
        "coverage": float(coverage),
        "entropy": float(entropy),
        "score": float(0.5 * coverage + 0.5 * entropy),
        "n_occupied_bins": int(occupied),
        "n_bins": int(n_bins),
    }


# ---------------------------------------------------------------------------
# Sweep configurations
# ---------------------------------------------------------------------------
# Each entry: (name, action_type, action_params)
SWEEP_CONFIGS = [
    # ── Baseline types ──────────────────────────────────────────────────────
    ("uniform",             "uniform",   {}),
    ("gaussian_narrow",     "gaussian",  {"scale_factor": 0.20}),
    ("gaussian_medium",     "gaussian",  {"scale_factor": 0.33}),
    ("gaussian_wide",       "gaussian",  {"scale_factor": 0.50}),
    ("multisine",           "multisine", {}),
    ("bangbang",            "bangbang",  {}),
    ("bangbang_slow",       "bangbang",  {"min_hold": 5, "max_hold": 50}),
    # ── OU variants ─────────────────────────────────────────────────────────
    # slow mean-reversion, low noise → long correlated sweeps
    ("ou_slow_low",         "ou",        {"theta": 0.05, "sigma": 1.0}),
    # default
    ("ou_default",          "ou",        {"theta": 0.15, "sigma": 2.5}),
    # medium reversion, high noise → energetic but still correlated
    ("ou_medium_high",      "ou",        {"theta": 0.30, "sigma": 4.0}),
    # fast reversion → nearly white noise
    ("ou_fast",             "ou",        {"theta": 1.00, "sigma": 4.0}),
    # very slow, full-range → wide, slow sweeps
    ("ou_wide_sweep",       "ou",        {"theta": 0.03, "sigma": 2.0}),
    # ── Mixed variants ──────────────────────────────────────────────────────
    # Default mixed (equal weight over 5 types)
    ("mixed_uniform",       "mixed",     {}),
    # Upweight multisine + ou, drop bangbang
    ("mixed_smooth",        "mixed",     {"weights": [0.10, 0.10, 0.30, 0.00, 0.50]}),
    # Upweight ou variants (simulated via equal weight but ou_medium_high params)
    ("mixed_ou_heavy",      "mixed",     {"weights": [0.10, 0.10, 0.50, 0.10, 0.20]}),
    # Heavy multisine bias — expected to match 1A distribution well
    ("mixed_multisine_heavy","mixed",    {"weights": [0.05, 0.05, 0.15, 0.05, 0.70]}),
    # All types equally, including bangbang for high-acceleration coverage
    ("mixed_all_equal",     "mixed",     {"weights": [0.20, 0.20, 0.20, 0.20, 0.20]}),
    # Drop gaussian + bangbang (poor performers)
    ("mixed_no_bang_gauss", "mixed",     {"weights": [0.25, 0.00, 0.50, 0.00, 0.25]}),
]


def run_sweep(cfg, n_trajectories: int, seed: int, output_dir: Path) -> list[dict]:
    results = []
    print(f"\n{'#':>3}  {'Config':<28} {'cov4d':>6} {'H4d':>6} {'score':>6}  elapsed")
    print("─" * 65)

    for idx, (name, atype, aparams) in enumerate(SWEEP_CONFIGS):
        t0 = time.time()
        states, _ = generate_random_rollout_data(
            cfg,
            n_trajectories,
            action_type=atype,
            seed=seed,
            action_params=aparams,
        )
        metrics = compute_4d_coverage(states)
        elapsed = time.time() - t0

        entry = {
            "name": name,
            "action_type": atype,
            "action_params": aparams,
            **metrics,
        }
        results.append(entry)
        print(f"{idx+1:>3}  {name:<28} {metrics['coverage']:>6.3f} {metrics['entropy']:>6.3f} "
              f"{metrics['score']:>6.3f}  {elapsed:.1f}s")

    return results


def plot_sweep_ranking(results: list[dict], output_dir: Path):
    results_sorted = sorted(results, key=lambda x: -x["score"])
    names = [r["name"] for r in results_sorted]
    covs  = [r["coverage"] for r in results_sorted]
    ents  = [r["entropy"]  for r in results_sorted]
    scores= [r["score"]    for r in results_sorted]

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, vals, title, color in [
        (axes[0], covs,   "4D Coverage (fraction of 10⁴ bins)",   "#4C72B0"),
        (axes[1], ents,   "4D Entropy (normalised Shannon)",        "#DD8452"),
        (axes[2], scores, "Composite Score (0.5×cov + 0.5×ent)",   "#55A868"),
    ]:
        bars = ax.bar(x, vals, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=7)

    best = results_sorted[0]
    fig.suptitle(
        f"Action Distribution Sweep — {len(results)} configs × {BINS_PER_DIM}⁴ bins\n"
        f"Best: '{best['name']}' (score={best['score']:.3f})",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    out = output_dir / "sweep_ranking.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved ranking chart to {out}")


def main():
    parser = argparse.ArgumentParser(description="Action distribution sweep")
    parser.add_argument("--config", type=str, default="configs/medium.yaml")
    parser.add_argument("--output-dir", type=str, default="outputs/action_sweep")
    parser.add_argument("--n-trajectories", type=int, default=1000,
                        help="Trajectories per config (more = slower but more accurate)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(SWEEP_CONFIGS)} configs × {args.n_trajectories} traj each")
    results = run_sweep(cfg, args.n_trajectories, args.seed, output_dir)

    # Save
    out_json = output_dir / "sweep_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_json}")

    plot_sweep_ranking(results, output_dir)

    # Print top-5 ranked
    results_sorted = sorted(results, key=lambda x: -x["score"])
    print(f"\n{'Rank':<5} {'Config':<28} {'type':<15} {'cov':>6} {'H':>6} {'score':>7}")
    print("─" * 72)
    for rank, r in enumerate(results_sorted[:10], 1):
        print(f"{rank:<5} {r['name']:<28} {r['action_type']:<15} "
              f"{r['coverage']:>6.3f} {r['entropy']:>6.3f} {r['score']:>7.3f}")

    best = results_sorted[0]
    print(f"\n★ Best config: '{best['name']}' — type={best['action_type']}, "
          f"params={best['action_params']}, score={best['score']:.4f}")


if __name__ == "__main__":
    main()
