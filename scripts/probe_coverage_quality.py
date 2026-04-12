#!/usr/bin/env python3
"""Stage 1B: Train quick probe models to test whether 4D coverage score predicts model quality.

For each of N action configs, trains a small MLP (256×3, 2k traj, 30 epochs) and
records final val_loss and OOD MSE. Then correlates 4D coverage/entropy/score with
model quality to answer: does coverage predict generalization?

Scientific question:
    Is 4D state-space coverage (or entropy) a reliable proxy for model quality?
    Or is there a coverage-learnability tradeoff (e.g. bangbang has high coverage
    but produces discontinuous dynamics that are hard to learn)?

Usage:
    python scripts/probe_coverage_quality.py \\
        --config configs/medium.yaml \\
        --sweep-results outputs/action_sweep/sweep_results.json \\
        --output-dir outputs/coverage_quality_probe \\
        --n-train 2000 --epochs 30 --device cuda:0

Outputs:
    outputs/coverage_quality_probe/
        probe_results.json      — per-config {coverage, entropy, score, val_loss, ood_mse}
        correlation_plot.png    — scatter: coverage_score vs val_loss / OOD_mse
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
from trackzero.data.dataset import TrajectoryDataset
from trackzero.data.ood_references import generate_ood_reference_data
from trackzero.data.random_rollout import generate_random_rollout_data
from trackzero.eval.harness import EvalHarness
from trackzero.policy.mlp import MLPPolicy, load_checkpoint
from trackzero.policy.train import TrainingConfig, train


def train_probe(cfg, action_type, action_params, n_train, epochs, val_states, val_actions,
                device, seed, output_dir, name):
    """Train a small probe model and return val_loss."""
    print(f"  Generating {n_train} traj ({action_type}, params={action_params})...", flush=True)
    t0 = time.time()
    train_states, train_actions = generate_random_rollout_data(
        cfg, n_train, action_type=action_type, seed=seed, action_params=action_params
    )
    print(f"    data gen: {time.time()-t0:.1f}s", flush=True)

    out = output_dir / name
    out.mkdir(parents=True, exist_ok=True)
    train_cfg = TrainingConfig(
        hidden_dim=256,
        n_hidden=3,
        batch_size=32768,
        lr=1e-3,
        epochs=epochs,
        seed=seed,
        output_dir=str(out),
    )
    t0 = time.time()
    model, logs = train(
        train_states, train_actions,
        val_states, val_actions,
        cfg=train_cfg,
        tau_max=cfg.pendulum.tau_max,
        device=device,
    )
    best_val = min(l.val_loss for l in logs)
    print(f"    train: {time.time()-t0:.1f}s  best_val={best_val:.4f}", flush=True)
    return best_val, model


def eval_ood_quick(model, cfg, ood_states, ood_actions, n_eval=100):
    """Evaluate a model on pre-generated OOD data. Returns mean MSE_total."""
    harness = EvalHarness(cfg)
    policy = MLPPolicy(model, tau_max=cfg.pendulum.tau_max)
    summary = harness.evaluate_policy(policy, ood_states, ood_actions, max_trajectories=n_eval)
    return summary.mean_mse_total


def plot_correlations(results, output_dir):
    """Scatter plots: coverage_score vs val_loss and OOD_mse."""
    scores   = [r["score"]   for r in results]
    covs     = [r["coverage"] for r in results]
    ents     = [r["entropy"]  for r in results]
    val_loss = [r["val_loss"] for r in results]
    ood_mse  = [r["ood_mse"]  for r in results]
    names    = [r["name"]     for r in results]
    atypes   = [r["action_type"] for r in results]

    # Color by action type
    type_colors = {
        "uniform": "#e6194b", "gaussian": "#f58231", "ou": "#3cb44b",
        "bangbang": "#4363d8", "multisine": "#911eb4", "mixed": "#42d4f4",
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    def scatter_ax(ax, x, y, xlabel, ylabel, title, log_y=True):
        for i, (xi, yi, nm, at) in enumerate(zip(x, y, names, atypes)):
            c = type_colors.get(at, "#aaaaaa")
            ax.scatter(xi, yi, color=c, s=80, zorder=3)
            ax.annotate(nm, (xi, yi), textcoords="offset points",
                        xytext=(4, 3), fontsize=6, color=c)
        # Pearson r
        xarr, yarr = np.array(x), np.array(y)
        if log_y:
            yarr_log = np.log(np.clip(yarr, 1e-10, None))
            r = np.corrcoef(xarr, yarr_log)[0, 1]
            ax.set_yscale("log")
            ax.set_title(f"{title}\nPearson r(x, log y) = {r:.3f}", fontsize=9)
        else:
            r = np.corrcoef(xarr, yarr)[0, 1]
            ax.set_title(f"{title}\nPearson r = {r:.3f}", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.3)
        # fit line
        if len(x) > 2:
            z = np.polyfit(xarr, np.log(np.clip(yarr, 1e-10, None)) if log_y else yarr, 1)
            xfit = np.linspace(min(xarr), max(xarr), 50)
            yfit = np.exp(np.polyval(z, xfit)) if log_y else np.polyval(z, xfit)
            ax.plot(xfit, yfit, "k--", alpha=0.4, linewidth=1)

    scatter_ax(axes[0, 0], scores,   val_loss, "Composite score (0.5×cov + 0.5×H)", "Val loss",    "Score → Val Loss")
    scatter_ax(axes[0, 1], scores,   ood_mse,  "Composite score", "Mean OOD MSE",   "Score → OOD MSE")
    scatter_ax(axes[0, 2], covs,     val_loss, "4D Coverage",  "Val loss",          "Coverage → Val Loss")
    scatter_ax(axes[1, 0], covs,     ood_mse,  "4D Coverage",  "Mean OOD MSE",      "Coverage → OOD MSE")
    scatter_ax(axes[1, 1], ents,     val_loss, "4D Entropy",   "Val loss",          "Entropy → Val Loss")
    scatter_ax(axes[1, 2], ents,     ood_mse,  "4D Entropy",   "Mean OOD MSE",      "Entropy → OOD MSE")

    # Legend
    handles = [plt.Line2D([0],[0], marker='o', color='w',
                          markerfacecolor=c, markersize=8, label=t)
               for t, c in type_colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8,
               title="Action type", bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Stage 1B: Does 4D Coverage Predict Model Quality?\n"
        f"{len(results)} action configs × 2k traj × 30 ep probe models (256×3 MLP)",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = output_dir / "correlation_plot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved correlation plot to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/medium.yaml")
    parser.add_argument("--val-data", type=str, default="data/medium/test.h5")
    parser.add_argument("--sweep-results", type=str,
                        default="outputs/action_sweep/sweep_results.json",
                        help="JSON from search_action_distribution.py — used to get coverage scores")
    parser.add_argument("--output-dir", type=str, default="outputs/coverage_quality_probe")
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-ood-eval", type=int, default=100,
                        help="OOD trajectories for quick evaluation (mixed_ood type)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sweep coverage results
    with open(args.sweep_results) as f:
        sweep = json.load(f)

    print(f"Loaded {len(sweep)} configs from sweep")
    print(f"Training {args.n_train} traj × {args.epochs} epochs probe models...")

    # Load validation data
    val_ds = TrajectoryDataset(args.val_data)
    val_states = val_ds.get_all_states()
    val_actions = val_ds.get_all_actions()
    val_ds.close()
    print(f"Val set: {len(val_states)} trajectories")

    # Pre-generate OOD evaluation data (mixed: chirp+step+random_walk+sawtooth+pulse)
    print(f"Generating {args.n_ood_eval} mixed OOD reference trajectories...")
    ood_states, ood_actions = generate_ood_reference_data(
        cfg, args.n_ood_eval, action_type="mixed_ood", seed=args.seed
    )
    print(f"  OOD states: {ood_states.shape}")

    results = []
    for entry in sweep:
        name = entry["name"]
        atype = entry["action_type"]
        aparams = entry.get("action_params", {})
        coverage = entry["coverage"]
        entropy = entry["entropy"]
        score = entry["score"]

        print(f"\n[{name}] type={atype} score={score:.3f}")
        val_loss, model = train_probe(
            cfg, atype, aparams, args.n_train, args.epochs,
            val_states, val_actions,
            args.device, args.seed, output_dir, name,
        )
        ood_mse = eval_ood_quick(model, cfg, ood_states, ood_actions, n_eval=args.n_ood_eval)
        print(f"  → val_loss={val_loss:.5f}  ood_mse={ood_mse:.5e}")

        results.append({
            "name": name,
            "action_type": atype,
            "action_params": aparams,
            "coverage": coverage,
            "entropy": entropy,
            "score": score,
            "val_loss": val_loss,
            "ood_mse": ood_mse,
        })

    # Save
    with open(output_dir / "probe_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot correlations
    plot_correlations(results, output_dir)

    # Print ranked by OOD MSE
    ranked = sorted(results, key=lambda x: x["ood_mse"])
    print(f"\n{'Rank':<5} {'Config':<28} {'type':<12} {'score':>7} {'val_loss':>10} {'ood_mse':>12}")
    print("─" * 80)
    for rank, r in enumerate(ranked, 1):
        print(f"{rank:<5} {r['name']:<28} {r['action_type']:<12} "
              f"{r['score']:>7.3f} {r['val_loss']:>10.5f} {r['ood_mse']:>12.4e}")

    best = ranked[0]
    print(f"\n★ Best for OOD: '{best['name']}' "
          f"(type={best['action_type']}, params={best['action_params']}, "
          f"ood_mse={best['ood_mse']:.4e})")


if __name__ == "__main__":
    main()
