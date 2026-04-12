#!/usr/bin/env python3
"""Plot learning curves from training logs.

Usage:
    python scripts/plot_learning_curve.py \
        --logs "Stage 1A:outputs/stage1a_scaled/training_log.json" \
               "Stage 1B:outputs/stage1b_scaled/training_log.json" \
        --output outputs/learning_curves.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_log(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Plot learning curves from training logs")
    parser.add_argument("--logs", nargs="+", required=True,
                        help="Training logs in 'Label:path.json' format")
    parser.add_argument("--output", type=str, default="outputs/learning_curves.png",
                        help="Output path for the figure")
    parser.add_argument("--metric", type=str, default="val_loss",
                        choices=["val_loss", "train_loss"],
                        help="Which metric to plot (default: val_loss)")
    parser.add_argument("--log-scale", action="store_true",
                        help="Use log scale for y-axis")
    parser.add_argument("--smooth", type=int, default=1,
                        help="Rolling window size for smoothing (default: 1 = no smoothing)")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_val, ax_train = axes

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, log_spec in enumerate(args.logs):
        parts = log_spec.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Expected 'Label:path', got: {log_spec}")
        label, path = parts

        logs = load_log(path)
        epochs = [l["epoch"] for l in logs]
        val_losses = [l["val_loss"] for l in logs]
        train_losses = [l["train_loss"] for l in logs]

        def smooth(values, w):
            if w <= 1:
                return values
            arr = np.array(values)
            out = np.convolve(arr, np.ones(w) / w, mode="valid")
            return out.tolist()

        smooth_val = smooth(val_losses, args.smooth)
        smooth_train = smooth(train_losses, args.smooth)
        smooth_epochs = epochs[args.smooth - 1:]

        color = colors[i % len(colors)]

        ax_val.plot(smooth_epochs, smooth_val, label=label, color=color, linewidth=2)
        ax_train.plot(smooth_epochs, smooth_train, label=label, color=color, linewidth=2)

        # Mark best val epoch
        best_idx = int(np.argmin(val_losses))
        ax_val.scatter([epochs[best_idx]], [val_losses[best_idx]],
                       color=color, s=80, zorder=5, marker="*")

        # Print summary
        print(f"{label}: best_val={min(val_losses):.6f} (epoch {epochs[best_idx]}), "
              f"final_val={val_losses[-1]:.6f}")

    for ax, title in [(ax_val, "Validation Loss"), (ax_train, "Training Loss")]:
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("MSE Loss", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        if args.log_scale:
            ax.set_yscale("log")

    fig.suptitle("Learning Curves", fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved learning curves to {out_path}")


if __name__ == "__main__":
    main()
