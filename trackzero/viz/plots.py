"""Diagnostic plots: coverage, error histograms, trajectory diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trackzero.eval.harness import EvalSummary


def plot_coverage(
    q_hist: np.ndarray,
    v_hist: np.ndarray,
    q_edges: np.ndarray,
    v_edges: np.ndarray,
    q_coverage: float,
    v_coverage: float,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot state-space coverage heatmaps."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Position coverage
    ax = axes[0]
    im = ax.pcolormesh(q_edges, q_edges, np.log1p(q_hist.T), cmap="viridis")
    ax.set_xlabel("q1 (rad)")
    ax.set_ylabel("q2 (rad)")
    ax.set_title(f"Position Coverage ({q_coverage:.1%} bins occupied)")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="log(1 + count)")

    # Velocity coverage
    ax = axes[1]
    im = ax.pcolormesh(v_edges, v_edges, np.log1p(v_hist.T), cmap="viridis")
    ax.set_xlabel("dq1 (rad/s)")
    ax.set_ylabel("dq2 (rad/s)")
    ax.set_title(f"Velocity Coverage ({v_coverage:.1%} bins occupied)")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="log(1 + count)")

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_error_histogram(
    summary: EvalSummary,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot histogram and CDF of tracking errors."""
    mse_q = [r.mse_q for r in summary.results]
    mse_v = [r.mse_v for r in summary.results]
    mse_total = [r.mse_total for r in summary.results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, data, label in zip(
        axes, [mse_q, mse_v, mse_total], ["MSE_q", "MSE_v", "MSE_total"]
    ):
        ax.hist(data, bins=50, density=True, alpha=0.7, color="steelblue")
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_title(f"{label} Distribution")
        ax.axvline(np.mean(data), color="red", linestyle="--", label=f"Mean={np.mean(data):.2e}")
        ax.axvline(np.median(data), color="orange", linestyle="--", label=f"Median={np.median(data):.2e}")
        ax.legend(fontsize=8)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_error_cdf(
    summary: EvalSummary,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot CDF of tracking errors."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for data, label, color in [
        ([r.mse_q for r in summary.results], "MSE_q", "blue"),
        ([r.mse_v for r in summary.results], "MSE_v", "green"),
        ([r.mse_total for r in summary.results], "MSE_total", "red"),
    ]:
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, cdf, label=label, color=color)

    ax.set_xlabel("Error")
    ax.set_ylabel("CDF")
    ax.set_title("Tracking Error CDF")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_trajectory_diagnostics(
    ref_states: np.ndarray,
    actual_states: np.ndarray,
    control_dt: float,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot reference vs actual trajectory for a single rollout."""
    T = len(ref_states) - 1
    times = np.arange(T + 1) * control_dt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = ["q1", "q2", "dq1", "dq2"]
    units = ["rad", "rad", "rad/s", "rad/s"]

    for i, (ax, label, unit) in enumerate(zip(axes.flat, labels, units)):
        ax.plot(times, ref_states[:, i], "b-", label="Reference", alpha=0.7)
        ax.plot(times, actual_states[:, i], "r--", label="Actual", alpha=0.7)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{label} ({unit})")
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_policy_comparison(
    summaries: list[tuple[str, "EvalSummary"]],
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Compare multiple policies: CDF + bar chart side by side.

    Args:
        summaries: list of (label, EvalSummary) pairs
        save_path: optional save path
    """
    colors = ["steelblue", "darkorange", "seagreen", "crimson", "purple"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- left: CDF of MSE_total on log scale ---
    ax = axes[0]
    for (label, summary), color in zip(summaries, colors):
        data = np.sort([r.mse_total for r in summary.results])
        cdf = np.arange(1, len(data) + 1) / len(data)
        ax.plot(data, cdf, label=label, color=color, linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("MSE_total (log scale)")
    ax.set_ylabel("CDF")
    ax.set_title("MSE_total CDF — policy comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- right: bar chart of mean metrics ---
    ax = axes[1]
    metric_labels = ["Mean\nMSE_q", "Mean\nMSE_v", "Mean\nMSE_total"]
    x = np.arange(len(metric_labels))
    width = 0.8 / len(summaries)
    offsets = np.linspace(-(len(summaries) - 1) / 2, (len(summaries) - 1) / 2, len(summaries)) * width
    for (label, summary), color, offset in zip(summaries, colors, offsets):
        values = [summary.mean_mse_q, summary.mean_mse_v, summary.mean_mse_total]
        ax.bar(x + offset, values, width=width, label=label, color=color, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Error (log scale)")
    ax.set_title("Mean tracking error by policy")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_oracle_verification(
    errors: np.ndarray,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot oracle torque recovery errors."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(errors.ravel(), bins=100, density=True, color="steelblue", alpha=0.7)
    ax.set_xlabel("Torque Error (Nm)")
    ax.set_ylabel("Density")
    ax.set_title(f"Oracle Error Distribution (max={np.max(np.abs(errors)):.2e})")

    ax = axes[1]
    ax.semilogy(np.sort(np.abs(errors.ravel())), ".", markersize=1)
    ax.set_xlabel("Sorted Index")
    ax.set_ylabel("|Error| (Nm)")
    ax.set_title("Sorted Absolute Errors")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
