"""Pendulum animation with optional reference overlay."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def _pendulum_xy(q1: float, q2: float, L: float = 0.5) -> tuple:
    """Compute (x, y) positions of pivot, joint, and tip."""
    # Joint 1 swings from pivot (0, 0)
    x1 = L * np.sin(q1)
    y1 = -L * np.cos(q1)

    # Joint 2 swings from end of link 1
    x2 = x1 + L * np.sin(q1 + q2)
    y2 = y1 - L * np.cos(q1 + q2)

    return (0, x1, x2), (0, y1, y2)


def animate_pendulum(
    states: np.ndarray,
    control_dt: float,
    link_length: float = 0.5,
    ref_states: Optional[np.ndarray] = None,
    save_path: Optional[str | Path] = None,
    fps: int = 30,
    skip: int = 1,
) -> animation.FuncAnimation:
    """Create an animation of the double pendulum.

    Args:
        states: (T+1, 4) state trajectory.
        control_dt: Time between control steps.
        link_length: Length of each link.
        ref_states: Optional reference trajectory for overlay.
        save_path: If given, save as MP4 or GIF.
        fps: Frames per second.
        skip: Show every skip-th frame (for speed).

    Returns:
        matplotlib FuncAnimation object.
    """
    L = link_length
    total_L = 2 * L

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-total_L * 1.2, total_L * 1.2)
    ax.set_ylim(-total_L * 1.2, total_L * 1.2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("Double Pendulum")

    # Actual pendulum
    (line,) = ax.plot([], [], "o-", color="steelblue", linewidth=2, markersize=6)
    trail_x, trail_y = [], []
    (trail,) = ax.plot([], [], "-", color="steelblue", alpha=0.2, linewidth=0.5)

    # Reference pendulum (if provided)
    ref_line = None
    if ref_states is not None:
        (ref_line,) = ax.plot([], [], "o-", color="red", linewidth=1.5, markersize=4, alpha=0.5)

    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10)

    frames = range(0, len(states), skip)

    def init():
        line.set_data([], [])
        trail.set_data([], [])
        time_text.set_text("")
        if ref_line:
            ref_line.set_data([], [])
        return (line, trail, time_text) + ((ref_line,) if ref_line else ())

    def update(frame):
        q1, q2 = states[frame, 0], states[frame, 1]
        xs, ys = _pendulum_xy(q1, q2, L)
        line.set_data(xs, ys)

        trail_x.append(xs[-1])
        trail_y.append(ys[-1])
        trail.set_data(trail_x, trail_y)

        t = frame * control_dt
        time_text.set_text(f"t = {t:.2f}s")

        artists = [line, trail, time_text]

        if ref_line is not None and frame < len(ref_states):
            rq1, rq2 = ref_states[frame, 0], ref_states[frame, 1]
            rxs, rys = _pendulum_xy(rq1, rq2, L)
            ref_line.set_data(rxs, rys)
            artists.append(ref_line)

        return tuple(artists)

    anim = animation.FuncAnimation(
        fig, update, frames=frames, init_func=init,
        blit=True, interval=1000 / fps,
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix == ".gif":
            anim.save(str(save_path), writer="pillow", fps=fps)
        else:
            anim.save(str(save_path), writer="ffmpeg", fps=fps)

    plt.close(fig)
    return anim
