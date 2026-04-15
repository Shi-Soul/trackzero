#!/usr/bin/env python3
"""Humanoid stick-figure visualization from saved checkpoints.

Loads already-saved humanoid policies and compares them with oracle and
zero-torque rollouts on the same benchmark reference, without training.

Outputs to outputs/viz_sim/:
  humanoid_oracle.gif
  humanoid_zerotorque.gif
  humanoid_horizon.gif
  humanoid_dagger_round0.gif
  humanoid_compare.gif

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.visualize_humanoid
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import mujoco
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_humanoid import (
    HUMANOID_XML, TAU_MAX,
    mj_to_flat, flat_to_mj, get_contact_flags, gen_benchmark,
)
from scripts.models_structured import _build_mlp

OUT = ROOT / "outputs" / "viz_sim"
OUT.mkdir(parents=True, exist_ok=True)

TAU_MAX_NP = np.array(TAU_MAX, dtype=np.float32)  # per-joint limits
FPS = 15
T_ROLLOUT = 300
SEED = 42

CHECKPOINTS = {
    "horizon": ROOT / "outputs/humanoid_horizon/model.pt",
    "dagger_round0": ROOT / "outputs/humanoid_dagger/model_round0.pt",
}

# ── body connectivity for stick-figure drawing ──────────────────────
# Each tuple (parent_id, child_id) draws one limb segment
LIMB_EDGES = [
    (1, 2), (2, 3),              # torso → lwaist → pelvis
    (3, 4), (4, 5), (5, 6),     # right leg
    (3, 7), (7, 8), (8, 9),     # left leg
    (1, 10), (10, 11),           # right arm
    (1, 12), (12, 13),           # left arm
]
JOINT_COLORS = {
    "right": "#e05050",
    "left":  "#5080e0",
    "torso": "#505050",
}


def _body_color(i: int) -> str:
    name = mujoco.mj_id2name(
        mujoco.MjModel.from_xml_string(HUMANOID_XML),
        mujoco.mjtObj.mjOBJ_BODY, i
    ) or ""
    if "right" in name:
        return JOINT_COLORS["right"]
    if "left" in name:
        return JOINT_COLORS["left"]
    return JOINT_COLORS["torso"]


def load_plain_mlp(ckpt_path: Path, dev: str):
    ckpt = torch.load(str(ckpt_path), map_location=dev, weights_only=False)
    in_dim = ckpt["net.0.weight"].shape[1]
    hidden = ckpt["net.0.weight"].shape[0]
    n_linear = len([k for k in ckpt if k.startswith("net.") and k.endswith(".weight")])
    layers = max(1, n_linear - 1)
    out_dim = ckpt["net.8.bias"].shape[0]
    model = _build_mlp(in_dim, out_dim, hidden, layers).to(dev)
    model_sd = {
        k.removeprefix("net."): v
        for k, v in ckpt.items()
        if k.startswith("net.")
    }
    model.load_state_dict(model_sd)
    model.eval()

    norm_m = ckpt["norm_m"]
    norm_s = ckpt["norm_s"]

    class Wrapped:
        def __call__(self, x):
            x = (x - norm_m.to(x.device)) / (norm_s.to(x.device) + 1e-8)
            return model(x)

    return Wrapped()


# ── rollout functions ─────────────────────────────────────────────────

def rollout_oracle(ref_s: np.ndarray, ref_a: np.ndarray, T: int):
    """Apply oracle torques (mj_inverse) at each step."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    flat_to_mj(ref_s[0], m, d)
    mujoco.mj_forward(m, d)
    xpos_hist = []
    for t in range(min(T, len(ref_a))):
        xpos_hist.append(d.xpos.copy())
        tau = np.clip(ref_a[t], -TAU_MAX_NP, TAU_MAX_NP)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
    return np.array(xpos_hist)   # (T, nbody, 3)


def rollout_policy(model_nn, ref_s: np.ndarray, T: int, dev: str):
    """Apply learned policy."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    flat_to_mj(ref_s[0], m, d)
    mujoco.mj_forward(m, d)
    xpos_hist = []
    for t in range(min(T, len(ref_s) - 1)):
        xpos_hist.append(d.xpos.copy())
        s = mj_to_flat(d.qpos.copy(), d.qvel.copy())
        flags = get_contact_flags(d)
        inp = np.concatenate([s, ref_s[t + 1], flags])
        with torch.no_grad():
            tau = model_nn(
                torch.tensor(inp, dtype=torch.float32)
                .unsqueeze(0).to(dev)).cpu().numpy()[0]
        d.ctrl[:] = np.clip(tau, -TAU_MAX_NP, TAU_MAX_NP)
        mujoco.mj_step(m, d)
    return np.array(xpos_hist)


def rollout_zero(ref_s: np.ndarray, T: int):
    """Apply zero torques (humanoid falls)."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    flat_to_mj(ref_s[0], m, d)
    mujoco.mj_forward(m, d)
    xpos_hist = []
    for t in range(T):
        xpos_hist.append(d.xpos.copy())
        d.ctrl[:] = 0.0
        mujoco.mj_step(m, d)
    return np.array(xpos_hist)


# ── stick figure rendering ────────────────────────────────────────────

def _draw_frame(ax, xpos_frame: np.ndarray, color: str, alpha: float = 1.0):
    """Draw one stick-figure frame on ax (XZ projection)."""
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.3, 2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    for pi, ci in LIMB_EDGES:
        x = [xpos_frame[pi, 0], xpos_frame[ci, 0]]
        z = [xpos_frame[pi, 2], xpos_frame[ci, 2]]
        ax.plot(x, z, "-", color=color, linewidth=2.5, alpha=alpha)
    for i in range(1, xpos_frame.shape[0]):
        ax.plot(xpos_frame[i, 0], xpos_frame[i, 2],
                "o", markersize=5, color=color, alpha=alpha)
    # ground line
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")


def make_gif(xpos_hist: np.ndarray, title: str, color: str,
             save_path: Path, fps: int = FPS):
    """Single-policy stick figure GIF."""
    fig, ax = plt.subplots(figsize=(3, 4))
    fig.patch.set_facecolor("#f8f8f8")

    def animate(t: int):
        ax.clear()
        _draw_frame(ax, xpos_hist[t], color)
        ax.set_title(f"{title}  t={t * 0.002:.2f}s", fontsize=8)

    ani = animation.FuncAnimation(
        fig, animate, frames=len(xpos_hist),
        interval=int(1000 / fps), blit=False)
    ani.save(str(save_path), writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Saved {save_path.name}  ({len(xpos_hist)} frames)")


def make_compare_gif(
    histories: list[tuple[np.ndarray, str, str]],
    ref_s: np.ndarray,
    save_path: Path,
    fps: int = FPS,
):
    """Side-by-side comparison GIF."""
    n = len(histories)
    T = min(len(h) for h, _, _ in histories)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4.5))
    fig.patch.set_facecolor("#f8f8f8")
    fig.suptitle("TRACK-ZERO: Humanoid Saved-Model Replay", fontsize=10, y=0.98)

    # Compute reference base height trace for overlay
    ref_height = []
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    for t in range(min(T, len(ref_s))):
        flat_to_mj(ref_s[t], m, d)
        mujoco.mj_forward(m, d)
        ref_height.append(d.xpos[1, 2])  # torso height

    def animate(t: int):
        for ax, (hist, label, col) in zip(axes, histories):
            ax.clear()
            _draw_frame(ax, hist[t], col)
            # Show height bar
            h_curr = hist[t][1, 2]  # torso height
            status = "✓ upright" if h_curr > 0.8 else "✗ fallen"
            ax.set_title(f"{label}\n{status}  h={h_curr:.2f}m", fontsize=8)
        fig.text(0.5, 0.01, f"t = {t * 0.002:.2f}s", ha="center", fontsize=8)

    ani = animation.FuncAnimation(
        fig, animate, frames=T,
        interval=int(1000 / fps), blit=False)
    ani.save(str(save_path), writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Saved {save_path.name}  ({T} frames)")


# ── entry point ───────────────────────────────────────────────────────

def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")

    # ── Get benchmark reference trajectory ──
    print("\n[1/3] Loading benchmark reference...")
    families = gen_benchmark(1, T_ROLLOUT + 1, seed=SEED)
    fam_key = list(families.keys())[0]
    ref_s, ref_a = families[fam_key][0]
    T = min(T_ROLLOUT, len(ref_a))
    print(f"  Reference family: {fam_key}, T={T}")

    print("[2/3] Loading saved humanoid policies...")
    policies = {}
    for name, ckpt in CHECKPOINTS.items():
        if ckpt.exists():
            policies[name] = load_plain_mlp(ckpt, dev)
            print(f"  Loaded {name}: {ckpt}")
        else:
            print(f"  SKIP {name}: missing checkpoint {ckpt}")

    # ── Run rollouts ──
    print("[3/3] Running rollouts...")
    print("  oracle...")
    hist_oracle = rollout_oracle(ref_s, ref_a, T)
    print("  zero torque...")
    hist_zero = rollout_zero(ref_s, T)
    histories = [(hist_oracle, "Oracle", "#2ca02c")]
    stats = {
        "oracle_avg_height": float(np.mean(hist_oracle[:, 1, 2])),
        "zero_torque_avg_height": float(np.mean(hist_zero[:, 1, 2])),
        "T": T,
        "reference_family": fam_key,
    }
    model_rollouts = {}
    for name, policy in policies.items():
        print(f"  {name}...")
        hist = rollout_policy(policy, ref_s, T, dev)
        model_rollouts[name] = hist
        histories.append((hist, name.replace("_", " ").title(), "#1f77b4" if name == "horizon" else "#9467bd"))
        stats[f"{name}_avg_height"] = float(np.mean(hist[:, 1, 2]))

    # ── Save individual GIFs ──
    print("\nSaving GIFs...")
    make_gif(hist_oracle, "Oracle (upper bound)", "#2ca02c",
             OUT / "humanoid_oracle.gif")
    make_gif(hist_zero, "Zero torque (lower bound)", "#d62728",
             OUT / "humanoid_zerotorque.gif")
    for name, hist in model_rollouts.items():
        make_gif(hist, f"Saved model: {name}",
                 "#1f77b4" if name == "horizon" else "#9467bd",
                 OUT / f"humanoid_{name}.gif")

    # ── Save comparison GIF ──
    make_compare_gif(histories[:4], ref_s, save_path=OUT / "humanoid_compare.gif")

    with open(OUT / "humanoid_viz_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats: {stats}")
    print("Done!")


if __name__ == "__main__":
    main()
