#!/usr/bin/env python3
"""TRACK-ZERO comprehensive simulation visualization.

Generates GIFs showing closed-loop tracking for all trained models:

  2-DOF double pendulum:
    oracle, stage1a, stage1b, stage1d_best, stage1d_50k,
    stage2c_noisy, residual (oracle-input)

  N-link chains:
    3-link (chain_3link_10k), 5-link (chain_5link_10k)

Reference types: step, chirp, random_walk (2-DOF); step, chirp (chains)
Output: outputs/viz_sim/*.gif
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from trackzero.config import load_config
from trackzero.data.ood_references import (
    generate_step_actions,
    generate_chirp_actions,
    generate_random_walk_actions,
)
from trackzero.oracle.inverse_dynamics import InverseDynamicsOracle
from trackzero.policy.mlp import InverseDynamicsMLP, load_checkpoint, MLPPolicy
from trackzero.sim.pendulum_model import build_chain_xml
from trackzero.sim.simulator import Simulator

OUT = ROOT / "outputs" / "viz_sim"
OUT.mkdir(parents=True, exist_ok=True)

T_VIZ = 250      # timesteps per rollout (2.5 s at dt=0.01)
FPS = 25
LINK_L = 0.5     # link length for rendering
SEED = 42


# ---------------------------------------------------------------------------
# Policy wrappers
# ---------------------------------------------------------------------------

def make_std_policy(ckpt_path: Path, tau_max: float, device: str = "cpu") -> MLPPolicy:
    model = load_checkpoint(str(ckpt_path), device=device)
    return MLPPolicy(model, tau_max=tau_max, device=device)


class ResidualPolicy:
    """oracle-augmented input residual policy (10D input)."""
    def __init__(self, ckpt_path: Path, oracle: InverseDynamicsOracle,
                 tau_max: float, device: str = "cpu"):
        ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        sd = ck["model_state_dict"]
        # infer dims from state dict
        in_dim = sd["net.0.weight"].shape[1]
        out_dim = sd[list(sd.keys())[-2]].shape[0]  # last weight row count
        hdim = sd["net.0.weight"].shape[0]
        n_hidden = sum(1 for k in sd if k.startswith("net.") and k.endswith(".weight"))
        model = InverseDynamicsMLP(
            state_dim=in_dim // 2, action_dim=out_dim,
            hidden_dim=hdim, n_hidden=n_hidden - 1
        )
        # directly load raw state dict (input_mean/std keys included)
        model.load_state_dict(sd, strict=True)
        model.eval().to(device)
        self._model = model
        self._oracle = oracle
        self._tau = tau_max
        self._dev = device

    def __call__(self, cur_state: np.ndarray, ref_next: np.ndarray) -> np.ndarray:
        orc = np.clip(
            self._oracle.compute_torque(np.asarray(cur_state, float),
                                        np.asarray(ref_next, float)),
            -self._tau, self._tau
        )
        x = np.concatenate([cur_state, ref_next, orc]).astype(np.float32)
        with torch.no_grad():
            out = self._model(torch.from_numpy(x).unsqueeze(0).to(self._dev))
        return np.clip(out.squeeze(0).cpu().numpy(), -self._tau, self._tau)


class ChainPolicy:
    """Policy for N-link chain, loaded from chain checkpoint format."""
    def __init__(self, ckpt_path: Path, tau_max: float, device: str = "cpu"):
        ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        n_links = ck["n_links"]
        hdim = ck["hdim"]
        nlayer = ck["nlayer"]
        input_dim = 4 * n_links
        model = InverseDynamicsMLP(
            state_dim=input_dim // 2,
            action_dim=n_links,
            hidden_dim=hdim,
            n_hidden=nlayer
        )
        model.load_state_dict(ck["model_state_dict"])
        model.eval().to(device)
        self._model = model
        self._n_links = n_links
        self._tau = tau_max
        self._dev = device

    @property
    def n_links(self) -> int:
        return self._n_links

    def __call__(self, cur_state: np.ndarray, ref_next: np.ndarray) -> np.ndarray:
        x = np.concatenate([cur_state, ref_next]).astype(np.float32)
        with torch.no_grad():
            out = self._model(torch.from_numpy(x).unsqueeze(0).to(self._dev))
        return np.clip(out.squeeze(0).cpu().numpy(), -self._tau, self._tau)


# ---------------------------------------------------------------------------
# Reference trajectory generation
# ---------------------------------------------------------------------------

def make_ref_2dof(action_type: str, cfg, seed: int = SEED) -> np.ndarray:
    """Return reference state trajectory (T+1, 4) for double pendulum."""
    rng = np.random.default_rng(seed)
    T = T_VIZ
    tau_max = cfg.pendulum.tau_max
    dt = cfg.simulation.control_dt

    generators = dict(
        step=generate_step_actions,
        chirp=generate_chirp_actions,
        random_walk=generate_random_walk_actions,
    )
    gen_fn = generators[action_type]
    actions = gen_fn(rng=rng, T=T, n_joints=2, tau_max=tau_max, dt=dt)

    sim = Simulator(cfg)
    q0 = rng.uniform(-1.0, 1.0, 2)
    v0 = rng.uniform(-1.0, 1.0, 2)
    states = sim.rollout(actions, q0=q0, v0=v0)
    return states[:T + 1]


def make_ref_chain(action_type: str, chain_sim, tau_max: float,
                   nq: int, seed: int = SEED) -> np.ndarray:
    """Return reference state trajectory (T+1, 2*nq) for N-link chain."""
    rng = np.random.default_rng(seed)
    T = T_VIZ
    dt = 0.01
    q0 = rng.uniform(-1.5, 1.5, nq)
    v0 = rng.uniform(-1.0, 1.0, nq)
    if action_type == "step":
        actions = generate_step_actions(rng=rng, T=T, n_joints=nq, tau_max=tau_max, dt=dt)
    else:  # chirp
        actions = generate_chirp_actions(rng=rng, T=T, n_joints=nq, tau_max=tau_max, dt=dt)
    return chain_sim.rollout(actions, q0, v0)[:T + 1]


# ---------------------------------------------------------------------------
# Closed-loop rollout
# ---------------------------------------------------------------------------

def rollout_2dof(policy, ref_states: np.ndarray, cfg) -> np.ndarray:
    sim = Simulator(cfg)
    T = min(len(ref_states) - 1, T_VIZ)
    q0 = ref_states[0, :2]
    v0 = ref_states[0, 2:]
    sim.reset(q0=q0, v0=v0)
    actual = np.zeros((T + 1, 4))
    actual[0] = sim.get_state()
    for t in range(T):
        u = policy(actual[t], ref_states[t + 1])
        actual[t + 1] = sim.step(u)
    return actual


def rollout_chain(policy, ref_states: np.ndarray, chain_sim) -> np.ndarray:
    nq = chain_sim.nq
    T = min(len(ref_states) - 1, T_VIZ)
    chain_sim.reset(ref_states[0, :nq], ref_states[0, nq:])
    actual = np.zeros((T + 1, 2 * nq))
    actual[0] = chain_sim.state()
    for t in range(T):
        u = policy(actual[t], ref_states[t + 1])
        actual[t + 1] = chain_sim.step(u)
    return actual


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def pendulum_xy_2dof(q: np.ndarray, L: float = LINK_L):
    x1 = L * np.sin(q[0])
    y1 = -L * np.cos(q[0])
    x2 = x1 + L * np.sin(q[0] + q[1])
    y2 = y1 - L * np.cos(q[0] + q[1])
    return [0, x1, x2], [0, y1, y2]


def chain_xy(q: np.ndarray, L: float = LINK_L):
    xs, ys = [0.0], [0.0]
    cum = 0.0
    for qi in q:
        cum += qi
        xs.append(xs[-1] + L * np.sin(cum))
        ys.append(ys[-1] - L * np.cos(cum))
    return xs, ys


# ---------------------------------------------------------------------------
# Animation builders
# ---------------------------------------------------------------------------

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#bcbd22",
]


def _make_pendulum_gif(
    label: str,
    ref_states: np.ndarray,
    rollouts: dict[str, np.ndarray],
    xy_fn,
    n_links: int,
    save_path: Path,
    title: str,
) -> None:
    """Animate multiple policy rollouts vs reference (planar pendulum/chain)."""
    T = min(len(ref_states) - 1, T_VIZ)
    total_L = n_links * LINK_L

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-total_L * 1.3, total_L * 1.3)
    ax.set_ylim(-total_L * 1.3, total_L * 1.3)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=10)

    # Reference line
    (ref_line,) = ax.plot([], [], "k--", lw=1.5, label="Reference", alpha=0.8)

    # Policy lines
    policy_lines = {}
    for i, (name, _) in enumerate(rollouts.items()):
        c = PALETTE[i % len(PALETTE)]
        (ln,) = ax.plot([], [], "o-", color=c, lw=2, markersize=4, label=name)
        policy_lines[name] = ln

    ax.legend(loc="upper right", fontsize=7, ncol=2)
    time_txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9)

    def init():
        ref_line.set_data([], [])
        for ln in policy_lines.values():
            ln.set_data([], [])
        time_txt.set_text("")
        return [ref_line, time_txt] + list(policy_lines.values())

    def update(frame):
        q_ref = ref_states[frame, :n_links]
        rxs, rys = xy_fn(q_ref)
        ref_line.set_data(rxs, rys)
        for name, traj in rollouts.items():
            q = traj[frame, :n_links]
            xs, ys = xy_fn(q)
            policy_lines[name].set_data(xs, ys)
        time_txt.set_text(f"t = {frame * 0.01:.2f}s")
        return [ref_line, time_txt] + list(policy_lines.values())

    anim = animation.FuncAnimation(
        fig, update, frames=range(T + 1), init_func=init,
        blit=True, interval=1000 / FPS
    )
    anim.save(str(save_path), writer="pillow", fps=FPS)
    plt.close(fig)
    print(f"  Saved: {save_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# ChainSim (reused from run_chain_experiment)
# ---------------------------------------------------------------------------

import mujoco

class ChainSim:
    def __init__(self, n_links: int, cfg):
        xml = build_chain_xml(n_links, cfg.pendulum, cfg.simulation)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.nq = self.model.nq
        self.nv = self.model.nv
        self._sub = cfg.simulation.substeps
        self._tau = cfg.pendulum.tau_max

    def reset(self, q0, v0):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = q0
        self.data.qvel[:] = v0
        mujoco.mj_forward(self.model, self.data)

    def state(self):
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, -self._tau, self._tau)
        for _ in range(self._sub):
            mujoco.mj_step(self.model, self.data)
        return self.state()

    def rollout(self, actions, q0, v0):
        self.reset(q0, v0)
        T = len(actions)
        sd = self.nq + self.nv
        states = np.zeros((T + 1, sd))
        states[0] = self.state()
        for t in range(T):
            states[t + 1] = self.step(actions[t])
        return states


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CHECKPOINTS_2DOF = {
    "stage1a":       ROOT / "outputs/stage1a/best_model.pt",
    "stage1b":       ROOT / "outputs/stage1b/best_model.pt",
    "stage1d_best":  ROOT / "outputs/arch_1024x6_wd0.0001_lr_cosine_10k/best_model.pt",
    "stage1d_50k":   ROOT / "outputs/random_50k_1024x6/best_model.pt",
    "stage2c_noisy": ROOT / "outputs/stage2c_noise0.05_frac0.5/best_model.pt",
}

CHECKPOINTS_RESIDUAL = ROOT / "outputs/residual_10k/best_model.pt"

CHECKPOINTS_CHAIN = {
    3: ROOT / "outputs/chain_3link_10k/best_model.pt",
    5: ROOT / "outputs/chain_5link_10k/best_model.pt",
}

REF_TYPES_2DOF = ["step", "chirp", "random_walk"]
REF_TYPES_CHAIN = ["step", "chirp"]


def main():
    cfg = load_config()
    tau_max = cfg.pendulum.tau_max
    device = "cpu"

    print("=" * 60)
    print("TRACK-ZERO Simulation Visualization")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load 2-DOF policies
    # ------------------------------------------------------------------
    print("\n[1/3] Loading 2-DOF policies...")

    oracle = InverseDynamicsOracle(cfg)
    oracle_policy = oracle.as_policy("shooting")

    policies_2dof: dict[str, object] = {"oracle": oracle_policy}

    for name, ckpt in CHECKPOINTS_2DOF.items():
        if not ckpt.exists():
            print(f"  SKIP {name}: checkpoint not found ({ckpt})")
            continue
        policies_2dof[name] = make_std_policy(ckpt, tau_max, device)
        print(f"  Loaded {name}")

    if CHECKPOINTS_RESIDUAL.exists():
        policies_2dof["residual"] = ResidualPolicy(
            CHECKPOINTS_RESIDUAL, oracle, tau_max, device
        )
        print("  Loaded residual")

    # ------------------------------------------------------------------
    # 2. Generate 2-DOF GIFs (one per reference type)
    # ------------------------------------------------------------------
    print("\n[2/3] Generating 2-DOF double pendulum GIFs...")

    for ref_type in REF_TYPES_2DOF:
        print(f"\n  Reference: {ref_type}")
        ref_states = make_ref_2dof(ref_type, cfg, seed=SEED)

        rollouts: dict[str, np.ndarray] = {}
        for name, policy in policies_2dof.items():
            try:
                traj = rollout_2dof(policy, ref_states, cfg)
                rollouts[name] = traj
                print(f"    Rollout {name}: OK")
            except Exception as exc:
                print(f"    Rollout {name}: FAILED ({exc})")

        gif_path = OUT / f"2dof_{ref_type}.gif"
        _make_pendulum_gif(
            label=ref_type,
            ref_states=ref_states,
            rollouts=rollouts,
            xy_fn=pendulum_xy_2dof,
            n_links=2,
            save_path=gif_path,
            title=f"2-DOF Double Pendulum — {ref_type} reference",
        )

    # ------------------------------------------------------------------
    # 3. Generate N-link chain GIFs
    # ------------------------------------------------------------------
    print("\n[3/3] Generating N-link chain GIFs...")

    for n_links, ckpt in CHECKPOINTS_CHAIN.items():
        if not ckpt.exists():
            print(f"  SKIP {n_links}-link: checkpoint not found")
            continue

        chain_sim = ChainSim(n_links, cfg)
        chain_policy = ChainPolicy(ckpt, tau_max, device)
        print(f"\n  {n_links}-link chain (nq={chain_sim.nq})")

        for ref_type in REF_TYPES_CHAIN:
            ref_states = make_ref_chain(ref_type, chain_sim, tau_max, chain_sim.nq)
            try:
                actual = rollout_chain(chain_policy, ref_states, chain_sim)
            except Exception as exc:
                print(f"    Rollout {ref_type}: FAILED ({exc})")
                continue

            gif_path = OUT / f"chain{n_links}_{ref_type}.gif"
            _make_pendulum_gif(
                label=ref_type,
                ref_states=ref_states,
                rollouts={f"TRACK-ZERO chain{n_links}": actual},
                xy_fn=lambda q, L=LINK_L: chain_xy(q, L),
                n_links=n_links,
                save_path=gif_path,
                title=f"{n_links}-link Chain — {ref_type} reference",
            )
            print(f"    {ref_type}: OK")

    print("\n" + "=" * 60)
    print(f"All GIFs written to: {OUT.relative_to(ROOT)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
