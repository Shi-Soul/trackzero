"""Reachability-guided single-step data generation for Stage 1D.

Instead of rolling out trajectories under random torques (which concentrate
near low-energy attractors), we sample states uniformly across the feasible
region and apply random actions. Each (s, a, s') tuple is a valid training
sample for inverse dynamics.
"""

from __future__ import annotations

import numpy as np

from trackzero.config import Config
from trackzero.sim.simulator import Simulator


def generate_reachability_data(
    cfg: Config,
    n_transitions: int,
    q_range: tuple[float, float] = (-np.pi, np.pi),
    v_range: tuple[float, float] = (-15.0, 15.0),
    seed: int = 42,
    progress_every: int = 100000,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate single-step (state, action, next_state) data.

    Samples states uniformly in [q_range]^2 x [v_range]^2, applies
    random actions in [-tau_max, tau_max]^2, simulates one step.

    Returns:
        states: (N, 2, 4) — [current_state, next_state] per sample
        actions: (N, 1, 2) — applied torque per sample
    """
    rng = np.random.default_rng(seed)
    sim = Simulator(cfg)
    tau_max = cfg.pendulum.tau_max

    states = np.zeros((n_transitions, 2, 4), dtype=np.float32)
    actions = np.zeros((n_transitions, 1, 2), dtype=np.float32)

    for i in range(n_transitions):
        q = rng.uniform(q_range[0], q_range[1], size=2)
        v = rng.uniform(v_range[0], v_range[1], size=2)
        a = rng.uniform(-tau_max, tau_max, size=2)

        sim.reset(q0=q, v0=v)
        s0 = sim.get_state()
        s1 = sim.step(a)

        states[i, 0] = s0
        states[i, 1] = s1
        actions[i, 0] = a

        if progress_every and (i + 1) % progress_every == 0:
            print(f"  Generated {i+1}/{n_transitions} transitions")

    return states, actions


def generate_reachability_data_batched(
    cfg: Config,
    n_transitions: int,
    q_range: tuple[float, float] = (-np.pi, np.pi),
    v_range: tuple[float, float] = (-15.0, 15.0),
    seed: int = 42,
    batch_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched version: generates in chunks with progress reporting.

    Same interface as generate_reachability_data but processes in batches
    for better memory management and progress tracking.
    """
    rng = np.random.default_rng(seed)
    sim = Simulator(cfg)
    tau_max = cfg.pendulum.tau_max

    all_states = []
    all_actions = []
    done = 0

    while done < n_transitions:
        bs = min(batch_size, n_transitions - done)
        s_batch = np.zeros((bs, 2, 4), dtype=np.float32)
        a_batch = np.zeros((bs, 1, 2), dtype=np.float32)

        for j in range(bs):
            q = rng.uniform(q_range[0], q_range[1], size=2)
            v = rng.uniform(v_range[0], v_range[1], size=2)
            a = rng.uniform(-tau_max, tau_max, size=2)

            sim.reset(q0=q, v0=v)
            s_batch[j, 0] = sim.get_state()
            s_batch[j, 1] = sim.step(a)
            a_batch[j, 0] = a

        all_states.append(s_batch)
        all_actions.append(a_batch)
        done += bs
        print(f"  Generated {done}/{n_transitions} transitions")

    return np.concatenate(all_states), np.concatenate(all_actions)


def generate_mixed_reachability_data(
    cfg: Config,
    n_single_step: int,
    n_short_traj: int,
    short_traj_len: int = 50,
    q_range: tuple[float, float] = (-np.pi, np.pi),
    v_range: tuple[float, float] = (-15.0, 15.0),
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Mix single-step and short-trajectory reachability data.

    Single-step data provides uniform state coverage.
    Short trajectories provide local temporal coherence.

    Returns data in unified format: states (N, T+1, 4), actions (N, T, 2)
    where T=1 for single-step and T=short_traj_len for trajectories.
    Note: These have different T dimensions so are returned separately.
    """
    rng = np.random.default_rng(seed)
    sim = Simulator(cfg)
    tau_max = cfg.pendulum.tau_max

    # Single-step data
    print(f"Generating {n_single_step} single-step transitions...")
    ss_states = np.zeros((n_single_step, 2, 4), dtype=np.float32)
    ss_actions = np.zeros((n_single_step, 1, 2), dtype=np.float32)

    for i in range(n_single_step):
        q = rng.uniform(q_range[0], q_range[1], size=2)
        v = rng.uniform(v_range[0], v_range[1], size=2)
        a = rng.uniform(-tau_max, tau_max, size=2)
        sim.reset(q0=q, v0=v)
        ss_states[i, 0] = sim.get_state()
        ss_states[i, 1] = sim.step(a)
        ss_actions[i, 0] = a
        if (i + 1) % 100000 == 0:
            print(f"  Single-step: {i+1}/{n_single_step}")

    # Short trajectories from random initial states
    print(f"Generating {n_short_traj} short trajectories (len={short_traj_len})...")
    st_states = np.zeros(
        (n_short_traj, short_traj_len + 1, 4), dtype=np.float32
    )
    st_actions = np.zeros(
        (n_short_traj, short_traj_len, 2), dtype=np.float32
    )

    for i in range(n_short_traj):
        q = rng.uniform(q_range[0], q_range[1], size=2)
        v = rng.uniform(v_range[0], v_range[1], size=2)
        acts = rng.uniform(-tau_max, tau_max, size=(short_traj_len, 2))
        traj = sim.rollout(acts, q0=q, v0=v)
        st_states[i] = traj
        st_actions[i] = acts
        if (i + 1) % 10000 == 0:
            print(f"  Short traj: {i+1}/{n_short_traj}")

    return ss_states, ss_actions, st_states, st_actions
