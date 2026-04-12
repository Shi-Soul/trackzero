"""Out-of-distribution reference trajectory generation.

These use signal families that are NOT used during TRACK-ZERO training, so
evaluating on them tests genuine generalization beyond the training distribution.

Available generators (each takes keyword args rng, T, n_joints, tau_max, dt):
    chirp            — frequency-sweep sinusoid
    step             — step function with instantaneous transitions
    random_walk      — integrated Gaussian noise (Brownian-like)
    sawtooth         — sawtooth wave
    pulse            — random rectangular pulses
"""

from __future__ import annotations

import numpy as np

from trackzero.config import Config
from trackzero.sim.simulator import Simulator


def generate_chirp_actions(
    rng: np.random.Generator,
    T: int,
    n_joints: int,
    tau_max: float,
    dt: float,
) -> np.ndarray:
    """Linear frequency-sweep (chirp) torque signal."""
    t = np.arange(T) * dt
    duration = T * dt
    actions = np.zeros((T, n_joints))
    for j in range(n_joints):
        f_start = rng.uniform(0.1, 0.5)
        f_end = rng.uniform(1.0, 5.0)
        # Instantaneous phase: integral of 2*pi*f(tau) d(tau)
        phase = 2.0 * np.pi * (
            f_start * t + 0.5 * (f_end - f_start) * t ** 2 / max(duration, 1e-9)
        )
        amp = rng.uniform(0.3 * tau_max, tau_max)
        phase_offset = rng.uniform(0.0, 2.0 * np.pi)
        actions[:, j] = np.clip(
            amp * np.sin(phase + phase_offset), -tau_max, tau_max
        )
    return actions


def generate_step_actions(
    rng: np.random.Generator,
    T: int,
    n_joints: int,
    tau_max: float,
    dt: float,
) -> np.ndarray:
    """Step function with randomly placed level changes."""
    actions = np.zeros((T, n_joints))
    n_steps = int(rng.integers(3, 10))
    for j in range(n_joints):
        switch_times = np.sort(rng.integers(0, T, size=n_steps - 1)).tolist()
        switch_times = [0] + switch_times + [T]
        levels = rng.uniform(-tau_max, tau_max, size=n_steps)
        for k in range(n_steps):
            t0, t1 = switch_times[k], switch_times[k + 1]
            actions[t0:t1, j] = levels[k]
    return np.clip(actions, -tau_max, tau_max)


def generate_random_walk_actions(
    rng: np.random.Generator,
    T: int,
    n_joints: int,
    tau_max: float,
    dt: float,
) -> np.ndarray:
    """Random walk (integrated Gaussian) torque signal."""
    sigma = tau_max * 0.3 * np.sqrt(dt)
    actions = np.zeros((T, n_joints))
    for j in range(n_joints):
        x = rng.uniform(-tau_max * 0.5, tau_max * 0.5)
        for t in range(T):
            x = np.clip(x + sigma * rng.standard_normal(), -tau_max, tau_max)
            actions[t, j] = x
    return actions


def generate_sawtooth_actions(
    rng: np.random.Generator,
    T: int,
    n_joints: int,
    tau_max: float,
    dt: float,
) -> np.ndarray:
    """Sawtooth wave torque signal."""
    t = np.arange(T) * dt
    actions = np.zeros((T, n_joints))
    for j in range(n_joints):
        freq = rng.uniform(0.2, 2.0)
        amp = rng.uniform(0.3 * tau_max, tau_max)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        # Normalised sawtooth in [-1, 1]: 2*(frac - 0.5)
        frac = (t * freq + phase / (2.0 * np.pi)) % 1.0
        actions[:, j] = np.clip(amp * (2.0 * frac - 1.0), -tau_max, tau_max)
    return actions


def generate_pulse_actions(
    rng: np.random.Generator,
    T: int,
    n_joints: int,
    tau_max: float,
    dt: float,
) -> np.ndarray:
    """Random rectangular pulses."""
    actions = np.zeros((T, n_joints))
    n_pulses = int(rng.integers(3, 15))
    for j in range(n_joints):
        for _ in range(n_pulses):
            start = int(rng.integers(0, T))
            dur = int(rng.integers(1, max(2, T // 20)))
            amp = rng.uniform(-tau_max, tau_max)
            actions[start:min(start + dur, T), j] = amp
    return actions


# Registry: maps generator name -> callable with signature
#   (rng, T, n_joints, tau_max, dt) -> np.ndarray (T, n_joints)
OOD_ACTION_GENERATORS: dict = {
    "chirp": generate_chirp_actions,
    "step": generate_step_actions,
    "random_walk": generate_random_walk_actions,
    "sawtooth": generate_sawtooth_actions,
    "pulse": generate_pulse_actions,
}


def generate_ood_reference_data(
    cfg: Config,
    n_trajectories: int,
    action_type: str = "mixed_ood",
    seed: int = 0,
    use_gpu: bool = False,
    gpu_device: str = "cuda:0",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate out-of-distribution reference trajectories.

    Args:
        cfg: Project configuration.
        n_trajectories: Number of trajectories.
        action_type: One of the OOD_ACTION_GENERATORS keys, or 'mixed_ood'
                     to cycle through all types in round-robin order.
        seed: Random seed.
        use_gpu: If True, attempt GPU rollout (falls back to CPU on error).
        gpu_device: CUDA device string.

    Returns:
        states:  (N, T+1, 4)
        actions: (N, T, 2)
    """
    T = int(cfg.dataset.trajectory_duration / cfg.simulation.control_dt)
    tau_max = cfg.pendulum.tau_max
    n_joints = 2
    dt = cfg.simulation.control_dt

    rng = np.random.default_rng(seed)
    sim = Simulator(cfg)

    all_states = np.zeros((n_trajectories, T + 1, 4), dtype=np.float64)
    all_actions = np.zeros((n_trajectories, T, n_joints), dtype=np.float64)

    ood_keys = list(OOD_ACTION_GENERATORS.keys())

    for i in range(n_trajectories):
        q0 = rng.uniform(
            cfg.dataset.initial_state.q_range[0],
            cfg.dataset.initial_state.q_range[1],
            size=2,
        )
        v0 = rng.uniform(
            cfg.dataset.initial_state.v_range[0],
            cfg.dataset.initial_state.v_range[1],
            size=2,
        )

        if action_type == "mixed_ood":
            gen_fn = OOD_ACTION_GENERATORS[ood_keys[i % len(ood_keys)]]
        else:
            gen_fn = OOD_ACTION_GENERATORS[action_type]

        actions = gen_fn(rng=rng, T=T, n_joints=n_joints, tau_max=tau_max, dt=dt)
        states = sim.rollout(actions, q0=q0, v0=v0)
        all_states[i] = states
        all_actions[i] = actions

    return all_states, all_actions
