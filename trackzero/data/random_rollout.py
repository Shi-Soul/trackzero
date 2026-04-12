"""Random rollout data generation for Stage 1B (TRACK-ZERO v0).

Instead of multisine references, the pendulum is driven with various types of
random torque sequences to study what state-action coverage they produce.
"""

from __future__ import annotations

import numpy as np

from trackzero.config import Config
from trackzero.data.multisine import generate_multisine_actions
from trackzero.sim.simulator import Simulator


def _ou_noise(
    rng: np.random.Generator,
    T: int,
    n: int,
    theta: float = 0.15,
    sigma: float = 1.0,
    dt: float = 0.01,
    tau_max: float = 5.0,
) -> np.ndarray:
    """Ornstein-Uhlenbeck correlated noise."""
    x = rng.uniform(-tau_max * 0.5, tau_max * 0.5, size=n)
    out = np.zeros((T, n))
    for t in range(T):
        x = x + (-theta * x * dt + sigma * np.sqrt(dt) * rng.standard_normal(n))
        out[t] = np.clip(x, -tau_max, tau_max)
    return out


def _generate_actions_one(
    rng: np.random.Generator,
    action_type: str,
    T: int,
    n_joints: int,
    tau_max: float,
    dt: float,
    cfg: Config,
    action_params: dict | None = None,
) -> np.ndarray:
    """Generate a single torque sequence of the requested type.

    Args:
        action_params: Optional dict of per-type hyperparameters:
            - ou: theta (mean-reversion, default 0.15), sigma (volatility, default tau_max*0.5)
            - gaussian: scale_factor (fraction of tau_max for std, default 1/3)
            - bangbang: min_hold (default 1), max_hold (default T//10)
            - multisine: passed through to generate_multisine_actions (unused currently)
            - mixed: weights (list of 5 floats, one per type: uniform/gaussian/ou/bangbang/multisine)
    """
    p = action_params or {}

    if action_type == "uniform":
        return rng.uniform(-tau_max, tau_max, size=(T, n_joints))

    elif action_type == "gaussian":
        scale_factor = p.get("scale_factor", 1.0 / 3.0)
        scale = tau_max * scale_factor
        return np.clip(rng.normal(0.0, scale, size=(T, n_joints)), -tau_max, tau_max)

    elif action_type == "ou":
        theta = p.get("theta", 0.15)
        sigma = p.get("sigma", tau_max * 0.5)
        return _ou_noise(rng, T, n_joints, theta=theta, sigma=sigma, dt=dt, tau_max=tau_max)

    elif action_type == "bangbang":
        min_hold = int(p.get("min_hold", 1))
        max_hold = int(p.get("max_hold", max(2, T // 10)))
        actions = np.zeros((T, n_joints))
        for j in range(n_joints):
            sign = rng.choice([-1.0, 1.0])
            t = 0
            while t < T:
                hold = int(rng.integers(min_hold, max_hold + 1))
                end = min(t + hold, T)
                actions[t:end, j] = sign * tau_max
                sign = -sign
                t = end
        return actions

    elif action_type == "multisine":
        return generate_multisine_actions(rng, cfg.dataset, cfg.pendulum, dt)

    elif action_type == "mixed":
        subtypes = ["uniform", "gaussian", "ou", "bangbang", "multisine"]
        weights = p.get("weights", None)
        if weights is not None:
            weights = np.array(weights, dtype=float)
            weights = weights / weights.sum()
            chosen = subtypes[rng.choice(len(subtypes), p=weights)]
        else:
            chosen = subtypes[int(rng.integers(0, len(subtypes)))]
        return _generate_actions_one(rng, chosen, T, n_joints, tau_max, dt, cfg, action_params)

    else:
        raise ValueError(f"Unknown action_type: {action_type!r}")


def generate_random_rollout_data(
    cfg: Config,
    n_trajectories: int,
    action_type: str = "mixed",
    seed: int = 0,
    use_gpu: bool = False,
    gpu_device: str = "cuda:0",
    action_params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate training data by rolling out random torque sequences.

    Args:
        cfg: Project configuration.
        n_trajectories: Number of rollout trajectories to generate.
        action_type: Torque distribution — one of
            'uniform', 'gaussian', 'ou', 'bangbang', 'multisine', 'mixed'.
        seed: Random seed.
        use_gpu: If True, use GPU-parallel simulation (requires mujoco_warp).
                 Falls back to CPU if mujoco_warp is unavailable.
        gpu_device: CUDA device string (e.g. 'cuda:0').
        action_params: Optional dict of per-type hyperparameters (see
            _generate_actions_one for supported keys).

    Returns:
        states:  (N, T+1, 4) state trajectories.
        actions: (N, T, 2)  applied torques.
    """
    T = int(cfg.dataset.trajectory_duration / cfg.simulation.control_dt)
    tau_max = cfg.pendulum.tau_max
    n_joints = 2
    dt = cfg.simulation.control_dt

    rng = np.random.default_rng(seed)

    if use_gpu:
        try:
            return _generate_random_rollout_gpu(
                cfg, n_trajectories, action_type, rng, T, n_joints, tau_max, dt, gpu_device
            )
        except ImportError:
            import warnings
            warnings.warn("mujoco_warp not available; falling back to CPU rollout.")

    sim = Simulator(cfg)
    all_states = np.zeros((n_trajectories, T + 1, 4), dtype=np.float64)
    all_actions = np.zeros((n_trajectories, T, n_joints), dtype=np.float64)

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
        actions = _generate_actions_one(rng, action_type, T, n_joints, tau_max, dt, cfg, action_params)
        states = sim.rollout(actions, q0=q0, v0=v0)
        all_states[i] = states
        all_actions[i] = actions

    return all_states, all_actions


def _generate_random_rollout_gpu(
    cfg: Config,
    n_trajectories: int,
    action_type: str,
    rng: np.random.Generator,
    T: int,
    n_joints: int,
    tau_max: float,
    dt: float,
    gpu_device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU-parallel rollout using GPUSimulator."""
    from trackzero.sim.gpu_simulator import GPUSimulator

    all_actions_np = np.zeros((n_trajectories, T, n_joints), dtype=np.float32)
    q0_all = np.zeros((n_trajectories, 2), dtype=np.float32)
    v0_all = np.zeros((n_trajectories, 2), dtype=np.float32)

    for i in range(n_trajectories):
        q0_all[i] = rng.uniform(
            cfg.dataset.initial_state.q_range[0],
            cfg.dataset.initial_state.q_range[1],
            size=2,
        )
        v0_all[i] = rng.uniform(
            cfg.dataset.initial_state.v_range[0],
            cfg.dataset.initial_state.v_range[1],
            size=2,
        )
        all_actions_np[i] = _generate_actions_one(
            rng, action_type, T, n_joints, tau_max, dt, cfg
        ).astype(np.float32)

    gpu_sim = GPUSimulator(cfg, n_worlds=min(n_trajectories, 4096), device=gpu_device)
    states, actions_out = gpu_sim.rollout_batch_chunked(all_actions_np, q0_all, v0_all)
    return states.astype(np.float64), actions_out.astype(np.float64)
