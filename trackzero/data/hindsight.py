"""Hindsight-relabeling data generation for Stage 1D."""

from __future__ import annotations

import numpy as np
import torch

from trackzero.config import Config
from trackzero.data.ood_references import generate_ood_reference_data
from trackzero.data.random_rollout import generate_random_rollout_data
from trackzero.eval.harness import EvalHarness
from trackzero.sim.simulator import Simulator


def generate_reference_batch(
    cfg: Config,
    n_trajectories: int,
    source: str,
    seed: int,
    gpu_device: str = "cuda:0",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate arbitrary reference trajectories for hindsight rollouts."""
    if source in {"mixed_ood", "chirp", "step", "random_walk", "sawtooth", "pulse"}:
        return generate_ood_reference_data(cfg, n_trajectories, action_type=source, seed=seed)
    if source in {"uniform", "gaussian", "ou", "bangbang", "multisine", "mixed"}:
        return generate_random_rollout_data(
            cfg,
            n_trajectories,
            action_type=source,
            seed=seed,
            use_gpu=True,
            gpu_device=gpu_device,
        )
    raise ValueError(f"Unknown hindsight reference source: {source}")


def rollout_policy_as_hindsight_data(
    cfg: Config,
    policy,
    ref_states: np.ndarray,
    ref_actions: np.ndarray,
    gpu_device: str = "cuda:0",
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out a policy against arbitrary references and relabel achieved motion.

    Returns:
        actual_states: (N, T+1, 4) achieved trajectories
        actual_actions: (N, T, 2) actions applied by the policy
    """
    n_traj, t_plus_1, _ = ref_states.shape
    t_steps = t_plus_1 - 1

    try:
        from trackzero.sim.gpu_simulator import GPUSimulator
        return _rollout_policy_gpu(cfg, policy, ref_states, gpu_device)
    except ImportError:
        return _rollout_policy_cpu(cfg, policy, ref_states)


def _rollout_policy_gpu(
    cfg: Config,
    policy,
    ref_states: np.ndarray,
    gpu_device: str,
) -> tuple[np.ndarray, np.ndarray]:
    n_traj, t_plus_1, _ = ref_states.shape
    t_steps = t_plus_1 - 1

    gpu_sim = GPUSimulator(cfg, n_worlds=max(n_traj, 1), device=gpu_device)
    q0_full = torch.zeros(gpu_sim.n_worlds, 2, dtype=torch.float32, device=gpu_device)
    v0_full = torch.zeros(gpu_sim.n_worlds, 2, dtype=torch.float32, device=gpu_device)
    q0_full[:n_traj] = torch.tensor(ref_states[:, 0, :2], dtype=torch.float32, device=gpu_device)
    v0_full[:n_traj] = torch.tensor(ref_states[:, 0, 2:], dtype=torch.float32, device=gpu_device)
    gpu_sim.reset_envs(q0_full, v0_full)

    actual_states = np.zeros((n_traj, t_steps + 1, 4), dtype=np.float32)
    actual_actions = np.zeros((n_traj, t_steps, 2), dtype=np.float32)
    actual_states[:, 0] = ref_states[:, 0].astype(np.float32)
    current_states = actual_states[:, 0].copy()
    has_batch = hasattr(policy, "batch_call")

    for t in range(t_steps):
        ref_next = ref_states[:, t + 1].astype(np.float32)
        if has_batch:
            actions_np = policy.batch_call(current_states, ref_next).astype(np.float32)
        else:
            actions_np = np.stack([policy(current_states[i], ref_next[i]) for i in range(n_traj)]).astype(np.float32)
        actual_actions[:, t] = actions_np

        ctrl = torch.zeros(gpu_sim.n_worlds, 2, dtype=torch.float32, device=gpu_device)
        ctrl[:n_traj] = torch.tensor(actions_np, dtype=torch.float32, device=gpu_device)
        qpos, qvel = gpu_sim.step_envs(ctrl)
        current_states[:, :2] = qpos[:n_traj].cpu().numpy()
        current_states[:, 2:] = qvel[:n_traj].cpu().numpy()
        actual_states[:, t + 1] = current_states

    return actual_states.astype(np.float64), actual_actions.astype(np.float64)


def _rollout_policy_cpu(
    cfg: Config,
    policy,
    ref_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched CPU rollout: batch policy calls across all trajectories at each timestep,
    but step each simulator individually (physics must be sequential per trajectory)."""
    n_traj, t_plus_1, _ = ref_states.shape
    t_steps = t_plus_1 - 1
    actual_states = np.zeros((n_traj, t_steps + 1, 4), dtype=np.float64)
    actual_actions = np.zeros((n_traj, t_steps, 2), dtype=np.float64)

    # Initialize all simulators
    sims = []
    for i in range(n_traj):
        sim = Simulator(cfg)
        sim.reset(q0=ref_states[i, 0, :2], v0=ref_states[i, 0, 2:])
        actual_states[i, 0] = sim.get_state()
        sims.append(sim)

    has_batch = hasattr(policy, "batch_call")

    for t in range(t_steps):
        current_batch = actual_states[:, t].astype(np.float32)
        ref_next_batch = ref_states[:, t + 1].astype(np.float32)

        if has_batch:
            actions_np = policy.batch_call(current_batch, ref_next_batch)
        else:
            actions_np = np.stack([
                policy(current_batch[i], ref_next_batch[i]) for i in range(n_traj)
            ])
        actual_actions[:, t] = actions_np

        for i in range(n_traj):
            actual_states[i, t + 1] = sims[i].step(actions_np[i])

        if t % 100 == 0:
            print(f"  hindsight rollout step {t}/{t_steps}")

    return actual_states, actual_actions


def evaluate_teacher_on_requested_refs(
    cfg: Config,
    policy,
    ref_states: np.ndarray,
    ref_actions: np.ndarray,
    gpu_device: str = "cuda:0",
):
    """Measure how far the teacher deviates from requested references."""
    harness = EvalHarness(cfg)
    try:
        return harness.evaluate_policy(
            policy,
            ref_states,
            ref_actions,
            max_trajectories=len(ref_states),
            use_gpu=True,
            gpu_device=gpu_device,
        )
    except ImportError:
        return harness.evaluate_policy(
            policy,
            ref_states,
            ref_actions,
            max_trajectories=len(ref_states),
            use_gpu=False,
        )
