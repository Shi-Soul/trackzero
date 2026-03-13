"""Policy evaluation harness with tracking metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from trackzero.config import Config
from trackzero.sim.simulator import Simulator


@dataclass
class TrackingResult:
    """Per-trajectory tracking result."""
    trajectory_idx: int
    mse_q: float
    mse_v: float
    mse_total: float
    max_error_q: float
    max_error_v: float
    pct95_error_q: float
    pct95_error_v: float
    error_q_curve: list[float] = field(default_factory=list)
    error_v_curve: list[float] = field(default_factory=list)


@dataclass
class EvalSummary:
    """Aggregate evaluation metrics."""
    n_trajectories: int
    mean_mse_q: float
    mean_mse_v: float
    mean_mse_total: float
    median_mse_total: float
    max_mse_total: float
    mean_max_error_q: float
    mean_pct95_error_q: float
    results: list[TrackingResult] = field(default_factory=list)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> EvalSummary:
        with open(path) as f:
            d = json.load(f)
        results = [TrackingResult(**r) for r in d.pop("results")]
        return cls(**d, results=results)


def angular_error(q_ref: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute angular tracking error, handling wraparound.

    Returns error in [-pi, pi] for each element.
    """
    return np.arctan2(np.sin(q_ref - q), np.cos(q_ref - q))


class EvalHarness:
    """Evaluate a tracking policy against reference trajectories."""

    def __init__(self, cfg: Optional[Config] = None):
        if cfg is None:
            cfg = Config()
        self.cfg = cfg
        self.vel_weight = cfg.eval.mse_velocity_weight

    def evaluate_trajectory(
        self,
        policy: Callable[[np.ndarray, np.ndarray], np.ndarray],
        ref_states: np.ndarray,
        ref_actions: np.ndarray,
        trajectory_idx: int = 0,
    ) -> TrackingResult:
        """Evaluate a policy on a single reference trajectory.

        The policy rolls out from the reference initial state, using the
        policy to select actions at each step. Tracking error is measured
        against the reference states.

        Args:
            policy: callable(current_state, ref_next_state) -> action
            ref_states: (T+1, 4) reference state trajectory
            ref_actions: (T, 2) reference actions (unused, for interface compat)
            trajectory_idx: Index for identification.

        Returns:
            TrackingResult with all metrics.
        """
        sim = Simulator(self.cfg)
        T = len(ref_actions)

        # Start from reference initial state
        q0 = ref_states[0, :2]
        v0 = ref_states[0, 2:]
        sim.reset(q0=q0, v0=v0)

        # Roll out with policy
        actual_states = np.zeros((T + 1, 4))
        actual_states[0] = sim.get_state()

        for t in range(T):
            action = policy(actual_states[t], ref_states[t + 1])
            actual_states[t + 1] = sim.step(action)

        # Compute errors (skip initial state since it matches by construction)
        q_errors = angular_error(ref_states[1:, :2], actual_states[1:, :2])  # (T, 2)
        v_errors = ref_states[1:, 2:] - actual_states[1:, 2:]               # (T, 2)

        # Per-timestep scalar errors
        q_err_per_step = np.mean(q_errors ** 2, axis=1)  # (T,)
        v_err_per_step = np.mean(v_errors ** 2, axis=1)  # (T,)

        mse_q = float(np.mean(q_err_per_step))
        mse_v = float(np.mean(v_err_per_step))
        mse_total = mse_q + self.vel_weight * mse_v

        max_q = float(np.max(np.abs(q_errors)))
        max_v = float(np.max(np.abs(v_errors)))

        pct95_q = float(np.percentile(np.abs(q_errors), 95))
        pct95_v = float(np.percentile(np.abs(v_errors), 95))

        return TrackingResult(
            trajectory_idx=trajectory_idx,
            mse_q=mse_q,
            mse_v=mse_v,
            mse_total=mse_total,
            max_error_q=max_q,
            max_error_v=max_v,
            pct95_error_q=pct95_q,
            pct95_error_v=pct95_v,
            error_q_curve=q_err_per_step.tolist(),
            error_v_curve=v_err_per_step.tolist(),
        )

    def evaluate_trajectory_openloop(
        self,
        ref_states: np.ndarray,
        ref_actions: np.ndarray,
        trajectory_idx: int = 0,
    ) -> TrackingResult:
        """Replay ref_actions directly (bypassing policy) to verify determinism.

        This measures simulator reproducibility, not policy quality.
        Error should be zero/machine-epsilon for deterministic simulation.
        """
        sim = Simulator(self.cfg)
        T = len(ref_actions)

        q0 = ref_states[0, :2]
        v0 = ref_states[0, 2:]
        actual_states = sim.rollout(ref_actions, q0=q0, v0=v0)

        q_errors = angular_error(ref_states[1:, :2], actual_states[1:, :2])
        v_errors = ref_states[1:, 2:] - actual_states[1:, 2:]

        q_err_per_step = np.mean(q_errors ** 2, axis=1)
        v_err_per_step = np.mean(v_errors ** 2, axis=1)

        mse_q = float(np.mean(q_err_per_step))
        mse_v = float(np.mean(v_err_per_step))
        mse_total = mse_q + self.vel_weight * mse_v

        return TrackingResult(
            trajectory_idx=trajectory_idx,
            mse_q=mse_q,
            mse_v=mse_v,
            mse_total=mse_total,
            max_error_q=float(np.max(np.abs(q_errors))),
            max_error_v=float(np.max(np.abs(v_errors))),
            pct95_error_q=float(np.percentile(np.abs(q_errors), 95)),
            pct95_error_v=float(np.percentile(np.abs(v_errors), 95)),
            error_q_curve=q_err_per_step.tolist(),
            error_v_curve=v_err_per_step.tolist(),
        )

    def evaluate_policy(
        self,
        policy: Callable[[np.ndarray, np.ndarray], np.ndarray],
        states: np.ndarray,
        actions: np.ndarray,
        max_trajectories: Optional[int] = None,
        progress_callback=None,
        use_gpu: bool = False,
        gpu_batch_size: int = 4096,
        gpu_device: str = "cuda:0",
        gpu_sim=None,
    ) -> EvalSummary:
        """Evaluate policy on a batch of reference trajectories.

        Args:
            policy: callable(current_state, ref_next_state) -> action
            states: (N, T+1, 4) reference states
            actions: (N, T, 2) reference actions
            max_trajectories: Limit the number of trajectories to evaluate.
            progress_callback: Optional callable(n_done, n_total).
            use_gpu: If True, use GPU-parallel simulation for rollouts.
            gpu_batch_size: Number of parallel environments on GPU.
            gpu_device: CUDA device string (e.g. "cuda:0", "cuda:1").
            gpu_sim: Optional pre-built GPUSimulator to reuse.

        Returns:
            EvalSummary with aggregate metrics.
        """
        N = len(states)
        if max_trajectories is not None:
            N = min(N, max_trajectories)

        if use_gpu:
            results = self._evaluate_policy_gpu(
                policy, states[:N], actions[:N], gpu_batch_size, gpu_device,
                gpu_sim=gpu_sim,
            )
        else:
            results = []
            for i in range(N):
                result = self.evaluate_trajectory(
                    policy, states[i], actions[i], trajectory_idx=i
                )
                results.append(result)
                if progress_callback and (i + 1) % 100 == 0:
                    progress_callback(i + 1, N)

        mse_totals = [r.mse_total for r in results]

        summary = EvalSummary(
            n_trajectories=N,
            mean_mse_q=float(np.mean([r.mse_q for r in results])),
            mean_mse_v=float(np.mean([r.mse_v for r in results])),
            mean_mse_total=float(np.mean(mse_totals)),
            median_mse_total=float(np.median(mse_totals)),
            max_mse_total=float(np.max(mse_totals)),
            mean_max_error_q=float(np.mean([r.max_error_q for r in results])),
            mean_pct95_error_q=float(np.mean([r.pct95_error_q for r in results])),
            results=results,
        )

        return summary

    def _evaluate_policy_gpu(
        self,
        policy: Callable,
        ref_states: np.ndarray,
        ref_actions: np.ndarray,
        gpu_batch_size: int,
        gpu_device: str = "cuda:0",
        gpu_sim=None,
    ) -> list[TrackingResult]:
        """GPU-batched policy evaluation.

        Runs policy inference and simulation in lockstep: at each timestep,
        the policy is called on all trajectories simultaneously (batched),
        then all environments are stepped in parallel on GPU.
        """
        import torch
        from trackzero.sim.gpu_simulator import GPUSimulator

        N, T_plus_1, _ = ref_states.shape
        T = T_plus_1 - 1

        if gpu_sim is None or gpu_sim.n_worlds < N:
            gpu_sim = GPUSimulator(self.cfg, n_worlds=max(N, 1), device=gpu_device)

        # Reset to reference initial states
        q0_t = torch.tensor(ref_states[:, 0, :2], dtype=torch.float32, device=gpu_device)
        v0_t = torch.tensor(ref_states[:, 0, 2:], dtype=torch.float32, device=gpu_device)
        gpu_sim.reset_envs(q0_t, v0_t)

        # Batched closed-loop rollout
        actual_states = np.zeros((N, T + 1, 4), dtype=np.float32)
        actual_states[:, 0] = ref_states[:, 0].astype(np.float32)

        current_states = actual_states[:, 0].copy()

        for t in range(T):
            # Batched policy inference (numpy)
            ref_next = ref_states[:, t + 1].astype(np.float32)
            actions_np = np.stack([
                policy(current_states[i], ref_next[i]) for i in range(N)
            ])

            # Step all envs on GPU
            ctrl = torch.tensor(actions_np, dtype=torch.float32, device=gpu_device)
            qpos, qvel = gpu_sim.step_envs(ctrl)

            # Record states
            current_states[:, :2] = qpos.cpu().numpy()
            current_states[:, 2:] = qvel.cpu().numpy()
            actual_states[:, t + 1] = current_states

        # Compute metrics for all trajectories
        results = []
        for i in range(N):
            q_errors = angular_error(ref_states[i, 1:, :2], actual_states[i, 1:, :2])
            v_errors = ref_states[i, 1:, 2:] - actual_states[i, 1:, 2:]

            q_err_per_step = np.mean(q_errors ** 2, axis=1)
            v_err_per_step = np.mean(v_errors ** 2, axis=1)

            mse_q = float(np.mean(q_err_per_step))
            mse_v = float(np.mean(v_err_per_step))

            results.append(TrackingResult(
                trajectory_idx=i,
                mse_q=mse_q,
                mse_v=mse_v,
                mse_total=mse_q + self.vel_weight * mse_v,
                max_error_q=float(np.max(np.abs(q_errors))),
                max_error_v=float(np.max(np.abs(v_errors))),
                pct95_error_q=float(np.percentile(np.abs(q_errors), 95)),
                pct95_error_v=float(np.percentile(np.abs(v_errors), 95)),
                error_q_curve=q_err_per_step.tolist(),
                error_v_curve=v_err_per_step.tolist(),
            ))

        return results
