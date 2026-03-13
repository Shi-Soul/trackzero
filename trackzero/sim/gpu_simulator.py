"""GPU-parallel MuJoCo simulator using mujoco_warp.

Runs thousands of pendulum simulations simultaneously on GPU for fast
data generation, evaluation, and potential RL training.

Uses CUDA graph capture (following mjlab's pattern) to minimize Python
dispatch overhead: mjw.step() is captured once and replayed via
wp.capture_launch().
"""

from __future__ import annotations

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from trackzero.config import Config
from trackzero.sim.pendulum_model import build_pendulum_xml


@wp.kernel
def _copy_ctrl_kernel(
    actions_flat: wp.array2d(dtype=wp.float32),
    ctrl: wp.array2d(dtype=wp.float32),
    offset: int,
):
    """Copy pre-baked actions[offset + i, :] -> ctrl[i, :] on GPU."""
    i = wp.tid()
    for j in range(ctrl.shape[1]):
        ctrl[i, j] = actions_flat[offset + i, j]


@wp.kernel
def _record_state_kernel(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    out_q: wp.array2d(dtype=wp.float32),
    out_v: wp.array2d(dtype=wp.float32),
    offset: int,
):
    """Record qpos/qvel into flat output arrays at given offset."""
    i = wp.tid()
    for j in range(qpos.shape[1]):
        out_q[offset + i, j] = qpos[i, j]
    for j in range(qvel.shape[1]):
        out_v[offset + i, j] = qvel[i, j]


class GPUSimulator:
    """GPU-parallel simulator for batched pendulum rollouts.

    Runs n_worlds independent simulations simultaneously using MuJoCo Warp.
    All state and action data stays on GPU to minimize transfer overhead.

    Uses CUDA graph capture (following mjlab's pattern) for the physics step,
    and PyTorch tensors via wp.to_torch() for zero-copy action writes.
    """

    def __init__(self, cfg: Config, n_worlds: int = 4096, device: str = "cuda:0"):
        self.cfg = cfg
        self.n_worlds = n_worlds
        self.device = device

        self._substeps = cfg.simulation.substeps
        self._nq = 2
        self._nv = 2
        self._nu = 2
        self._tau_max = cfg.pendulum.tau_max

        # Build MuJoCo model
        xml = build_pendulum_xml(cfg.pendulum, cfg.simulation)
        self._cpu_model = mujoco.MjModel.from_xml_string(xml)

        # Upload to GPU — use ScopedDevice to ensure arrays go to the right GPU
        with wp.ScopedDevice(device):
            self._gpu_model = mjw.put_model(self._cpu_model)
            self._gpu_data = mjw.put_data(
                self._cpu_model, mujoco.MjData(self._cpu_model),
                nworld=n_worlds,
            )

            # Warm up kernels (must step once before graph capture)
            mjw.step(self._gpu_model, self._gpu_data)
            wp.synchronize()

            # Capture CUDA graphs (following mjlab pattern)
            self._step_graph = None
            self._forward_graph = None
            self._create_graphs()

        # Create PyTorch tensor views of warp arrays (zero-copy)
        self._ctrl_torch = wp.to_torch(self._gpu_data.ctrl)
        self._qpos_torch = wp.to_torch(self._gpu_data.qpos)
        self._qvel_torch = wp.to_torch(self._gpu_data.qvel)

    def _create_graphs(self):
        """Capture CUDA graphs for step and forward."""
        if not wp.is_mempool_enabled(self.device):
            return

        with wp.ScopedDevice(self.device):
            with wp.ScopedCapture() as capture:
                mjw.step(self._gpu_model, self._gpu_data)
            self._step_graph = capture.graph

            with wp.ScopedCapture() as capture:
                mjw.forward(self._gpu_model, self._gpu_data)
            self._forward_graph = capture.graph

    def _step(self):
        """Execute one physics step (with CUDA graph if available)."""
        if self._step_graph is not None:
            wp.capture_launch(self._step_graph)
        else:
            mjw.step(self._gpu_model, self._gpu_data)

    def _forward(self):
        """Execute forward kinematics (with CUDA graph if available)."""
        if self._forward_graph is not None:
            wp.capture_launch(self._forward_graph)
        else:
            mjw.forward(self._gpu_model, self._gpu_data)

    def rollout_batch(
        self,
        actions: np.ndarray,
        q0: np.ndarray,
        v0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Roll out batched trajectories on GPU.

        Args:
            actions: (N, T, nu) torque sequences. N <= n_worlds.
            q0: (N, nq) initial positions.
            v0: (N, nv) initial velocities.

        Returns:
            states: (N, T+1, nq+nv) state trajectories.
            actions_out: (N, T, nu) clipped actions.
        """
        N, T, nu = actions.shape
        assert N <= self.n_worlds
        assert nu == self._nu
        nw = self.n_worlds

        # Pad to n_worlds if needed
        if N < nw:
            pad_n = nw - N
            actions = np.concatenate([actions, np.zeros((pad_n, T, nu), dtype=np.float32)], axis=0)
            q0 = np.concatenate([q0, np.zeros((pad_n, self._nq), dtype=np.float32)], axis=0)
            v0 = np.concatenate([v0, np.zeros((pad_n, self._nv), dtype=np.float32)], axis=0)

        # Clip and reshape actions to (T, nw, nu) as a contiguous torch tensor on GPU
        actions_clipped = np.clip(actions, -self._tau_max, self._tau_max).astype(np.float32)
        actions_gpu = torch.from_numpy(
            actions_clipped.transpose(1, 0, 2).copy()  # (T, nw, nu), contiguous
        ).to(self._ctrl_torch.device, non_blocking=True)

        # Allocate GPU output: store all states as torch tensors
        out_q = torch.zeros((T + 1, nw, self._nq), device=self._ctrl_torch.device)
        out_v = torch.zeros((T + 1, nw, self._nv), device=self._ctrl_torch.device)

        # Reset state via in-place writes (preserves warp array addresses for CUDA graph)
        self._qpos_torch[:] = torch.from_numpy(q0.astype(np.float32)).to(
            self._qpos_torch.device, non_blocking=True
        )
        self._qvel_torch[:] = torch.from_numpy(v0.astype(np.float32)).to(
            self._qvel_torch.device, non_blocking=True
        )
        self._forward()

        # Record initial state
        out_q[0] = self._qpos_torch
        out_v[0] = self._qvel_torch

        # Step loop — all on GPU, using CUDA graph for physics
        for t in range(T):
            # Write controls in-place (zero-copy: torch -> warp)
            self._ctrl_torch[:] = actions_gpu[t]
            # Substeps with CUDA graph replay
            for _ in range(self._substeps):
                self._step()
            # Record state (torch copy on GPU)
            out_q[t + 1] = self._qpos_torch
            out_v[t + 1] = self._qvel_torch

        torch.cuda.synchronize()

        # Single GPU->CPU transfer
        q_np = out_q[:, :N, :].cpu().numpy()  # (T+1, N, nq)
        v_np = out_v[:, :N, :].cpu().numpy()  # (T+1, N, nv)

        # Combine to (N, T+1, 4)
        states = np.concatenate([q_np, v_np], axis=-1).transpose(1, 0, 2)
        actions_out = actions_clipped[:N]

        return states, actions_out

    def rollout_batch_chunked(
        self,
        actions: np.ndarray,
        q0: np.ndarray,
        v0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Roll out arbitrarily many trajectories by chunking to n_worlds batches.

        Args:
            actions: (N, T, nu) torque sequences. N can be any size.
            q0: (N, nq) initial positions.
            v0: (N, nv) initial velocities.

        Returns:
            states: (N, T+1, nq+nv)
            actions_out: (N, T, nu)
        """
        N = actions.shape[0]
        T = actions.shape[1]

        all_states = np.zeros((N, T + 1, self._nq + self._nv), dtype=np.float32)
        all_actions = np.zeros((N, T, self._nu), dtype=np.float32)

        for start in range(0, N, self.n_worlds):
            end = min(start + self.n_worlds, N)
            chunk_states, chunk_actions = self.rollout_batch(
                actions[start:end], q0[start:end], v0[start:end],
            )
            all_states[start:end] = chunk_states
            all_actions[start:end] = chunk_actions

        return all_states, all_actions

    def step_envs(
        self,
        ctrl: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Step all environments by one control step (for RL use).

        Args:
            ctrl: (n_worlds, nu) torque tensor on GPU.

        Returns:
            qpos: (n_worlds, nq) joint positions after step.
            qvel: (n_worlds, nv) joint velocities after step.
        """
        self._ctrl_torch[:] = ctrl.clamp(-self._tau_max, self._tau_max)
        for _ in range(self._substeps):
            self._step()
        return self._qpos_torch.clone(), self._qvel_torch.clone()

    def reset_envs(
        self,
        q0: torch.Tensor,
        v0: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ):
        """Reset specified (or all) environments.

        Args:
            q0: (n_worlds, nq) or (len(env_ids), nq) initial positions.
            v0: (n_worlds, nv) or (len(env_ids), nv) initial velocities.
            env_ids: Optional subset of environment indices to reset.
        """
        if env_ids is None:
            self._qpos_torch[:] = q0
            self._qvel_torch[:] = v0
        else:
            self._qpos_torch[env_ids] = q0
            self._qvel_torch[env_ids] = v0
        self._forward()
