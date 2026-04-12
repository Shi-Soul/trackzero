"""GPU-vectorized environment for RL exploration.

Wraps GPUSimulator as a batched RL env with reset/step interface.
All tensors stay on GPU.
"""

from __future__ import annotations

import torch

from trackzero.config import Config
from trackzero.sim.gpu_simulator import GPUSimulator


class VecDoublePendulumEnv:
    """Batched double-pendulum env on GPU for RL training."""

    def __init__(
        self,
        cfg: Config,
        n_envs: int = 4096,
        episode_len: int = 500,
        device: str = "cuda:0",
        q_range: tuple[float, float] = (-3.14159, 3.14159),
        v_range: tuple[float, float] = (-15.0, 15.0),
    ):
        self.cfg = cfg
        self.n_envs = n_envs
        self.episode_len = episode_len
        self.device = device
        self.q_range = q_range
        self.v_range = v_range
        self.tau_max = cfg.pendulum.tau_max
        self.sim = GPUSimulator(cfg, n_worlds=n_envs, device=device)
        self.obs_dim = 4
        self.act_dim = 2
        self.step_count = torch.zeros(n_envs, dtype=torch.long, device=device)

    def reset(self) -> torch.Tensor:
        """Reset all envs to random states. Returns obs (n_envs, 4)."""
        q0 = torch.empty(self.n_envs, 2, device=self.device).uniform_(
            self.q_range[0], self.q_range[1])
        v0 = torch.empty(self.n_envs, 2, device=self.device).uniform_(
            self.v_range[0], self.v_range[1])
        self.sim.reset_envs(q0, v0)
        self.step_count.zero_()
        return torch.cat([q0, v0], dim=-1)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Step all envs. Returns (obs, done_mask)."""
        qpos, qvel = self.sim.step_envs(actions)
        obs = torch.cat([qpos, qvel], dim=-1)
        self.step_count += 1
        done = self.step_count >= self.episode_len
        return obs, done

    def partial_reset(self, done_mask: torch.Tensor) -> torch.Tensor:
        """Reset only done envs. Returns full obs tensor."""
        if done_mask.any():
            idx = done_mask.nonzero(as_tuple=True)[0]
            n = idx.shape[0]
            q0 = torch.empty(n, 2, device=self.device).uniform_(
                self.q_range[0], self.q_range[1])
            v0 = torch.empty(n, 2, device=self.device).uniform_(
                self.v_range[0], self.v_range[1])
            self.sim.reset_envs(q0, v0, env_ids=idx)
            self.step_count[idx] = 0
        qpos = self.sim._qpos_torch.clone()
        qvel = self.sim._qvel_torch.clone()
        return torch.cat([qpos, qvel], dim=-1)
