"""Tests for GPU-parallel simulator."""

import numpy as np
import pytest
import torch

from trackzero.config import Config
from trackzero.sim.gpu_simulator import GPUSimulator
from trackzero.sim.simulator import Simulator


@pytest.fixture(scope="module")
def cfg():
    return Config()


@pytest.fixture(scope="module")
def gpu_sim(cfg):
    return GPUSimulator(cfg, n_worlds=64)


@pytest.fixture(scope="module")
def cpu_sim(cfg):
    return Simulator(cfg)


class TestGPUSimulator:
    def test_rollout_shape(self, gpu_sim, cfg):
        T = int(cfg.dataset.trajectory_duration / cfg.simulation.control_dt)
        N = 8
        rng = np.random.default_rng(0)
        q0 = rng.uniform(-np.pi, np.pi, (N, 2)).astype(np.float32)
        v0 = rng.uniform(-3, 3, (N, 2)).astype(np.float32)
        actions = rng.uniform(-5, 5, (N, T, 2)).astype(np.float32)

        states, acts = gpu_sim.rollout_batch(actions, q0, v0)
        assert states.shape == (N, T + 1, 4)
        assert acts.shape == (N, T, 2)

    def test_correctness_vs_cpu(self, gpu_sim, cpu_sim, cfg):
        """GPU and CPU simulators should produce the same trajectories (up to float32 precision)."""
        T = int(cfg.dataset.trajectory_duration / cfg.simulation.control_dt)
        N = 10
        rng = np.random.default_rng(42)
        q0 = rng.uniform(-np.pi, np.pi, (N, 2)).astype(np.float32)
        v0 = rng.uniform(-3, 3, (N, 2)).astype(np.float32)
        actions = rng.uniform(-5, 5, (N, T, 2)).astype(np.float32)

        gpu_states, _ = gpu_sim.rollout_batch(actions, q0, v0)

        for i in range(N):
            cpu_states = cpu_sim.rollout(
                actions[i].astype(np.float64),
                q0=q0[i].astype(np.float64),
                v0=v0[i].astype(np.float64),
            )
            max_err = np.max(np.abs(cpu_states - gpu_states[i].astype(np.float64)))
            assert max_err < 1e-3, f"traj {i}: max error {max_err:.2e}"

    def test_deterministic(self, gpu_sim, cfg):
        T = int(cfg.dataset.trajectory_duration / cfg.simulation.control_dt)
        N = 8
        rng = np.random.default_rng(7)
        q0 = rng.uniform(-np.pi, np.pi, (N, 2)).astype(np.float32)
        v0 = rng.uniform(-3, 3, (N, 2)).astype(np.float32)
        actions = rng.uniform(-5, 5, (N, T, 2)).astype(np.float32)

        s1, _ = gpu_sim.rollout_batch(actions, q0, v0)
        s2, _ = gpu_sim.rollout_batch(actions, q0, v0)
        np.testing.assert_allclose(s1, s2, atol=1e-5)

    def test_action_clipping(self, gpu_sim, cfg):
        T = 10
        N = 4
        q0 = np.zeros((N, 2), dtype=np.float32)
        v0 = np.zeros((N, 2), dtype=np.float32)
        # Actions way beyond tau_max
        actions = np.full((N, T, 2), 100.0, dtype=np.float32)

        _, acts = gpu_sim.rollout_batch(actions, q0, v0)
        assert np.all(np.abs(acts) <= cfg.pendulum.tau_max + 1e-6)

    def test_no_nans(self, gpu_sim, cfg):
        T = int(cfg.dataset.trajectory_duration / cfg.simulation.control_dt)
        N = 32
        rng = np.random.default_rng(99)
        q0 = rng.uniform(-np.pi, np.pi, (N, 2)).astype(np.float32)
        v0 = rng.uniform(-3, 3, (N, 2)).astype(np.float32)
        actions = rng.uniform(-5, 5, (N, T, 2)).astype(np.float32)

        states, _ = gpu_sim.rollout_batch(actions, q0, v0)
        assert not np.any(np.isnan(states))

    def test_chunked_matches_batch(self, gpu_sim, cfg):
        """Chunked rollout should produce identical results to batched."""
        T = 50  # shorter for speed
        N = 100  # more than n_worlds=64
        rng = np.random.default_rng(11)
        q0 = rng.uniform(-np.pi, np.pi, (N, 2)).astype(np.float32)
        v0 = rng.uniform(-3, 3, (N, 2)).astype(np.float32)
        actions = rng.uniform(-5, 5, (N, T, 2)).astype(np.float32)

        states, _ = gpu_sim.rollout_batch_chunked(actions, q0, v0)
        assert states.shape == (N, T + 1, 4)

        # First chunk should match direct batch
        s1, _ = gpu_sim.rollout_batch(actions[:64], q0[:64], v0[:64])
        np.testing.assert_allclose(states[:64], s1, atol=1e-5)

    def test_step_envs(self, gpu_sim, cfg):
        """Test the RL-style step_envs interface."""
        device = gpu_sim._ctrl_torch.device
        q0 = torch.zeros(gpu_sim.n_worlds, 2, device=device)
        v0 = torch.zeros(gpu_sim.n_worlds, 2, device=device)
        gpu_sim.reset_envs(q0, v0)

        ctrl = torch.randn(gpu_sim.n_worlds, 2, device=device)
        qpos, qvel = gpu_sim.step_envs(ctrl)
        assert qpos.shape == (gpu_sim.n_worlds, 2)
        assert qvel.shape == (gpu_sim.n_worlds, 2)
        assert not torch.any(torch.isnan(qpos))

    def test_reset_envs_subset(self, gpu_sim, cfg):
        """Test partial environment reset."""
        device = gpu_sim._ctrl_torch.device
        q0 = torch.zeros(gpu_sim.n_worlds, 2, device=device)
        v0 = torch.zeros(gpu_sim.n_worlds, 2, device=device)
        gpu_sim.reset_envs(q0, v0)

        # Step once
        ctrl = torch.ones(gpu_sim.n_worlds, 2, device=device)
        gpu_sim.step_envs(ctrl)

        # Reset only envs 0 and 5
        env_ids = torch.tensor([0, 5], device=device)
        new_q = torch.ones(2, 2, device=device) * 1.0
        new_v = torch.zeros(2, 2, device=device)
        gpu_sim.reset_envs(new_q, new_v, env_ids=env_ids)

        assert torch.allclose(gpu_sim._qpos_torch[0], torch.ones(2, device=device))
        assert torch.allclose(gpu_sim._qpos_torch[5], torch.ones(2, device=device))
