"""Tests for dataset generation and loading."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from trackzero.config import Config
from trackzero.data.generator import generate_dataset
from trackzero.data.dataset import TrajectoryDataset


@pytest.fixture
def cfg():
    return Config()


@pytest.fixture
def small_dataset(cfg, tmp_path):
    """Generate a small dataset for testing."""
    path = tmp_path / "test_ds.h5"
    generate_dataset(cfg, path, n_trajectories=10, seed=42, num_workers=0)
    return path


class TestDatasetGeneration:
    def test_generate_creates_file(self, cfg, tmp_path):
        path = tmp_path / "out.h5"
        result = generate_dataset(cfg, path, n_trajectories=5, seed=0, num_workers=0)
        assert result.exists()

    def test_generate_correct_shapes(self, cfg, tmp_path):
        path = tmp_path / "out.h5"
        generate_dataset(cfg, path, n_trajectories=5, seed=0, num_workers=0)
        ds = TrajectoryDataset(path)
        T = int(cfg.dataset.trajectory_duration / cfg.simulation.control_dt)
        assert ds.states.shape == (5, T + 1, 4)
        assert ds.actions.shape == (5, T, 2)
        ds.close()


class TestTrajectoryDataset:
    def test_len(self, small_dataset):
        ds = TrajectoryDataset(small_dataset)
        assert len(ds) == 10
        ds.close()

    def test_getitem(self, small_dataset):
        ds = TrajectoryDataset(small_dataset)
        states, actions = ds[0]
        assert states.ndim == 2
        assert actions.ndim == 2
        ds.close()

    def test_roundtrip_consistency(self, cfg, tmp_path):
        """Regenerate trajectory and compare to stored data."""
        from trackzero.sim.simulator import Simulator
        from trackzero.data.multisine import generate_multisine_actions

        path = tmp_path / "rt.h5"
        seed = 42
        generate_dataset(cfg, path, n_trajectories=3, seed=seed, num_workers=0)

        ds = TrajectoryDataset(path)
        for i in range(3):
            stored_states, stored_actions = ds[i]

            # Regenerate with same seed
            rng = np.random.default_rng(seed + i)
            sim = Simulator(cfg)
            q0 = rng.uniform(cfg.dataset.initial_state.q_range[0],
                            cfg.dataset.initial_state.q_range[1], size=2)
            v0 = rng.uniform(cfg.dataset.initial_state.v_range[0],
                            cfg.dataset.initial_state.v_range[1], size=2)
            actions = generate_multisine_actions(rng, cfg.dataset, cfg.pendulum, cfg.simulation.control_dt)
            states = sim.rollout(actions, q0=q0, v0=v0)

            np.testing.assert_allclose(stored_states, states, atol=1e-12)
            np.testing.assert_allclose(stored_actions, actions, atol=1e-12)
        ds.close()

    def test_coverage_computation(self, small_dataset):
        ds = TrajectoryDataset(small_dataset)
        cov = ds.compute_coverage(n_bins=10)
        assert 0 <= cov["q_coverage"] <= 1
        assert 0 <= cov["v_coverage"] <= 1
        assert cov["q_hist"].shape == (10, 10)
        assert cov["v_hist"].shape == (10, 10)
        ds.close()
