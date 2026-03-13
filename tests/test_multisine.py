"""Tests for multisine signal generation."""

import numpy as np
import pytest

from trackzero.config import Config
from trackzero.data.multisine import (
    sample_multisine_params,
    evaluate_multisine,
    generate_multisine_actions,
)


@pytest.fixture
def cfg():
    return Config()


class TestMultisine:
    def test_sample_params_shape(self, cfg):
        rng = np.random.default_rng(42)
        params = sample_multisine_params(rng, cfg.dataset, cfg.pendulum)

        K = params["K"]
        assert 3 <= K <= 8
        assert params["freqs"].shape == (2, K)
        assert params["amplitudes"].shape == (2, K)
        assert params["phases"].shape == (2, K)

    def test_freq_range(self, cfg):
        rng = np.random.default_rng(42)
        for _ in range(20):
            params = sample_multisine_params(rng, cfg.dataset, cfg.pendulum)
            assert np.all(params["freqs"] >= 0.1)
            assert np.all(params["freqs"] <= 3.0)

    def test_evaluate_clipping(self, cfg):
        rng = np.random.default_rng(42)
        params = sample_multisine_params(rng, cfg.dataset, cfg.pendulum)
        times = np.arange(500) * 0.01
        actions = evaluate_multisine(params, times, cfg.pendulum.tau_max)

        assert actions.shape == (500, 2)
        assert np.all(actions >= -cfg.pendulum.tau_max)
        assert np.all(actions <= cfg.pendulum.tau_max)

    def test_generate_actions_shape(self, cfg):
        rng = np.random.default_rng(42)
        actions = generate_multisine_actions(rng, cfg.dataset, cfg.pendulum, cfg.simulation.control_dt)
        T = int(cfg.dataset.trajectory_duration / cfg.simulation.control_dt)
        assert actions.shape == (T, 2)

    def test_different_seeds_different_signals(self, cfg):
        actions1 = generate_multisine_actions(
            np.random.default_rng(1), cfg.dataset, cfg.pendulum, cfg.simulation.control_dt
        )
        actions2 = generate_multisine_actions(
            np.random.default_rng(2), cfg.dataset, cfg.pendulum, cfg.simulation.control_dt
        )
        assert not np.allclose(actions1, actions2)

    def test_signal_is_smooth(self, cfg):
        """Multisine signals should be smooth (bounded derivative)."""
        rng = np.random.default_rng(42)
        actions = generate_multisine_actions(rng, cfg.dataset, cfg.pendulum, cfg.simulation.control_dt)
        # Finite differences should be bounded
        diffs = np.diff(actions, axis=0)
        max_diff = np.max(np.abs(diffs))
        # With max freq 3Hz and dt=0.01, max derivative ~ 2*pi*3*tau_max/sqrt(K)
        # Should be well under tau_max per step
        assert max_diff < cfg.pendulum.tau_max
