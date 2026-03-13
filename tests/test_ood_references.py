"""Tests for OOD reference trajectory generation."""

import numpy as np
import pytest

from trackzero.config import Config
from trackzero.data.ood_references import (
    OOD_ACTION_GENERATORS,
    generate_chirp_actions,
    generate_step_actions,
    generate_random_walk_actions,
    generate_sawtooth_actions,
    generate_pulse_actions,
    generate_ood_reference_data,
)


@pytest.fixture
def cfg():
    return Config()


class TestActionGenerators:
    @pytest.mark.parametrize("gen_name,gen_fn", list(OOD_ACTION_GENERATORS.items()))
    def test_shape_and_bounds(self, gen_name, gen_fn):
        rng = np.random.default_rng(0)
        T, n_joints, tau_max = 100, 2, 5.0
        actions = gen_fn(rng=rng, T=T, n_joints=n_joints, tau_max=tau_max, dt=0.01)
        assert actions.shape == (T, n_joints)
        assert np.all(np.abs(actions) <= tau_max + 1e-10)

    @pytest.mark.parametrize("gen_name,gen_fn", list(OOD_ACTION_GENERATORS.items()))
    def test_not_all_zeros(self, gen_name, gen_fn):
        rng = np.random.default_rng(1)
        actions = gen_fn(rng=rng, T=200, n_joints=2, tau_max=5.0, dt=0.01)
        assert np.max(np.abs(actions)) > 0.1

    @pytest.mark.parametrize("gen_name,gen_fn", list(OOD_ACTION_GENERATORS.items()))
    def test_deterministic(self, gen_name, gen_fn):
        a1 = gen_fn(rng=np.random.default_rng(42), T=100, n_joints=2, tau_max=5.0, dt=0.01)
        a2 = gen_fn(rng=np.random.default_rng(42), T=100, n_joints=2, tau_max=5.0, dt=0.01)
        np.testing.assert_array_equal(a1, a2)


class TestOODReferenceData:
    def test_mixed_ood_shape(self, cfg):
        states, actions = generate_ood_reference_data(cfg, 10, action_type="mixed_ood", seed=0)
        T = int(cfg.dataset.trajectory_duration / cfg.simulation.control_dt)
        assert states.shape == (10, T + 1, 4)
        assert actions.shape == (10, T, 2)

    def test_single_type(self, cfg):
        states, actions = generate_ood_reference_data(cfg, 5, action_type="chirp", seed=0)
        assert states.shape[0] == 5

    def test_actions_within_bounds(self, cfg):
        _, actions = generate_ood_reference_data(cfg, 10, seed=0)
        assert np.all(np.abs(actions) <= cfg.pendulum.tau_max + 1e-10)
