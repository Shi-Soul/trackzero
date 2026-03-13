"""Tests for the inverse dynamics oracle."""

import numpy as np
import pytest

from trackzero.config import Config
from trackzero.oracle.inverse_dynamics import InverseDynamicsOracle
from trackzero.sim.simulator import Simulator


@pytest.fixture
def cfg():
    return Config()


@pytest.fixture
def oracle(cfg):
    return InverseDynamicsOracle(cfg)


@pytest.fixture
def sim(cfg):
    return Simulator(cfg)


class TestInverseDynamicsOracle:
    def test_roundtrip_accuracy(self, oracle, sim):
        """Forward sim -> record qacc -> mj_inverse should recover exact torques."""
        rng = np.random.default_rng(42)

        for _ in range(10):
            T = 100
            actions = rng.uniform(-4, 4, size=(T, 2))
            q0 = rng.uniform(-np.pi, np.pi, size=2)
            v0 = rng.uniform(-3, 3, size=2)

            states, qaccs = sim.rollout_with_qacc(actions, q0=q0, v0=v0)
            recovered = oracle.recover_actions(states, qaccs)
            clipped = np.clip(actions, -sim.pend_cfg.tau_max, sim.pend_cfg.tau_max)

            max_err = np.max(np.abs(recovered - clipped))
            assert max_err < 1e-10, f"Round-trip error {max_err}"

    def test_zero_torque_free_fall(self, oracle, sim):
        """Free-fall (zero torque) should be exactly recoverable."""
        sim.reset(q0=np.array([0.5, -0.3]), v0=np.zeros(2))
        T = 50
        actions = np.zeros((T, 2))
        states, qaccs = sim.rollout_with_qacc(actions, q0=np.array([0.5, -0.3]), v0=np.zeros(2))
        recovered = oracle.recover_actions(states, qaccs)

        max_err = np.max(np.abs(recovered))
        assert max_err < 1e-10

    def test_as_policy_interface(self, oracle):
        """as_policy() should return a callable with correct signature."""
        policy = oracle.as_policy()
        state = np.array([0.1, -0.2, 0.5, -0.3])
        next_state = np.array([0.11, -0.19, 0.45, -0.28])
        action = policy(state, next_state)
        assert action.shape == (2,)

    def test_compute_torque_from_qacc(self, oracle):
        """Direct qacc -> torque computation should work."""
        qpos = np.array([0.0, 0.0])
        qvel = np.array([0.0, 0.0])
        qacc = np.array([1.0, -1.0])
        torque = oracle.compute_torque_from_qacc(qpos, qvel, qacc)
        assert torque.shape == (2,)
        # With non-zero qacc, torque should be non-zero
        assert not np.allclose(torque, 0)

    def test_shooting_roundtrip(self, oracle, sim):
        """Shooting oracle should recover exact torques through the simulator."""
        rng = np.random.default_rng(99)

        for _ in range(5):
            T = 20
            actions = rng.uniform(-4, 4, size=(T, 2))
            q0 = rng.uniform(-np.pi, np.pi, size=2)
            v0 = rng.uniform(-2, 2, size=2)

            states = sim.rollout(actions, q0=q0, v0=v0)
            clipped = np.clip(actions, -sim.pend_cfg.tau_max, sim.pend_cfg.tau_max)

            for t in range(T):
                recovered = oracle.compute_torque_shooting(states[t], states[t + 1])
                err = np.max(np.abs(recovered - clipped[t]))
                assert err < 1e-8, f"Shooting error {err} at step {t}"

    def test_shooting_as_policy_through_harness(self, oracle, sim, cfg):
        """Shooting oracle through eval harness should give near-zero error.

        Note: MSE is not exactly zero due to chaotic compounding of
        machine-epsilon per-step errors over the trajectory. Per-step
        error is ~1e-14, but over 100 steps of a chaotic double pendulum,
        MSE may reach ~1e-8. This is still 5+ orders of magnitude below
        any learned policy.
        """
        from trackzero.eval.harness import EvalHarness

        rng = np.random.default_rng(42)
        T = 100
        actions = rng.uniform(-3, 3, size=(T, 2))
        q0 = rng.uniform(-np.pi, np.pi, size=2)
        v0 = rng.uniform(-2, 2, size=2)
        ref_states = sim.rollout(actions, q0=q0, v0=v0)

        harness = EvalHarness(cfg)
        policy = oracle.as_policy(mode="shooting")
        result = harness.evaluate_trajectory(policy, ref_states, actions)

        assert result.mse_q < 1e-6, f"Shooting oracle MSE_q too high: {result.mse_q}"
        assert result.mse_total < 1e-6, f"Shooting oracle MSE_total too high: {result.mse_total}"
