"""Tests for the evaluation harness."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from trackzero.config import Config
from trackzero.eval.harness import EvalHarness, EvalSummary, angular_error
from trackzero.sim.simulator import Simulator
from trackzero.oracle.inverse_dynamics import InverseDynamicsOracle


@pytest.fixture
def cfg():
    return Config()


class TestAngularError:
    def test_zero_error(self):
        err = angular_error(np.array([1.0]), np.array([1.0]))
        np.testing.assert_allclose(err, 0.0, atol=1e-15)

    def test_wraparound(self):
        """Error between pi and -pi should be ~0, not 2*pi."""
        err = angular_error(np.array([np.pi - 0.01]), np.array([-np.pi + 0.01]))
        assert abs(err[0]) < 0.05

    def test_sign(self):
        err = angular_error(np.array([1.0]), np.array([0.0]))
        assert err[0] > 0  # ref > actual -> positive error


class TestEvalHarness:
    def test_oracle_near_zero_error(self, cfg):
        """Shooting oracle through eval harness should give near-zero error.

        Per-step error is ~1e-14 but compounds over the chaotic trajectory.
        Over 50 steps, MSE should stay well below 1e-6.
        """
        sim = Simulator(cfg)
        oracle = InverseDynamicsOracle(cfg)
        harness = EvalHarness(cfg)

        rng = np.random.default_rng(42)
        T = 50

        # Generate a reference trajectory
        actions = rng.uniform(-3, 3, size=(T, 2))
        q0 = rng.uniform(-np.pi, np.pi, size=2)
        v0 = rng.uniform(-2, 2, size=2)
        ref_states = sim.rollout(actions, q0=q0, v0=v0)

        # Evaluate shooting oracle policy
        policy = oracle.as_policy(mode="shooting")
        result = harness.evaluate_trajectory(policy, ref_states, actions)

        assert result.mse_q < 1e-6, f"MSE_q too high: {result.mse_q}"
        assert result.mse_total < 1e-6, f"MSE_total too high: {result.mse_total}"

    def test_openloop_replay_zero_error(self, cfg):
        """Open-loop replay of ref_actions should give zero error."""
        sim = Simulator(cfg)
        harness = EvalHarness(cfg)

        rng = np.random.default_rng(42)
        T = 50
        actions = rng.uniform(-3, 3, size=(T, 2))
        q0 = rng.uniform(-np.pi, np.pi, size=2)
        v0 = rng.uniform(-2, 2, size=2)
        ref_states = sim.rollout(actions, q0=q0, v0=v0)

        result = harness.evaluate_trajectory_openloop(ref_states, actions)
        assert result.mse_total < 1e-20, f"Openloop error not zero: {result.mse_total}"

    def test_zero_policy_has_error(self, cfg):
        """Zero-torque policy should have measurable tracking error."""
        sim = Simulator(cfg)
        harness = EvalHarness(cfg)

        rng = np.random.default_rng(42)
        T = 50
        actions = rng.uniform(-3, 3, size=(T, 2))
        q0 = rng.uniform(-1, 1, size=2)
        v0 = rng.uniform(-1, 1, size=2)
        ref_states = sim.rollout(actions, q0=q0, v0=v0)

        zero_policy = lambda s, ns: np.zeros(2)
        result = harness.evaluate_trajectory(zero_policy, ref_states, actions)

        assert result.mse_total > 0.01

    def test_evaluate_policy_batch(self, cfg):
        """Batch evaluation should produce correct summary."""
        sim = Simulator(cfg)
        oracle = InverseDynamicsOracle(cfg)
        harness = EvalHarness(cfg)

        rng = np.random.default_rng(42)
        N, T = 5, 30

        all_states = np.zeros((N, T + 1, 4))
        all_actions = np.zeros((N, T, 2))
        for i in range(N):
            acts = rng.uniform(-3, 3, size=(T, 2))
            q0 = rng.uniform(-np.pi, np.pi, size=2)
            v0 = rng.uniform(-1, 1, size=2)
            all_states[i] = sim.rollout(acts, q0=q0, v0=v0)
            all_actions[i] = acts

        policy = oracle.as_policy()
        summary = harness.evaluate_policy(policy, all_states, all_actions)

        assert summary.n_trajectories == N
        assert len(summary.results) == N
        assert summary.mean_mse_total >= 0

    def test_eval_summary_json_roundtrip(self, cfg, tmp_path):
        """EvalSummary should survive JSON serialization."""
        sim = Simulator(cfg)
        oracle = InverseDynamicsOracle(cfg)
        harness = EvalHarness(cfg)

        rng = np.random.default_rng(42)
        T = 20
        actions = rng.uniform(-2, 2, size=(T, 2))
        states = sim.rollout(actions, q0=np.zeros(2), v0=np.zeros(2))

        summary = harness.evaluate_policy(
            oracle.as_policy(),
            states[np.newaxis],
            actions[np.newaxis],
        )

        path = tmp_path / "results.json"
        summary.to_json(path)
        loaded = EvalSummary.from_json(path)

        assert loaded.n_trajectories == summary.n_trajectories
        assert abs(loaded.mean_mse_total - summary.mean_mse_total) < 1e-12
