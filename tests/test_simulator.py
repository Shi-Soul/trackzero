"""Tests for the MuJoCo double pendulum simulator."""

import numpy as np
import pytest
import mujoco

from trackzero.config import Config
from trackzero.sim.simulator import Simulator
from trackzero.sim.pendulum_model import build_pendulum_xml


@pytest.fixture
def cfg():
    return Config()


@pytest.fixture
def sim(cfg):
    return Simulator(cfg)


class TestPendulumModel:
    def test_xml_generates(self, cfg):
        xml = build_pendulum_xml(cfg.pendulum, cfg.simulation)
        assert "<mujoco" in xml
        assert "joint1" in xml
        assert "joint2" in xml

    def test_xml_loads_in_mujoco(self, cfg):
        xml = build_pendulum_xml(cfg.pendulum, cfg.simulation)
        model = mujoco.MjModel.from_xml_string(xml)
        assert model.nq == 2
        assert model.nv == 2
        assert model.nu == 2


class TestSimulator:
    def test_reset_default(self, sim):
        state = sim.reset()
        assert state.shape == (4,)
        np.testing.assert_array_equal(state, [0, 0, 0, 0])

    def test_reset_with_state(self, sim):
        q0 = np.array([1.0, -0.5])
        v0 = np.array([0.1, -0.2])
        state = sim.reset(q0=q0, v0=v0)
        np.testing.assert_allclose(state[:2], q0)
        np.testing.assert_allclose(state[2:], v0)

    def test_reset_random(self, sim):
        rng = np.random.default_rng(42)
        state = sim.reset(rng=rng)
        assert state.shape == (4,)
        # Should not be zero (with overwhelming probability)
        assert not np.allclose(state, 0)

    def test_step_changes_state(self, sim):
        sim.reset(q0=np.array([0.5, 0.0]), v0=np.zeros(2))
        state0 = sim.get_state()
        state1 = sim.step(np.zeros(2))
        # Gravity should cause motion
        assert not np.allclose(state0, state1)

    def test_step_with_torque(self, sim):
        sim.reset(q0=np.zeros(2), v0=np.zeros(2))
        state = sim.step(np.array([1.0, 0.0]))
        # Applied torque should cause angular velocity
        assert state[2] != 0  # dq1 changed

    def test_torque_clipping(self, sim):
        """Actions beyond tau_max should be clipped."""
        sim.reset(q0=np.zeros(2), v0=np.zeros(2))
        tau_max = sim.pend_cfg.tau_max

        # Apply excessive torque
        state_clipped = sim.step(np.array([100.0, 100.0]))
        sim.reset(q0=np.zeros(2), v0=np.zeros(2))
        state_max = sim.step(np.array([tau_max, tau_max]))

        np.testing.assert_allclose(state_clipped, state_max)

    def test_rollout_shape(self, sim):
        T = 100
        actions = np.zeros((T, 2))
        states = sim.rollout(actions, q0=np.zeros(2), v0=np.zeros(2))
        assert states.shape == (T + 1, 4)

    def test_rollout_with_qacc_shape(self, sim):
        T = 50
        actions = np.random.randn(T, 2)
        states, qaccs = sim.rollout_with_qacc(
            actions, q0=np.zeros(2), v0=np.zeros(2)
        )
        assert states.shape == (T + 1, 4)
        assert qaccs.shape == (T, 2)

    def test_inverse_dynamics_roundtrip(self, sim):
        """Key test: forward sim -> record qacc -> mj_inverse -> recover torque."""
        rng = np.random.default_rng(123)
        T = 50
        actions = rng.uniform(-3, 3, size=(T, 2))

        q0 = rng.uniform(-np.pi, np.pi, size=2)
        v0 = rng.uniform(-2, 2, size=2)
        states, qaccs = sim.rollout_with_qacc(actions, q0=q0, v0=v0)

        # Use mj_inverse to recover torques
        from trackzero.oracle.inverse_dynamics import InverseDynamicsOracle
        oracle = InverseDynamicsOracle(sim.cfg)

        recovered = oracle.recover_actions(states, qaccs)

        # Clipped actions are what was actually applied
        clipped_actions = np.clip(actions, -sim.pend_cfg.tau_max, sim.pend_cfg.tau_max)

        max_err = np.max(np.abs(recovered - clipped_actions))
        assert max_err < 1e-10, f"Round-trip error {max_err} exceeds threshold"

    def test_gravity_effect(self, sim):
        """Pendulum starting horizontal should fall under gravity."""
        sim.reset(q0=np.array([np.pi / 2, 0.0]), v0=np.zeros(2))
        for _ in range(10):
            sim.step(np.zeros(2))
        state = sim.get_state()
        # Should have gained angular velocity
        assert abs(state[2]) > 0.1
