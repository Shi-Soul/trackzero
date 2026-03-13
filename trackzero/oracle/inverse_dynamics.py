"""Inverse dynamics oracle using MuJoCo's mj_inverse and Newton shooting."""

from __future__ import annotations

from typing import Optional

import mujoco
import numpy as np

from trackzero.config import Config
from trackzero.sim.pendulum_model import build_pendulum_xml


class InverseDynamicsOracle:
    """Computes exact torques for desired transitions using mj_inverse.

    Three modes:
    1. Given (state, qacc): directly run mj_inverse (exact roundtrip).
    2. Given (state, next_state): finite-difference qacc + mj_inverse (approximate).
    3. Given (state, next_state): Newton shooting through the simulator (exact).
    """

    def __init__(self, cfg: Optional[Config] = None):
        if cfg is None:
            cfg = Config()
        self.cfg = cfg

        xml = build_pendulum_xml(cfg.pendulum, cfg.simulation)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        # Separate MjData for shooting (avoids mutating self.data)
        self._shoot_data = mujoco.MjData(self.model)

        self._control_dt = cfg.simulation.control_dt
        self._substeps = cfg.simulation.substeps
        self._tau_max = cfg.pendulum.tau_max
        self._nq = self.model.nq
        self._nv = self.model.nv

    def compute_torque_from_qacc(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        qacc: np.ndarray,
    ) -> np.ndarray:
        """Run mj_inverse to get torques for given state and acceleration.

        Args:
            qpos: Joint positions (2,)
            qvel: Joint velocities (2,)
            qacc: Joint accelerations (2,)

        Returns:
            torques: (2,) array of required torques.
        """
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.qacc[:] = qacc
        mujoco.mj_inverse(self.model, self.data)
        return self.data.qfrc_inverse[:self._nv].copy()

    def compute_torque(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
    ) -> np.ndarray:
        """Compute torque to transition from state to next_state.

        Uses finite differences to estimate qacc, then mj_inverse.
        This is approximate for large dt; for oracle use, prefer
        compute_torque_from_qacc with recorded qacc from forward sim.

        Args:
            state: [q1, q2, dq1, dq2]
            next_state: [q1', q2', dq1', dq2']

        Returns:
            torques: (2,)
        """
        qpos = state[:self._nq]
        qvel = state[self._nq:]
        next_qvel = next_state[self._nq:]

        # Finite-difference acceleration
        qacc = (next_qvel - qvel) / self._control_dt

        return self.compute_torque_from_qacc(qpos, qvel, qacc)

    def recover_actions(
        self,
        states: np.ndarray,
        qaccs: np.ndarray,
    ) -> np.ndarray:
        """Recover full action sequence from states and recorded accelerations.

        Args:
            states: (T+1, 4) state trajectory
            qaccs: (T, 2) recorded accelerations from forward simulation

        Returns:
            actions: (T, 2) recovered torques
        """
        T = len(qaccs)
        actions = np.zeros((T, self._nv))
        for t in range(T):
            actions[t] = self.compute_torque_from_qacc(
                states[t, :self._nq],
                states[t, self._nq:],
                qaccs[t],
            )
        return actions

    def _simulate_step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Run one control step (with substeps) in the shooting data, return next state."""
        self._shoot_data.qpos[:] = state[:self._nq]
        self._shoot_data.qvel[:] = state[self._nq:]
        self._shoot_data.ctrl[:] = np.clip(action, -self._tau_max, self._tau_max)
        mujoco.mj_forward(self.model, self._shoot_data)
        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self._shoot_data)
        return np.concatenate([
            self._shoot_data.qpos.copy(), self._shoot_data.qvel.copy()
        ])

    def compute_torque_shooting(
        self,
        current_state: np.ndarray,
        ref_next_state: np.ndarray,
        max_iter: int = 10,
        tol: float = 1e-12,
    ) -> np.ndarray:
        """Find the exact torque via Newton shooting through the simulator.

        Uses finite-difference mj_inverse as the initial guess, then
        Newton-refines by solving for the velocity residual. Handles
        torque saturation gracefully via least-squares solve.

        Args:
            current_state: [q1, q2, dq1, dq2]
            ref_next_state: [q1', q2', dq1', dq2']
            max_iter: Maximum Newton iterations.
            tol: Convergence tolerance on velocity residual.

        Returns:
            torques: (2,) exact torques (may be clipped to limits).
        """
        # Initial guess from finite-difference inverse dynamics
        tau = self.compute_torque(current_state, ref_next_state)

        ref_qvel = ref_next_state[self._nq:]
        eps = 1e-7

        for _ in range(max_iter):
            predicted = self._simulate_step(current_state, tau)
            residual = predicted[self._nq:] - ref_qvel

            if np.max(np.abs(residual)) < tol:
                break

            # Finite-difference Jacobian d(next_qvel)/d(tau)
            # Use central differences for better accuracy
            J = np.zeros((self._nv, self._nv))
            for j in range(self._nv):
                tau_plus = tau.copy()
                tau_minus = tau.copy()
                tau_plus[j] += eps
                tau_minus[j] -= eps
                pred_plus = self._simulate_step(current_state, tau_plus)
                pred_minus = self._simulate_step(current_state, tau_minus)
                J[:, j] = (pred_plus[self._nq:] - pred_minus[self._nq:]) / (2 * eps)

            # Use least-squares to handle saturated actuators (singular J)
            delta, _, _, _ = np.linalg.lstsq(J, residual, rcond=None)
            tau = tau - delta

        return tau

    def as_policy(self, mode: str = "shooting"):
        """Return a callable policy(current_state, ref_next_state) -> action.

        Args:
            mode: "shooting" for exact Newton solver (default),
                  "finite_difference" for fast approximate mj_inverse.
        """
        if mode == "shooting":
            def policy(current_state: np.ndarray, ref_next_state: np.ndarray) -> np.ndarray:
                return self.compute_torque_shooting(current_state, ref_next_state)
        elif mode == "finite_difference":
            def policy(current_state: np.ndarray, ref_next_state: np.ndarray) -> np.ndarray:
                return self.compute_torque(current_state, ref_next_state)
        else:
            raise ValueError(f"Unknown oracle mode: {mode}")
        return policy
