"""MuJoCo simulator wrapper for the double pendulum."""

from __future__ import annotations

from typing import Optional

import mujoco
import numpy as np

from trackzero.config import Config, PendulumConfig, SimulationConfig
from trackzero.sim.pendulum_model import build_pendulum_xml


class Simulator:
    """Wraps a MuJoCo model/data pair for the double pendulum.

    Provides step (with substeps), reset, state access, and rollout.
    """

    def __init__(self, cfg: Optional[Config] = None):
        if cfg is None:
            cfg = Config()
        self.cfg = cfg
        self.pend_cfg = cfg.pendulum
        self.sim_cfg = cfg.simulation

        xml = build_pendulum_xml(self.pend_cfg, self.sim_cfg)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self._substeps = self.sim_cfg.substeps
        self._nq = self.model.nq  # 2
        self._nv = self.model.nv  # 2
        self._nu = self.model.nu  # 2

    # ---- state access ----

    def get_state(self) -> np.ndarray:
        """Return [q1, q2, dq1, dq2] as a (4,) array."""
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    def set_state(self, state: np.ndarray) -> None:
        """Set state from [q1, q2, dq1, dq2]."""
        self.data.qpos[:] = state[:self._nq]
        self.data.qvel[:] = state[self._nq:]
        mujoco.mj_forward(self.model, self.data)

    def get_qacc(self) -> np.ndarray:
        """Return current joint accelerations (after mj_forward)."""
        return self.data.qacc.copy()

    # ---- simulation ----

    def reset(
        self,
        q0: Optional[np.ndarray] = None,
        v0: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Reset to given or random state. Returns initial state."""
        mujoco.mj_resetData(self.model, self.data)

        if q0 is not None:
            self.data.qpos[:] = q0
        elif rng is not None:
            ds = self.cfg.dataset
            self.data.qpos[:] = rng.uniform(
                ds.initial_state.q_range[0],
                ds.initial_state.q_range[1],
                size=self._nq,
            )

        if v0 is not None:
            self.data.qvel[:] = v0
        elif rng is not None:
            ds = self.cfg.dataset
            self.data.qvel[:] = rng.uniform(
                ds.initial_state.v_range[0],
                ds.initial_state.v_range[1],
                size=self._nv,
            )

        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def step(self, action: np.ndarray) -> np.ndarray:
        """Apply action for one control step (multiple substeps). Returns new state."""
        self.data.ctrl[:] = np.clip(action, -self.pend_cfg.tau_max, self.pend_cfg.tau_max)
        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self.data)
        return self.get_state()

    def rollout(
        self,
        actions: np.ndarray,
        q0: Optional[np.ndarray] = None,
        v0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Roll out a sequence of actions from a given initial state.

        Args:
            actions: (T, 2) array of torques.
            q0: Initial joint positions. If None, uses current state.
            v0: Initial joint velocities. If None, uses current state.

        Returns:
            states: (T+1, 4) array of states (including initial state).
        """
        T = len(actions)
        states = np.zeros((T + 1, self._nq + self._nv))

        if q0 is not None or v0 is not None:
            self.reset(q0=q0, v0=v0)

        states[0] = self.get_state()
        for t in range(T):
            states[t + 1] = self.step(actions[t])

        return states

    def rollout_with_qacc(
        self,
        actions: np.ndarray,
        q0: Optional[np.ndarray] = None,
        v0: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Like rollout, but also records qacc before each step.

        The qacc is computed by calling mj_forward at the current state
        with the control applied, *before* stepping. This qacc is
        compatible with mj_inverse for exact torque recovery.

        Returns:
            states: (T+1, 4) array
            qaccs: (T, 2) array of accelerations at each control step
        """
        T = len(actions)
        states = np.zeros((T + 1, self._nq + self._nv))
        qaccs = np.zeros((T, self._nv))

        if q0 is not None or v0 is not None:
            self.reset(q0=q0, v0=v0)

        states[0] = self.get_state()
        for t in range(T):
            # Set control and compute qacc at current state
            clipped = np.clip(actions[t], -self.pend_cfg.tau_max, self.pend_cfg.tau_max)
            self.data.ctrl[:] = clipped
            mujoco.mj_forward(self.model, self.data)
            qaccs[t] = self.data.qacc.copy()
            # Now step
            for _ in range(self._substeps):
                mujoco.mj_step(self.model, self.data)
            states[t + 1] = self.get_state()

        return states, qaccs
