"""Generate MuJoCo XML for a double pendulum from config."""

from __future__ import annotations

from trackzero.config import PendulumConfig, SimulationConfig

_INTEGRATOR_MAP = {
    "RK4": "RK4",
    "Euler": "Euler",
    "implicit": "implicit",
    "implicitfast": "implicitfast",
}


def build_pendulum_xml(
    pend: PendulumConfig | None = None,
    sim: SimulationConfig | None = None,
) -> str:
    """Return a MuJoCo XML string for a planar double pendulum.

    The pendulum swings in the x-z plane. Joint axes are along y.
    Each link is a capsule with mass concentrated at its center.
    """
    if pend is None:
        pend = PendulumConfig()
    if sim is None:
        sim = SimulationConfig()

    L = pend.link_length
    half_L = L / 2.0
    m = pend.link_mass
    ix, iy, iz = pend.link_inertia
    damp = pend.joint_damping
    tau = pend.tau_max
    g = pend.gravity
    dt = sim.dt
    integrator = _INTEGRATOR_MAP.get(sim.integrator, "RK4")

    xml = f"""\
<mujoco model="double_pendulum">
  <option gravity="0 0 -{g}" timestep="{dt}" integrator="{integrator}"/>

  <default>
    <joint axis="0 1 0" damping="{damp}" limited="false"/>
    <geom type="capsule" size="0.02" rgba="0.4 0.6 0.8 1"/>
  </default>

  <worldbody>
    <!-- Pivot fixed at origin -->
    <body name="link1" pos="0 0 0">
      <joint name="joint1" type="hinge"/>
      <inertial pos="0 0 -{half_L}" mass="{m}"
                diaginertia="{ix} {iy} {iz}"/>
      <geom name="geom1" fromto="0 0 0 0 0 -{L}"/>
      <body name="link2" pos="0 0 -{L}">
        <joint name="joint2" type="hinge"/>
        <inertial pos="0 0 -{half_L}" mass="{m}"
                  diaginertia="{ix} {iy} {iz}"/>
        <geom name="geom2" fromto="0 0 0 0 0 -{L}"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="motor1" joint="joint1" ctrllimited="true"
           ctrlrange="{-tau} {tau}"/>
    <motor name="motor2" joint="joint2" ctrllimited="true"
           ctrlrange="{-tau} {tau}"/>
  </actuator>
</mujoco>
"""
    return xml
