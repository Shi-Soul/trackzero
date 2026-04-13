"""Generate MuJoCo XML for planar chain models (2-link to N-link) from config."""

from __future__ import annotations

from trackzero.config import PendulumConfig, SimulationConfig

_INTEGRATOR_MAP = {
    "RK4": "RK4",
    "Euler": "Euler",
    "implicit": "implicit",
    "implicitfast": "implicitfast",
}


def build_chain_xml(
    n_links: int = 2,
    pend: PendulumConfig | None = None,
    sim: SimulationConfig | None = None,
) -> str:
    """Return MuJoCo XML for a planar N-link serial chain.

    Generalizes the double pendulum to arbitrary chain length.
    Each link is a capsule with identical mass/inertia, connected by hinge joints.
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

    # Build nested body XML for N links
    indent = "    "
    body_lines = []
    close_lines = []
    for i in range(1, n_links + 1):
        depth = i + 1
        prefix = indent * depth
        pos = '0 0 0' if i == 1 else f'0 0 -{L}'
        body_lines.append(f'{prefix}<body name="link{i}" pos="{pos}">')
        body_lines.append(f'{prefix}  <joint name="joint{i}" type="hinge"/>')
        body_lines.append(
            f'{prefix}  <inertial pos="0 0 -{half_L}" mass="{m}"'
            f' diaginertia="{ix} {iy} {iz}"/>'
        )
        body_lines.append(f'{prefix}  <geom name="geom{i}" fromto="0 0 0 0 0 -{L}"/>')
        close_lines.append(f'{prefix}</body>')

    bodies_xml = "\n".join(body_lines) + "\n" + "\n".join(reversed(close_lines))

    # Build actuator XML
    actuator_lines = []
    for i in range(1, n_links + 1):
        actuator_lines.append(
            f'    <motor name="motor{i}" joint="joint{i}" ctrllimited="true"'
            f' ctrlrange="{-tau} {tau}"/>'
        )
    actuators_xml = "\n".join(actuator_lines)

    xml = f"""\
<mujoco model="chain_{n_links}link">
  <option gravity="0 0 -{g}" timestep="{dt}" integrator="{integrator}"/>

  <default>
    <joint axis="0 1 0" damping="{damp}" limited="false"/>
    <geom type="capsule" size="0.02" rgba="0.4 0.6 0.8 1"/>
  </default>

  <worldbody>
{bodies_xml}
  </worldbody>

  <actuator>
{actuators_xml}
  </actuator>
</mujoco>
"""
    return xml


def build_pendulum_xml(
    pend: PendulumConfig | None = None,
    sim: SimulationConfig | None = None,
) -> str:
    """Return a MuJoCo XML string for a planar double pendulum.

    Convenience wrapper around build_chain_xml with n_links=2.
    """
    return build_chain_xml(n_links=2, pend=pend, sim=sim)
