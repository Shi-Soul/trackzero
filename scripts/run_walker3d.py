#!/usr/bin/env python3
"""3D walker: the critical Stage 3→4 bridge experiment.

Tests TRACK-ZERO on a 3D bipedal walker with freejoint, gravity, and
ground contact — combining all challenges encountered in Stage 3.

System: torso + 2 legs (hip-pitch, hip-yaw, knee each = 6 actuated DOF)
        + freejoint (7 qpos, 6 qvel) = nq=13, nv=12, nu=6
        Ground contact, gravity, 3D rotations.

Architectures tested:
  1. raw_mlp — baseline
  2. factored — does structured decomposition help?
  3. factored + per-link contact flags — full TRACK-ZERO recipe

Usage:
    CUDA_VISIBLE_DEVICES=6 python -m scripts.run_walker3d
"""
import argparse, json, os, time
import mujoco
import numpy as np
import torch

from trackzero.config import Config
from scripts.models_structured import RawMLP, FactoredMLP, ContactFactoredMLP
from scripts.run_structured import train_model

WALKER3D_XML = """
<mujoco model="walker3d">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.9 0.8 1"
          friction="1 0.5 0.5" conaffinity="1" contype="1"/>
    <body name="torso" pos="0 0 0.75">
      <freejoint name="root"/>
      <geom name="torso_g" type="capsule" fromto="0 -0.1 0 0 0.1 0"
            size="0.06" mass="5" conaffinity="1" contype="1"/>
      <!-- Left leg -->
      <body name="l_thigh" pos="0 0.1 0">
        <joint name="l_hip_y" type="hinge" axis="0 1 0"
               range="-1.57 0.5" damping="0.5"/>
        <joint name="l_hip_x" type="hinge" axis="1 0 0"
               range="-0.5 0.5" damping="0.5"/>
        <geom name="l_thigh_g" type="capsule" fromto="0 0 0 0 0 -0.28"
              size="0.04" mass="2" conaffinity="1" contype="1"/>
        <body name="l_shin" pos="0 0 -0.28">
          <joint name="l_knee" type="hinge" axis="0 1 0"
                 range="-2.5 0" damping="0.5"/>
          <geom name="l_shin_g" type="capsule" fromto="0 0 0 0 0 -0.28"
                size="0.03" mass="1" conaffinity="1" contype="1"/>
          <body name="l_foot" pos="0 0 -0.28">
            <geom name="l_foot_g" type="box" size="0.08 0.04 0.02"
                  pos="0.03 0 0" mass="0.5" conaffinity="1" contype="1"/>
          </body>
        </body>
      </body>
      <!-- Right leg -->
      <body name="r_thigh" pos="0 -0.1 0">
        <joint name="r_hip_y" type="hinge" axis="0 1 0"
               range="-1.57 0.5" damping="0.5"/>
        <joint name="r_hip_x" type="hinge" axis="1 0 0"
               range="-0.5 0.5" damping="0.5"/>
        <geom name="r_thigh_g" type="capsule" fromto="0 0 0 0 0 -0.28"
              size="0.04" mass="2" conaffinity="1" contype="1"/>
        <body name="r_shin" pos="0 0 -0.28">
          <joint name="r_knee" type="hinge" axis="0 1 0"
                 range="-2.5 0" damping="0.5"/>
          <geom name="r_shin_g" type="capsule" fromto="0 0 0 0 0 -0.28"
                size="0.03" mass="1" conaffinity="1" contype="1"/>
          <body name="r_foot" pos="0 0 -0.28">
            <geom name="r_foot_g" type="box" size="0.08 0.04 0.02"
                  pos="0.03 0 0" mass="0.5" conaffinity="1" contype="1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="l_hip_y" ctrlrange="-30 30"/>
    <motor joint="l_hip_x" ctrlrange="-30 30"/>
    <motor joint="l_knee" ctrlrange="-30 30"/>
    <motor joint="r_hip_y" ctrlrange="-30 30"/>
    <motor joint="r_hip_x" ctrlrange="-30 30"/>
    <motor joint="r_knee" ctrlrange="-30 30"/>
  </actuator>
</mujoco>
"""


def quat_to_rotvec(q):
    """Quaternion (w,x,y,z) -> rotation vector (3D)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    sin_half = np.sqrt(x*x + y*y + z*z)
    angle = 2.0 * np.arctan2(sin_half, np.abs(w))
    safe = np.where(sin_half > 1e-8, angle / sin_half, 2.0)
    sign = np.sign(w)
    sign = np.where(sign == 0, 1.0, sign)
    return np.stack([x, y, z], axis=-1) * (safe * sign)[..., None]


def rotvec_to_quat(rv):
    """Rotation vector (3D) -> quaternion (w,x,y,z)."""
    angle = np.linalg.norm(rv, axis=-1, keepdims=True)
    half = angle / 2.0
    safe = np.where(angle > 1e-8, np.sin(half) / angle, 0.5)
    w = np.cos(half)
    xyz = rv * safe
    return np.concatenate([w, xyz], axis=-1)


def mj_to_flat(qpos, qvel):
    """Convert MuJoCo state to flat Euclidean representation.
    qpos: [pos3, quat4, joints6] = 13
    qvel: [linvel3, angvel3, jointvel6] = 12
    flat: [pos3, rotvec3, joints6, vel12] = 24
    """
    pos = qpos[:3]
    quat = qpos[3:7]
    joints = qpos[7:]
    rv = quat_to_rotvec(quat)
    return np.concatenate([pos, rv, joints, qvel]).astype(np.float32)


def flat_to_mj(flat, model, data):
    """Set MuJoCo state from flat representation."""
    pos = flat[:3]
    rv = flat[3:6]
    joints = flat[6:12]
    vel = flat[12:24]
    quat = rotvec_to_quat(rv)
    data.qpos[:3] = pos
    data.qpos[3:7] = quat
    data.qpos[7:] = joints
    data.qvel[:] = vel
    mujoco.mj_forward(model, data)


# Geom-to-body mapping for per-link contact
# geom 0=floor, 1=torso, 2=l_thigh, 3=l_shin, 4=l_foot,
#                         5=r_thigh, 6=r_shin, 7=r_foot
N_CONTACT_BODIES = 7  # torso + 6 leg parts
CONTACT_GEOM_OFFSET = 1  # geom 1 maps to body index 0


def get_perlink_flags(data, n_geom_bodies=N_CONTACT_BODIES):
    """Return binary contact flag per body geom."""
    flags = np.zeros(n_geom_bodies, dtype=np.float32)
    for ci in range(data.ncon):
        g1 = data.contact[ci].geom1
        g2 = data.contact[ci].geom2
        for g in [g1, g2]:
            idx = g - CONTACT_GEOM_OFFSET
            if 0 <= idx < n_geom_bodies:
                flags[idx] = 1.0
    return flags


def gen_data(n_traj, traj_len, tau_max, seed=42):
    """Random rollout data with per-link contact flags."""
    rng = np.random.default_rng(seed)
    m = mujoco.MjModel.from_xml_string(WALKER3D_XML)
    d = mujoco.MjData(m)

    all_s, all_sn, all_a, all_flags = [], [], [], []

    for _ in range(n_traj):
        mujoco.mj_resetData(m, d)
        # Diverse initial states: random height, orientation, joint angles
        d.qpos[0:2] = rng.uniform(-0.5, 0.5, 2)  # xy
        d.qpos[2] = rng.uniform(0.2, 1.2)  # height
        # Random quaternion
        u = rng.standard_normal(4)
        d.qpos[3:7] = u / np.linalg.norm(u)
        d.qpos[7:] = rng.uniform(-1.0, 0.5, m.nq - 7)
        d.qvel[:] = rng.uniform(-3, 3, m.nv)
        mujoco.mj_forward(m, d)

        for _ in range(traj_len):
            s = mj_to_flat(d.qpos.copy(), d.qvel.copy())
            flags = get_perlink_flags(d)
            tau = rng.uniform(-tau_max, tau_max, m.nu).astype(np.float32)

            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            sn = mj_to_flat(d.qpos.copy(), d.qvel.copy())

            all_s.append(s)
            all_sn.append(sn)
            all_a.append(tau)
            all_flags.append(flags)

    S = np.array(all_s, dtype=np.float32)
    SN = np.array(all_sn, dtype=np.float32)
    A = np.array(all_a, dtype=np.float32)
    F = np.array(all_flags, dtype=np.float32)

    cfrac = (F.max(axis=1) > 0).mean()
    per_frac = F.mean(axis=0)
    print(f"  {len(S):,} pairs, any_contact={cfrac:.1%}")
    print(f"  Per-body: {np.array2string(per_frac, precision=2)}")
    return S, SN, A, F


def gen_benchmark(n_per_fam, traj_len, tau_max, seed=99):
    """Generate reference trajectories for benchmark."""
    rng = np.random.default_rng(seed)
    m = mujoco.MjModel.from_xml_string(WALKER3D_XML)
    d = mujoco.MjData(m)
    families = {}

    for fam, gen_fn in [
        ("uniform", lambda t, rng: rng.uniform(-tau_max, tau_max, m.nu)),
        ("step", lambda t, rng: tau_max * (2 * (rng.random(m.nu) > 0.5) - 1)
                 if t % 100 == 0 else None),
        ("chirp", lambda t, rng: tau_max * np.sin(
            2 * np.pi * (0.1 + 2.0 * t / traj_len) * t * m.opt.timestep
        ) * np.ones(m.nu)),
    ]:
        trajs = []
        for _ in range(n_per_fam):
            mujoco.mj_resetData(m, d)
            d.qpos[2] = rng.uniform(0.3, 1.0)
            u = rng.standard_normal(4)
            d.qpos[3:7] = u / np.linalg.norm(u)
            d.qpos[7:] = rng.uniform(-0.8, 0.3, m.nq - 7)
            d.qvel[:] = rng.uniform(-2, 2, m.nv)
            mujoco.mj_forward(m, d)

            states = [mj_to_flat(d.qpos.copy(), d.qvel.copy())]
            actions = []
            cur_tau = rng.uniform(-tau_max, tau_max, m.nu)
            for t in range(traj_len):
                new_tau = gen_fn(t, rng)
                if new_tau is not None:
                    cur_tau = new_tau.astype(np.float32)
                d.ctrl[:] = cur_tau
                mujoco.mj_step(m, d)
                states.append(mj_to_flat(d.qpos.copy(), d.qvel.copy()))
                actions.append(cur_tau.copy())
            trajs.append((np.array(states), np.array(actions)))
        families[fam] = trajs
    return families


def benchmark_walker(model_nn, families, device, tau_max, contact_mode):
    """Closed-loop benchmark with different contact representations."""
    m = mujoco.MjModel.from_xml_string(WALKER3D_XML)
    d = mujoco.MjData(m)
    results = {}

    for fam, trajs in families.items():
        mses = []
        for ref_s, ref_a in trajs:
            T = len(ref_a)
            # Set initial state from reference
            flat_to_mj(ref_s[0], m, d)
            errs = []
            for t in range(T):
                s = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if contact_mode == "none":
                    inp = np.concatenate([s, ref_s[t + 1]])
                elif contact_mode == "perlink":
                    flags = get_perlink_flags(d)
                    inp = np.concatenate([s, ref_s[t + 1], flags])

                with torch.no_grad():
                    tau = model_nn(
                        torch.tensor(inp, dtype=torch.float32)
                        .unsqueeze(0).to(device)
                    ).cpu().numpy()[0]
                d.ctrl[:] = np.clip(tau, -tau_max, tau_max)
                mujoco.mj_step(m, d)
                actual = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                errs.append(float(np.mean((actual - ref_s[t + 1]) ** 2)))
            mses.append(np.mean(errs))
        results[fam] = float(np.mean(mses))
    results["AGGREGATE"] = float(
        np.mean([results[k] for k in ["uniform", "step", "chirp"]]))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-traj", type=int, default=2000)
    parser.add_argument("--traj-len", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tau_max = 30.0
    sd = 24  # flat state dim: pos3+rotvec3+joints6+vel12
    nu = 6   # actuated DOF

    out_dir = "outputs/walker3d"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("3D WALKER: Stage 3→4 Bridge Experiment")
    print(f"  state_dim={sd}, actuated_dof={nu}")
    print("=" * 60)

    # Generate data
    t0 = time.time()
    print("\nGenerating training data...")
    S, SN, A, F = gen_data(args.n_traj, args.traj_len, tau_max)
    print(f"  Time: {time.time()-t0:.0f}s")

    # Generate benchmark
    print("Generating benchmark...")
    bench_fams = gen_benchmark(20, args.traj_len, tau_max)

    configs = [
        ("raw_mlp", "none"),
        ("factored", "none"),
        ("factored_contact", "perlink"),
    ]

    all_results = {}
    for label, contact_mode in configs:
        print(f"\n{'='*60}")
        print(f"Config: {label}")
        print(f"{'='*60}")

        if contact_mode == "none":
            X = np.concatenate([S, SN], axis=1)
        else:
            X = np.concatenate([S, SN, F], axis=1)
        Y = A

        if label == "raw_mlp":
            nn = RawMLP(sd, nu, hidden=512, layers=4)
        elif label == "factored":
            nn = FactoredMLP(sd, nu, hidden=512, layers=4)
        elif label == "factored_contact":
            nn = ContactFactoredMLP(sd, nu, n_contact=N_CONTACT_BODIES,
                                    hidden=512, layers=4)

        npar = sum(p.numel() for p in nn.parameters())
        print(f"  Parameters: {npar:,}")

        t0 = time.time()
        nn, val_loss = train_model(nn, X, Y, device, epochs=args.epochs)
        train_time = time.time() - t0

        bench = benchmark_walker(nn, bench_fams, device, tau_max,
                                 contact_mode)
        bench["val_loss"] = val_loss
        bench["params"] = npar
        bench["train_time"] = train_time
        all_results[label] = bench

        print(f"  AGG={bench['AGGREGATE']:.4e}  "
              f"uni={bench['uniform']:.4e}  "
              f"step={bench['step']:.4e}  "
              f"chirp={bench['chirp']:.4e}")
        print(f"  Time: {train_time:.0f}s")

    # Summary
    print(f"\n{'='*60}")
    print("3D WALKER SUMMARY")
    print(f"{'='*60}")
    for name, r in all_results.items():
        print(f"  {name:<20s} AGG={r['AGGREGATE']:.4e}  "
              f"time={r['train_time']:.0f}s")

    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
