#!/usr/bin/env python3
"""3D floating body with limbs — tests 3D rotations and nq≠nv.

System: torso with freejoint (6 DOF unactuated) + 4 arms × 2 joints
each (8 DOF actuated). Zero gravity, no ground contact.
Total: nq=15 (7 base + 8 joints), nv=14 (6 base + 8 joints), nu=8.

Key challenges vs 2D chains/biped:
  1. Quaternion orientation (nq ≠ nv)
  2. 3D rotations (SO(3), non-Euclidean)
  3. Cross-body coupling (arm torques rotate torso)

We convert quaternion state to rotation-vector form for the network,
making the state Euclidean with dim = nv_base + nq_joints + nv = 14+14=28.

Usage:
    CUDA_VISIBLE_DEVICES=4 python -m scripts.run_3d_body
"""
import argparse, json, os, time
import mujoco
import numpy as np
import torch

from scripts.models_structured import FactoredMLP, RawMLP
from scripts.run_structured import train_model


BODY3D_XML = """
<mujoco model="floating_body_3d">
  <option timestep="0.005" gravity="0 0 0" integrator="Euler"/>
  <default>
    <joint damping="0.3" armature="0.01"/>
    <geom condim="1" rgba="0.4 0.6 0.8 1"/>
  </default>
  <worldbody>
    <body name="torso" pos="0 0 0">
      <freejoint name="root"/>
      <geom type="box" size="0.15 0.15 0.1" mass="5"/>
      <!-- Arm +Y: joints around X axis (creates Y-Z rotation) -->
      <body name="arm_py_upper" pos="0 0.2 0">
        <joint name="shoulder_py" type="hinge" axis="1 0 0"
               range="-3.14 3.14" limited="true"/>
        <geom type="capsule" fromto="0 0 0 0 0.2 0" size="0.025" mass="0.5"/>
        <body name="arm_py_lower" pos="0 0.2 0">
          <joint name="elbow_py" type="hinge" axis="0 0 1"
                 range="-2.5 2.5" limited="true"/>
          <geom type="capsule" fromto="0 0 0 0 0.15 0" size="0.02" mass="0.3"/>
        </body>
      </body>
      <!-- Arm -Y: joints around X axis -->
      <body name="arm_ny_upper" pos="0 -0.2 0">
        <joint name="shoulder_ny" type="hinge" axis="1 0 0"
               range="-3.14 3.14" limited="true"/>
        <geom type="capsule" fromto="0 0 0 0 -0.2 0" size="0.025" mass="0.5"/>
        <body name="arm_ny_lower" pos="0 -0.2 0">
          <joint name="elbow_ny" type="hinge" axis="0 0 1"
                 range="-2.5 2.5" limited="true"/>
          <geom type="capsule" fromto="0 0 0 0 -0.15 0" size="0.02" mass="0.3"/>
        </body>
      </body>
      <!-- Arm +X: joints around Y axis (creates X-Z rotation) -->
      <body name="arm_px_upper" pos="0.2 0 0">
        <joint name="shoulder_px" type="hinge" axis="0 1 0"
               range="-3.14 3.14" limited="true"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.025" mass="0.5"/>
        <body name="arm_px_lower" pos="0.2 0 0">
          <joint name="elbow_px" type="hinge" axis="0 0 1"
                 range="-2.5 2.5" limited="true"/>
          <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.02" mass="0.3"/>
        </body>
      </body>
      <!-- Arm -X: joints around Y axis -->
      <body name="arm_nx_upper" pos="-0.2 0 0">
        <joint name="shoulder_nx" type="hinge" axis="0 1 0"
               range="-3.14 3.14" limited="true"/>
        <geom type="capsule" fromto="0 0 0 -0.2 0 0" size="0.025" mass="0.5"/>
        <body name="arm_nx_lower" pos="-0.2 0 0">
          <joint name="elbow_nx" type="hinge" axis="0 0 1"
                 range="-2.5 2.5" limited="true"/>
          <geom type="capsule" fromto="0 0 0 -0.15 0 0" size="0.02" mass="0.3"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="shoulder_py" ctrlrange="-10 10" ctrllimited="true"/>
    <motor joint="elbow_py" ctrlrange="-10 10" ctrllimited="true"/>
    <motor joint="shoulder_ny" ctrlrange="-10 10" ctrllimited="true"/>
    <motor joint="elbow_ny" ctrlrange="-10 10" ctrllimited="true"/>
    <motor joint="shoulder_px" ctrlrange="-10 10" ctrllimited="true"/>
    <motor joint="elbow_px" ctrlrange="-10 10" ctrllimited="true"/>
    <motor joint="shoulder_nx" ctrlrange="-10 10" ctrllimited="true"/>
    <motor joint="elbow_nx" ctrlrange="-10 10" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


def quat_to_rotvec(q):
    """Convert MuJoCo quaternion [w,x,y,z] to 3D rotation vector."""
    w, x, y, z = q
    sin_half = np.sqrt(x*x + y*y + z*z)
    if sin_half < 1e-8:
        return np.array([2*x, 2*y, 2*z])
    angle = 2.0 * np.arctan2(sin_half, abs(w))
    axis = np.array([x, y, z]) / sin_half
    if w < 0:
        angle = 2*np.pi - angle
        axis = -axis
    return axis * angle


def mj_state_to_flat(qpos, qvel):
    """Convert MuJoCo state (qpos with quaternion) to flat Euclidean state.

    MuJoCo: qpos = [pos(3), quat(4), joints(8)] = 15D
             qvel = [lin(3), ang(3), joints(8)] = 14D
    Output: [pos(3), rotvec(3), joints(8), lin(3), ang(3), joints(8)] = 28D
    """
    pos = qpos[:3]
    quat = qpos[3:7]
    joints = qpos[7:]
    rotvec = quat_to_rotvec(quat)
    return np.concatenate([pos, rotvec, joints, qvel]).astype(np.float32)


def gen_data(n_traj, traj_len, tau_max, seed=42):
    """Random rollout data for the 3D floating body."""
    rng = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_string(BODY3D_XML)
    data = mujoco.MjData(model)
    nq, nv, nu = model.nq, model.nv, model.nu
    print(f"  3D body: nq={nq}, nv={nv}, nu={nu}")

    all_s, all_sn, all_a = [], [], []
    for _ in range(n_traj):
        mujoco.mj_resetData(model, data)
        # Random initial: small displacement + random joint angles
        data.qpos[:3] = rng.uniform(-0.5, 0.5, size=3)
        # Random quaternion
        u = rng.normal(size=4)
        data.qpos[3:7] = u / np.linalg.norm(u)
        data.qpos[7:] = rng.uniform(-2.0, 2.0, size=nu)
        data.qvel[:] = rng.uniform(-3, 3, size=nv)
        mujoco.mj_forward(model, data)

        for _ in range(traj_len):
            s = mj_state_to_flat(data.qpos.copy(), data.qvel.copy())
            tau = rng.uniform(-tau_max, tau_max, size=nu)
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            sn = mj_state_to_flat(data.qpos.copy(), data.qvel.copy())
            all_s.append(s)
            all_sn.append(sn)
            all_a.append(tau)

    S = np.array(all_s, dtype=np.float32)
    SN = np.array(all_sn, dtype=np.float32)
    A = np.array(all_a, dtype=np.float32)
    sd = S.shape[1]
    print(f"  {len(S):,} pairs, flat_state_dim={sd}, action_dim={nu}")
    return S, SN, A, sd, nu


def gen_benchmark(n_per, traj_len, tau_max, seed=9999):
    """Benchmark references for the 3D body."""
    rng = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_string(BODY3D_XML)
    data = mujoco.MjData(model)
    nq, nv, nu = model.nq, model.nv, model.nu
    dt = model.opt.timestep
    families = {}

    for fam in ["uniform", "step", "chirp"]:
        trajs = []
        for _ in range(n_per):
            mujoco.mj_resetData(model, data)
            data.qpos[:3] = rng.uniform(-0.3, 0.3, size=3)
            u = rng.normal(size=4)
            data.qpos[3:7] = u / np.linalg.norm(u)
            data.qpos[7:] = rng.uniform(-1.5, 1.5, size=nu)
            data.qvel[:] = rng.uniform(-2, 2, size=nv)
            mujoco.mj_forward(model, data)

            T = traj_len
            actions = np.zeros((T, nu))
            if fam == "uniform":
                actions = rng.uniform(-tau_max, tau_max, size=(T, nu))
            elif fam == "step":
                for j in range(nu):
                    v1 = rng.uniform(-tau_max, tau_max)
                    sw = rng.integers(T // 4, 3 * T // 4)
                    actions[:sw, j] = v1
                    actions[sw:, j] = rng.uniform(-tau_max, tau_max)
            elif fam == "chirp":
                t_arr = np.arange(T) * dt
                for j in range(nu):
                    f0 = rng.uniform(0.1, 1)
                    f1 = rng.uniform(2, 10)
                    amp = rng.uniform(0.5, 1.0) * tau_max
                    ph = t_arr * (f0 + (f1-f0)*t_arr/(2*t_arr[-1]))
                    actions[:, j] = amp * np.sin(2*np.pi*ph)

            # Rollout and record flat states
            states = []
            states.append(mj_state_to_flat(
                data.qpos.copy(), data.qvel.copy()))
            for t in range(T):
                data.ctrl[:] = np.clip(actions[t], -tau_max, tau_max)
                mujoco.mj_step(model, data)
                states.append(mj_state_to_flat(
                    data.qpos.copy(), data.qvel.copy()))
            states = np.array(states)
            trajs.append((states, actions))
        families[fam] = trajs
    return families


def benchmark_3d(model_nn, families, device, tau_max):
    """Closed-loop benchmark on the 3D body."""
    mj = mujoco.MjModel.from_xml_string(BODY3D_XML)
    d = mujoco.MjData(mj)
    results = {}

    for fam, trajs in families.items():
        mses = []
        for ref_s, ref_a in trajs:
            T = len(ref_a)
            mujoco.mj_resetData(mj, d)
            # Recover MuJoCo state from flat ref state
            # ref_s[0] = [pos(3), rotvec(3), joints(8), vel(14)]
            # We need to set qpos and qvel from this
            _set_mj_from_flat(mj, d, ref_s[0])
            mujoco.mj_forward(mj, d)

            errs = []
            for t in range(T):
                s = mj_state_to_flat(d.qpos.copy(), d.qvel.copy())
                inp = np.concatenate([s, ref_s[t + 1]]).astype(np.float32)
                with torch.no_grad():
                    tau = model_nn(
                        torch.tensor(inp).unsqueeze(0).to(device)
                    ).cpu().numpy()[0]
                d.ctrl[:] = np.clip(tau, -tau_max, tau_max)
                mujoco.mj_step(mj, d)
                actual = mj_state_to_flat(d.qpos.copy(), d.qvel.copy())
                errs.append(float(np.mean((actual - ref_s[t+1])**2)))
            mses.append(np.mean(errs))
        results[fam] = float(np.mean(mses))
    results["AGGREGATE"] = float(
        np.mean([results[k] for k in ["uniform", "step", "chirp"]]))
    return results


def rotvec_to_quat(rv):
    """Convert rotation vector to MuJoCo quaternion [w,x,y,z]."""
    angle = np.linalg.norm(rv)
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rv / angle
    w = np.cos(angle / 2)
    xyz = axis * np.sin(angle / 2)
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def _set_mj_from_flat(model, data, flat_state):
    """Set MuJoCo qpos/qvel from flat Euclidean state."""
    nv = model.nv
    nu = model.nu
    pos = flat_state[:3]
    rotvec = flat_state[3:6]
    joints = flat_state[6:6+nu]
    vel = flat_state[6+nu:6+nu+nv]
    data.qpos[:3] = pos
    data.qpos[3:7] = rotvec_to_quat(rotvec)
    data.qpos[7:] = joints
    data.qvel[:] = vel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-traj", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tau_max = 10.0
    out_dir = "outputs/body3d"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("3D FLOATING BODY: Factored vs Raw MLP")
    print("  nq=15, nv=14, nu=8 (freejoint + 4 arms × 2 joints)")
    print("  Zero gravity, no contact — isolates 3D rotation challenge")
    print("=" * 60)

    # Generate data
    print("\nGenerating training data...")
    t0 = time.time()
    S, SN, A, sd, nu = gen_data(args.n_traj, 500, tau_max)
    print(f"  Data generated in {time.time()-t0:.1f}s")

    # Generate benchmark
    print("\nGenerating benchmark...")
    families = gen_benchmark(50, 500, tau_max)

    configs = [
        ("raw_mlp", "raw"),
        ("factored", "factored"),
    ]

    all_results = {}
    for label, arch in configs:
        print(f"\n{'='*60}")
        print(f"Config: {label}")
        print(f"{'='*60}")

        X = np.concatenate([S, SN], axis=1)
        Y = A

        if arch == "factored":
            nn_model = FactoredMLP(sd, nu, hidden=512, layers=4)
        else:
            nn_model = RawMLP(sd, nu, hidden=512, layers=4)

        npar = sum(p.numel() for p in nn_model.parameters())
        print(f"  Parameters: {npar:,}")

        nn_model, val_loss = train_model(
            nn_model, X, Y, device, epochs=args.epochs, bs=4096)

        bench = benchmark_3d(nn_model, families, device, tau_max)
        bench["val_loss"] = val_loss
        bench["params"] = npar
        all_results[label] = bench

        print(f"  AGG={bench['AGGREGATE']:.4e}  "
              f"uni={bench['uniform']:.4e}  "
              f"step={bench['step']:.4e}  "
              f"chirp={bench['chirp']:.4e}")

    # Summary
    print(f"\n{'='*60}")
    print("3D BODY SUMMARY")
    print(f"{'='*60}")
    for name, r in all_results.items():
        a = r['AGGREGATE']
        print(f"  {name:<20s} AGG={a:.4e}")
    if len(all_results) >= 2:
        agg_r = all_results["raw_mlp"]["AGGREGATE"]
        agg_f = all_results["factored"]["AGGREGATE"]
        print(f"  Factored advantage: {agg_r/agg_f:.1f}×")

    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
