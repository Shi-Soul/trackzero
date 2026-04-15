#!/usr/bin/env python3
"""Mini-humanoid: 12 actuated DOF + freejoint with gravity + contacts.

The most informative Stage 3→4 stepping stone. Tests:
  - 12 actuated DOF (2× the walker) with 3D rotations
  - Humanoid-like tree topology (torso → arms + legs)
  - Ground contact (feet)
  - Both global and hierarchical factored architectures

Morphology: torso + 2 legs (hip-pitch, hip-yaw, knee) +
            2 arms (shoulder-pitch, shoulder-yaw, elbow)
nq = 19 (7 freejoint + 12 hinge), nv = 18 (6 + 12), nu = 12

Usage:
    CUDA_VISIBLE_DEVICES=3 python -m scripts.run_mini_humanoid
"""
import argparse, json, os, time
import mujoco
import numpy as np
import torch
import torch.nn as nn

from scripts.models_structured import RawMLP, FactoredMLP, _build_mlp
from scripts.run_structured import train_model
from scripts.run_walker3d import quat_to_rotvec, rotvec_to_quat

MINI_HUMANOID_XML = """
<mujoco model="mini_humanoid">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1"
          rgba="0.8 0.9 0.8 1" friction="1 0.5 0.5"
          conaffinity="1" contype="1"/>
    <body name="torso" pos="0 0 0.9">
      <freejoint name="root"/>
      <geom name="torso_g" type="capsule"
            fromto="0 -0.12 0 0 0.12 0" size="0.07" mass="8"
            conaffinity="1" contype="1"/>
      <!-- Head (cosmetic, no joint) -->
      <geom name="head_g" type="sphere" pos="0 0 0.15" size="0.08"
            mass="1" conaffinity="1" contype="1"/>
      <!-- Left arm -->
      <body name="l_upper_arm" pos="0 0.15 0.05">
        <joint name="l_shoulder_y" type="hinge" axis="0 1 0"
               range="-3.14 1.0" damping="0.3"/>
        <joint name="l_shoulder_x" type="hinge" axis="1 0 0"
               range="-1.5 1.5" damping="0.3"/>
        <geom name="l_uarm_g" type="capsule" fromto="0 0 0 0 0 -0.25"
              size="0.03" mass="1" conaffinity="1" contype="1"/>
        <body name="l_forearm" pos="0 0 -0.25">
          <joint name="l_elbow" type="hinge" axis="0 1 0"
                 range="-2.5 0" damping="0.3"/>
          <geom name="l_farm_g" type="capsule" fromto="0 0 0 0 0 -0.2"
                size="0.025" mass="0.5" conaffinity="1" contype="1"/>
        </body>
      </body>
      <!-- Right arm -->
      <body name="r_upper_arm" pos="0 -0.15 0.05">
        <joint name="r_shoulder_y" type="hinge" axis="0 1 0"
               range="-3.14 1.0" damping="0.3"/>
        <joint name="r_shoulder_x" type="hinge" axis="1 0 0"
               range="-1.5 1.5" damping="0.3"/>
        <geom name="r_uarm_g" type="capsule" fromto="0 0 0 0 0 -0.25"
              size="0.03" mass="1" conaffinity="1" contype="1"/>
        <body name="r_forearm" pos="0 0 -0.25">
          <joint name="r_elbow" type="hinge" axis="0 1 0"
                 range="-2.5 0" damping="0.3"/>
          <geom name="r_farm_g" type="capsule" fromto="0 0 0 0 0 -0.2"
                size="0.025" mass="0.5" conaffinity="1" contype="1"/>
        </body>
      </body>
      <!-- Left leg -->
      <body name="l_thigh" pos="0 0.1 -0.1">
        <joint name="l_hip_y" type="hinge" axis="0 1 0"
               range="-1.57 0.5" damping="0.5"/>
        <joint name="l_hip_x" type="hinge" axis="1 0 0"
               range="-0.5 0.5" damping="0.5"/>
        <geom name="l_thigh_g" type="capsule" fromto="0 0 0 0 0 -0.3"
              size="0.04" mass="2.5" conaffinity="1" contype="1"/>
        <body name="l_shin" pos="0 0 -0.3">
          <joint name="l_knee" type="hinge" axis="0 1 0"
                 range="-2.5 0" damping="0.5"/>
          <geom name="l_shin_g" type="capsule"
                fromto="0 0 0 0 0 -0.28" size="0.03" mass="1.5"
                conaffinity="1" contype="1"/>
          <body name="l_foot" pos="0 0 -0.28">
            <geom name="l_foot_g" type="box" size="0.08 0.04 0.02"
                  pos="0.03 0 0" mass="0.5"
                  conaffinity="1" contype="1"/>
          </body>
        </body>
      </body>
      <!-- Right leg -->
      <body name="r_thigh" pos="0 -0.1 -0.1">
        <joint name="r_hip_y" type="hinge" axis="0 1 0"
               range="-1.57 0.5" damping="0.5"/>
        <joint name="r_hip_x" type="hinge" axis="1 0 0"
               range="-0.5 0.5" damping="0.5"/>
        <geom name="r_thigh_g" type="capsule" fromto="0 0 0 0 0 -0.3"
              size="0.04" mass="2.5" conaffinity="1" contype="1"/>
        <body name="r_shin" pos="0 0 -0.3">
          <joint name="r_knee" type="hinge" axis="0 1 0"
                 range="-2.5 0" damping="0.5"/>
          <geom name="r_shin_g" type="capsule"
                fromto="0 0 0 0 0 -0.28" size="0.03" mass="1.5"
                conaffinity="1" contype="1"/>
          <body name="r_foot" pos="0 0 -0.28">
            <geom name="r_foot_g" type="box" size="0.08 0.04 0.02"
                  pos="0.03 0 0" mass="0.5"
                  conaffinity="1" contype="1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="l_shoulder_y" ctrlrange="-20 20"/>
    <motor joint="l_shoulder_x" ctrlrange="-20 20"/>
    <motor joint="l_elbow" ctrlrange="-20 20"/>
    <motor joint="r_shoulder_y" ctrlrange="-20 20"/>
    <motor joint="r_shoulder_x" ctrlrange="-20 20"/>
    <motor joint="r_elbow" ctrlrange="-20 20"/>
    <motor joint="l_hip_y" ctrlrange="-40 40"/>
    <motor joint="l_hip_x" ctrlrange="-40 40"/>
    <motor joint="l_knee" ctrlrange="-40 40"/>
    <motor joint="r_hip_y" ctrlrange="-40 40"/>
    <motor joint="r_hip_x" ctrlrange="-40 40"/>
    <motor joint="r_knee" ctrlrange="-40 40"/>
  </actuator>
</mujoco>
"""


# Geom mapping: 0=floor, 1=torso, 2=head, 3=l_uarm, 4=l_farm,
#   5=r_uarm, 6=r_farm, 7=l_thigh, 8=l_shin, 9=l_foot,
#   10=r_thigh, 11=r_shin, 12=r_foot
N_BODY_GEOMS = 12  # all body geoms (excluding floor)
GEOM_OFFSET = 1    # floor is geom 0


def mj_to_flat(qpos, qvel):
    """qpos=[pos3, quat4, joints12]=19, qvel=[vel6, jointvel12]=18
    flat=[pos3, rotvec3, joints12, vel18]=36
    """
    pos = qpos[:3]
    rv = quat_to_rotvec(qpos[3:7])
    joints = qpos[7:]
    return np.concatenate([pos, rv, joints, qvel]).astype(np.float32)


def flat_to_mj(flat, model, data):
    """Inverse of mj_to_flat."""
    data.qpos[:3] = flat[:3]
    data.qpos[3:7] = rotvec_to_quat(flat[3:6])
    data.qpos[7:] = flat[6:18]
    data.qvel[:] = flat[18:36]
    mujoco.mj_forward(model, data)


def get_contact_flags(data):
    """Per-body-geom contact flags."""
    flags = np.zeros(N_BODY_GEOMS, dtype=np.float32)
    for ci in range(data.ncon):
        for g in [data.contact[ci].geom1, data.contact[ci].geom2]:
            idx = g - GEOM_OFFSET
            if 0 <= idx < N_BODY_GEOMS:
                flags[idx] = 1.0
    return flags


def gen_data(n_traj, traj_len, seed=42):
    """Random rollout data."""
    rng = np.random.default_rng(seed)
    m = mujoco.MjModel.from_xml_string(MINI_HUMANOID_XML)
    d = mujoco.MjData(m)
    tau_max_arms = 20.0
    tau_max_legs = 40.0
    tau_max = np.array([20]*6 + [40]*6, dtype=np.float32)

    all_s, all_sn, all_a, all_f = [], [], [], []

    for _ in range(n_traj):
        mujoco.mj_resetData(m, d)
        d.qpos[:2] = rng.uniform(-0.3, 0.3, 2)
        d.qpos[2] = rng.uniform(0.2, 1.3)
        u = rng.standard_normal(4)
        d.qpos[3:7] = u / np.linalg.norm(u)
        d.qpos[7:] = rng.uniform(-1.0, 0.5, m.nq - 7)
        d.qvel[:] = rng.uniform(-3, 3, m.nv)
        mujoco.mj_forward(m, d)

        for _ in range(traj_len):
            s = mj_to_flat(d.qpos.copy(), d.qvel.copy())
            flags = get_contact_flags(d)
            tau = rng.uniform(-1, 1, m.nu).astype(np.float32) * tau_max

            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            sn = mj_to_flat(d.qpos.copy(), d.qvel.copy())

            all_s.append(s)
            all_sn.append(sn)
            all_a.append(tau)
            all_f.append(flags)

    S = np.array(all_s, dtype=np.float32)
    SN = np.array(all_sn, dtype=np.float32)
    A = np.array(all_a, dtype=np.float32)
    F = np.array(all_f, dtype=np.float32)

    cfrac = (F.max(axis=1) > 0).mean()
    per_frac = F.mean(axis=0)
    print(f"  {len(S):,} pairs, any_contact={cfrac:.1%}")
    print(f"  Per-body: {np.array2string(per_frac, precision=2)}")
    return S, SN, A, F


def gen_benchmark(n_per_fam, traj_len, seed=99):
    """Generate reference trajectories."""
    rng = np.random.default_rng(seed)
    m = mujoco.MjModel.from_xml_string(MINI_HUMANOID_XML)
    d = mujoco.MjData(m)
    tau_max = np.array([20]*6 + [40]*6, dtype=np.float32)
    families = {}

    for fam, gen_fn in [
        ("uniform", lambda t, rng: rng.uniform(-1, 1, m.nu) * tau_max),
        ("step", lambda t, rng: tau_max * (2*(rng.random(m.nu)>0.5)-1)
                 if t % 100 == 0 else None),
        ("chirp", lambda t, rng: tau_max * np.sin(
            2*np.pi*(0.1 + 2.0*t/traj_len)*t*m.opt.timestep
        ) * np.ones(m.nu)),
    ]:
        trajs = []
        for _ in range(n_per_fam):
            mujoco.mj_resetData(m, d)
            d.qpos[2] = rng.uniform(0.3, 1.1)
            u = rng.standard_normal(4)
            d.qpos[3:7] = u / np.linalg.norm(u)
            d.qpos[7:] = rng.uniform(-0.8, 0.3, m.nq-7)
            d.qvel[:] = rng.uniform(-2, 2, m.nv)
            mujoco.mj_forward(m, d)

            states = [mj_to_flat(d.qpos.copy(), d.qvel.copy())]
            actions = []
            cur_tau = rng.uniform(-1, 1, m.nu).astype(np.float32) * tau_max
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


class LimbFactoredMLP(nn.Module):
    """Hierarchical factored with 4 limbs."""
    def __init__(self, state_dim, limb_dofs, n_contact=0,
                 hidden=512, layers=4):
        super().__init__()
        self.sd = state_dim
        self.limb_dofs = limb_dofs
        self.nu = sum(limb_dofs)
        self.n_contact = n_contact
        cond_dim = state_dim + n_contact

        self.limb_gain_nets = nn.ModuleList()
        self.limb_bias_nets = nn.ModuleList()
        for nd in limb_dofs:
            self.limb_gain_nets.append(
                _build_mlp(cond_dim, nd * state_dim, hidden, layers))
            self.limb_bias_nets.append(
                _build_mlp(cond_dim, nd, hidden, layers))

        full_in = 2 * state_dim + n_contact
        self.register_buffer("mu", torch.zeros(full_in))
        self.register_buffer("sigma", torch.ones(full_in))

    def set_norm(self, m, s):
        self.mu.copy_(torch.from_numpy(m).float())
        self.sigma.copy_(torch.from_numpy(s).float())

    def forward(self, x):
        sd = self.sd
        x_norm = (x - self.mu) / self.sigma
        state_raw = x[:, :sd]
        target_raw = x[:, sd:2*sd]
        state_n = x_norm[:, :sd]
        if self.n_contact > 0:
            cond = torch.cat([state_n, x_norm[:, 2*sd:]], dim=1)
        else:
            cond = state_n
        error = target_raw - state_raw
        taus = []
        for i, nd in enumerate(self.limb_dofs):
            A = self.limb_gain_nets[i](cond).reshape(-1, nd, sd)
            b = self.limb_bias_nets[i](cond)
            tau_limb = torch.bmm(A, error.unsqueeze(-1)).squeeze(-1) + b
            taus.append(tau_limb)
        return torch.cat(taus, dim=1)


def benchmark(model_nn, families, device, contact_mode):
    """Closed-loop benchmark."""
    m = mujoco.MjModel.from_xml_string(MINI_HUMANOID_XML)
    d = mujoco.MjData(m)
    tau_max = np.array([20]*6 + [40]*6, dtype=np.float32)
    results = {}

    for fam, trajs in families.items():
        mses = []
        for ref_s, ref_a in trajs:
            T = len(ref_a)
            flat_to_mj(ref_s[0], m, d)
            errs = []
            for t in range(T):
                s = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if contact_mode == "none":
                    inp = np.concatenate([s, ref_s[t+1]])
                else:
                    flags = get_contact_flags(d)
                    inp = np.concatenate([s, ref_s[t+1], flags])
                with torch.no_grad():
                    tau = model_nn(
                        torch.tensor(inp, dtype=torch.float32)
                        .unsqueeze(0).to(device)
                    ).cpu().numpy()[0]
                d.ctrl[:] = np.clip(tau, -tau_max, tau_max)
                mujoco.mj_step(m, d)
                actual = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                errs.append(float(np.mean((actual - ref_s[t+1])**2)))
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
    sd = 36  # pos3+rotvec3+joints12+vel18
    nu = 12

    # 4 limbs: left_arm(3), right_arm(3), left_leg(3), right_leg(3)
    limb_dofs = [3, 3, 3, 3]

    out_dir = "outputs/mini_humanoid"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("MINI-HUMANOID: 12 Actuated DOF")
    print(f"  state_dim={sd}, nu={nu}, limbs={limb_dofs}")
    print("=" * 60)

    print("\nGenerating data...")
    t0 = time.time()
    S, SN, A, F = gen_data(args.n_traj, args.traj_len)
    print(f"  Time: {time.time()-t0:.0f}s")

    print("Generating benchmark...")
    bench_fams = gen_benchmark(20, args.traj_len)

    configs = [
        ("raw_mlp", "none", RawMLP(sd, nu, hidden=1024, layers=4)),
        ("limb_factored", "none",
         LimbFactoredMLP(sd, limb_dofs, hidden=512, layers=4)),
        ("limb_factored_contact", "perlink",
         LimbFactoredMLP(sd, limb_dofs, n_contact=N_BODY_GEOMS,
                         hidden=512, layers=4)),
    ]

    all_results = {}
    for label, contact_mode, nn_model in configs:
        print(f"\n{'='*60}")
        print(f"Config: {label}")
        print(f"{'='*60}")

        if contact_mode == "none":
            X = np.concatenate([S, SN], axis=1)
        else:
            X = np.concatenate([S, SN, F], axis=1)
        Y = A

        npar = sum(p.numel() for p in nn_model.parameters())
        print(f"  Parameters: {npar:,}")

        t0 = time.time()
        nn_model, val_loss = train_model(nn_model, X, Y, device,
                                         epochs=args.epochs)
        train_time = time.time() - t0

        bench_r = benchmark(nn_model, bench_fams, device, contact_mode)
        bench_r["val_loss"] = val_loss
        bench_r["params"] = npar
        bench_r["train_time"] = train_time
        all_results[label] = bench_r

        print(f"  AGG={bench_r['AGGREGATE']:.4e}  time={train_time:.0f}s")

    print(f"\n{'='*60}")
    print("MINI-HUMANOID SUMMARY")
    print(f"{'='*60}")
    for name, r in all_results.items():
        print(f"  {name:<25s} AGG={r['AGGREGATE']:.4e}  "
              f"params={r['params']:,}")

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_dir}/results.json")


if __name__ == "__main__":
    main()
