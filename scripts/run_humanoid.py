#!/usr/bin/env python3
"""Stage 4: Full MuJoCo humanoid (21 actuated DOF).

Tests whether TRACK-ZERO scales to humanoid control without motion priors.
nq=28 (7 freejoint + 21 hinge), nv=27, nu=21, flat_state_dim=54

Architectures tested:
  1. raw_mlp + per-link contact (baseline)
  2. limb_factored + per-link contact (hypothesis: best for tree topology)

Limb structure: torso(3), right_leg(6), left_leg(6), right_arm(3), left_arm(3)

Usage:
    CUDA_VISIBLE_DEVICES=4 python -m scripts.run_humanoid
"""
import argparse, json, os, time
import mujoco
import numpy as np
import torch
import torch.nn as nn

from scripts.models_structured import RawMLP, _build_mlp
from scripts.run_structured import train_model
from scripts.run_walker3d import quat_to_rotvec, rotvec_to_quat

# ── Humanoid XML ──────────────────────────────────────────────────────
HUMANOID_XML = """
<mujoco model="humanoid">
  <compiler angle="degree" inertiafromgeom="true"/>
  <option timestep="0.002" integrator="implicit"/>
  <default>
    <joint limited="true" damping="2" armature="0.01"/>
    <geom contype="1" conaffinity="1" condim="3"
          friction="1 .1 .1" density="1000" margin="0.001"/>
  </default>
  <worldbody>
    <light diffuse=".8 .8 .8" pos="0 0 5"/>
    <geom type="plane" size="10 10 0.1" rgba=".8 .9 .8 1"/>
    <body name="torso" pos="0 0 1.4">
      <freejoint name="root"/>
      <geom name="torso1" type="capsule"
            fromto="0 -.07 0 0 .07 0" size="0.07"/>
      <geom name="head" type="sphere" pos="0 0 .19" size=".09"/>
      <geom name="uwaist" type="capsule"
            fromto="-.01 -.06 -.12 -.01 .06 -.12" size="0.06"/>
      <body name="lwaist" pos="-.01 0 -0.260">
        <joint name="abdomen_z" type="hinge" pos="0 0 0.065"
               axis="0 0 1" range="-45 45" stiffness="6" damping="0.5"/>
        <joint name="abdomen_y" type="hinge" pos="0 0 0.065"
               axis="0 1 0" range="-75 30" stiffness="6" damping="0.5"/>
        <geom name="lwaist" type="capsule"
              fromto="0 -.06 0 0 .06 0" size="0.06"/>
        <body name="pelvis" pos="0 0 -0.165">
          <joint name="abdomen_x" type="hinge" pos="0 0 0.1"
                 axis="1 0 0" range="-35 35" stiffness="6" damping="0.5"/>
          <geom name="butt" type="capsule"
                fromto="-.02 -.07 0 -.02 .07 0" size="0.09"/>
          <body name="right_thigh" pos="0 -0.1 -0.04">
            <joint name="right_hip_x" type="hinge" axis="1 0 0"
                   range="-25 5" stiffness="5" damping="0.5"/>
            <joint name="right_hip_z" type="hinge" axis="0 0 1"
                   range="-60 35" stiffness="5" damping="0.5"/>
            <joint name="right_hip_y" type="hinge" axis="0 1 0"
                   range="-110 20" stiffness="5" damping="0.5"/>
            <geom name="right_thigh1" type="capsule"
                  fromto="0 0 0 0 0.01 -.34" size="0.06"/>
            <body name="right_shin" pos="0 0.01 -0.403">
              <joint name="right_knee" type="hinge" axis="0 -1 0"
                     range="-160 -2" stiffness="3" damping="0.3"/>
              <geom name="right_shin1" type="capsule"
                    fromto="0 0 0 0 0 -.3" size="0.049"/>
              <body name="right_foot" pos="0 0 -.39">
                <joint name="right_ankle_y" type="hinge" axis="0 1 0"
                       range="-50 50" stiffness="2" damping="0.1"/>
                <joint name="right_ankle_x" type="hinge"
                       axis="1 0 0.5" range="-50 50"
                       stiffness="2" damping="0.1"/>
                <geom name="right_foot" type="box"
                      size="0.075 0.045 0.025" pos="0.045 0 0.025"/>
              </body>
            </body>
          </body>
          <body name="left_thigh" pos="0 0.1 -0.04">
            <joint name="left_hip_x" type="hinge" axis="1 0 0"
                   range="-25 5" stiffness="5" damping="0.5"/>
            <joint name="left_hip_z" type="hinge" axis="0 0 1"
                   range="-60 35" stiffness="5" damping="0.5"/>
            <joint name="left_hip_y" type="hinge" axis="0 1 0"
                   range="-120 20" stiffness="5" damping="0.5"/>
            <geom name="left_thigh1" type="capsule"
                  fromto="0 0 0 0 0.01 -.34" size="0.06"/>
            <body name="left_shin" pos="0 0.01 -0.403">
              <joint name="left_knee" type="hinge" axis="0 -1 0"
                     range="-160 -2" stiffness="3" damping="0.3"/>
              <geom name="left_shin1" type="capsule"
                    fromto="0 0 0 0 0 -.3" size="0.049"/>
              <body name="left_foot" pos="0 0 -.39">
                <joint name="left_ankle_y" type="hinge" axis="0 1 0"
                       range="-50 50" stiffness="2" damping="0.1"/>
                <joint name="left_ankle_x" type="hinge"
                       axis="1 0 0.5" range="-50 50"
                       stiffness="2" damping="0.1"/>
                <geom name="left_foot" type="box"
                      size="0.075 0.045 0.025" pos="0.045 0 0.025"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="right_upper_arm" pos="0 -0.17 0.06">
        <joint name="right_shoulder1" type="hinge" axis="2 1 1"
               range="-85 60" stiffness="2" damping="0.2"/>
        <joint name="right_shoulder2" type="hinge" axis="0 -1 1"
               range="-85 60" stiffness="2" damping="0.2"/>
        <geom name="right_uarm1" type="capsule"
              fromto="0 0 0 .16 -.16 -.16" size="0.04"/>
        <body name="right_lower_arm" pos=".18 -.18 -.18">
          <joint name="right_elbow" type="hinge" axis="0 -1 1"
                 range="-90 50" stiffness="1" damping="0.1"/>
          <geom name="right_larm" type="capsule"
                fromto="0.01 0.01 0.01 .17 .17 .17" size="0.031"/>
        </body>
      </body>
      <body name="left_upper_arm" pos="0 0.17 0.06">
        <joint name="left_shoulder1" type="hinge" axis="2 -1 1"
               range="-60 85" stiffness="2" damping="0.2"/>
        <joint name="left_shoulder2" type="hinge" axis="0 1 1"
               range="-60 85" stiffness="2" damping="0.2"/>
        <geom name="left_uarm1" type="capsule"
              fromto="0 0 0 .16 .16 -.16" size="0.04"/>
        <body name="left_lower_arm" pos=".18 .18 -.18">
          <joint name="left_elbow" type="hinge" axis="0 -1 -1"
                 range="-90 50" stiffness="1" damping="0.1"/>
          <geom name="left_larm" type="capsule"
                fromto="0.01 -0.01 0.01 .17 -.17 .17" size="0.031"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="abdomen_z" gear="40"/>
    <motor joint="abdomen_y" gear="40"/>
    <motor joint="abdomen_x" gear="40"/>
    <motor joint="right_hip_x" gear="40"/>
    <motor joint="right_hip_z" gear="40"/>
    <motor joint="right_hip_y" gear="120"/>
    <motor joint="right_knee" gear="80"/>
    <motor joint="right_ankle_y" gear="20"/>
    <motor joint="right_ankle_x" gear="20"/>
    <motor joint="left_hip_x" gear="40"/>
    <motor joint="left_hip_z" gear="40"/>
    <motor joint="left_hip_y" gear="120"/>
    <motor joint="left_knee" gear="80"/>
    <motor joint="left_ankle_y" gear="20"/>
    <motor joint="left_ankle_x" gear="20"/>
    <motor joint="right_shoulder1" gear="25"/>
    <motor joint="right_shoulder2" gear="25"/>
    <motor joint="right_elbow" gear="25"/>
    <motor joint="left_shoulder1" gear="25"/>
    <motor joint="left_shoulder2" gear="25"/>
    <motor joint="left_elbow" gear="25"/>
  </actuator>
</mujoco>
"""

# ── Constants ─────────────────────────────────────────────────────────
N_BODY_GEOMS = 15   # all body geoms (excl. floor)
GEOM_OFFSET = 1     # floor is geom 0
NU = 21
SD = 54  # pos3 + rotvec3 + joints21 + vel27
LIMB_DOFS = [3, 6, 6, 3, 3]  # torso, rleg, lleg, rarm, larm

# Torque limits per actuator (from gear ratios, ctrl range ±1)
TAU_MAX = np.array([
    40, 40, 40,               # abdomen
    40, 40, 120, 80, 20, 20,  # right leg
    40, 40, 120, 80, 20, 20,  # left leg
    25, 25, 25,               # right arm
    25, 25, 25,               # left arm
], dtype=np.float32)


# ── State conversion ──────────────────────────────────────────────────
def mj_to_flat(qpos, qvel):
    """qpos=[pos3,quat4,joints21]=28, qvel=[vel6,jointvel21]=27 → 54D."""
    pos = qpos[:3]
    rv = quat_to_rotvec(qpos[3:7])
    joints = qpos[7:]
    return np.concatenate([pos, rv, joints, qvel]).astype(np.float32)


def flat_to_mj(flat, model, data):
    data.qpos[:3] = flat[:3]
    data.qpos[3:7] = rotvec_to_quat(flat[3:6])
    data.qpos[7:] = flat[6:27]
    data.qvel[:] = flat[27:54]
    mujoco.mj_forward(model, data)


def get_contact_flags(data):
    flags = np.zeros(N_BODY_GEOMS, dtype=np.float32)
    for ci in range(data.ncon):
        for g in [data.contact[ci].geom1, data.contact[ci].geom2]:
            idx = g - GEOM_OFFSET
            if 0 <= idx < N_BODY_GEOMS:
                flags[idx] = 1.0
    return flags


# ── Data generation ───────────────────────────────────────────────────
def gen_data(n_traj, traj_len, seed=42):
    rng = np.random.default_rng(seed)
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    all_s, all_sn, all_a, all_f = [], [], [], []
    nan_resets = 0

    for ti in range(n_traj):
        mujoco.mj_resetData(m, d)
        d.qpos[:2] = rng.uniform(-0.1, 0.1, 2)
        d.qpos[2] = rng.uniform(0.5, 1.4)
        u = rng.standard_normal(4)
        d.qpos[3:7] = u / np.linalg.norm(u)
        d.qpos[7:] = rng.uniform(-0.5, 0.2, m.nq - 7)
        d.qvel[:] = rng.uniform(-1, 1, m.nv)
        mujoco.mj_forward(m, d)

        for _ in range(traj_len):
            # NaN check
            if np.any(np.isnan(d.qpos)) or np.any(np.isnan(d.qvel)):
                nan_resets += 1
                mujoco.mj_resetData(m, d)
                d.qpos[2] = rng.uniform(0.5, 1.4)
                u = rng.standard_normal(4)
                d.qpos[3:7] = u / np.linalg.norm(u)
                mujoco.mj_forward(m, d)
                continue

            s = mj_to_flat(d.qpos.copy(), d.qvel.copy())
            flags = get_contact_flags(d)
            # Use 15% of max torque for stability
            tau = (rng.uniform(-0.15, 0.15, m.nu) * TAU_MAX).astype(
                np.float32)

            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            if np.any(np.isnan(d.qpos)) or np.any(np.isnan(d.qvel)):
                nan_resets += 1
                mujoco.mj_resetData(m, d)
                d.qpos[2] = rng.uniform(0.5, 1.4)
                u = rng.standard_normal(4)
                d.qpos[3:7] = u / np.linalg.norm(u)
                mujoco.mj_forward(m, d)
                continue

            sn = mj_to_flat(d.qpos.copy(), d.qvel.copy())
            # Reject huge values that cause overflow in normalization
            if (np.any(np.abs(s) > 50) or np.any(np.abs(sn) > 50)
                    or np.any(np.isnan(sn))):
                nan_resets += 1
                mujoco.mj_resetData(m, d)
                d.qpos[2] = rng.uniform(0.5, 1.4)
                u = rng.standard_normal(4)
                d.qpos[3:7] = u / np.linalg.norm(u)
                mujoco.mj_forward(m, d)
                continue

            all_s.append(s)
            all_sn.append(sn)
            all_a.append(tau)
            all_f.append(flags)

        if (ti + 1) % 500 == 0:
            print(f"  gen_data: {ti+1}/{n_traj}, "
                  f"nan_resets={nan_resets}")

    S = np.array(all_s, dtype=np.float32)
    SN = np.array(all_sn, dtype=np.float32)
    A = np.array(all_a, dtype=np.float32)
    F = np.array(all_f, dtype=np.float32)

    cfrac = (F.max(axis=1) > 0).mean()
    per_frac = F.mean(axis=0)
    print(f"  {len(S):,} pairs, any_contact={cfrac:.1%}")
    return S, SN, A, F


def gen_benchmark(n_per_fam, traj_len, seed=99):
    rng = np.random.default_rng(seed)
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    families = {}
    # Use 15% of max torque for stability (matching training data)
    scale = 0.15

    for fam, gen_fn in [
        ("uniform", lambda t, rng: (rng.uniform(-scale, scale, m.nu)
                                    * TAU_MAX)),
        ("step", lambda t, rng: scale * TAU_MAX
                 * (2*(rng.random(m.nu) > 0.5) - 1)
                 if t % 80 == 0 else None),
        ("chirp", lambda t, rng: scale * TAU_MAX * np.sin(
            2*np.pi*(0.1+2.0*t/traj_len)*t*m.opt.timestep)
            * np.ones(m.nu)),
    ]:
        trajs = []
        for _ in range(n_per_fam):
            mujoco.mj_resetData(m, d)
            d.qpos[2] = rng.uniform(0.6, 1.2)
            u = rng.standard_normal(4)
            d.qpos[3:7] = u / np.linalg.norm(u)
            d.qpos[7:] = rng.uniform(-0.4, 0.1, m.nq - 7)
            d.qvel[:] = rng.uniform(-1, 1, m.nv)
            mujoco.mj_forward(m, d)

            states = [mj_to_flat(d.qpos.copy(), d.qvel.copy())]
            actions = []
            cur_tau = (rng.uniform(-scale, scale, m.nu)
                       * TAU_MAX).astype(np.float32)
            for t in range(traj_len):
                # NaN guard
                if np.any(np.isnan(d.qpos)):
                    break
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


# ── Limb-Factored architecture ────────────────────────────────────────
class LimbFactoredMLP(nn.Module):
    """Separate gain/bias per kinematic subtree."""
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


# ── Benchmark ─────────────────────────────────────────────────────────
def benchmark_humanoid(model_nn, families, device, contact_mode):
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    results = {}

    for fam, trajs in families.items():
        mses = []
        for ref_s, ref_a in trajs:
            T = len(ref_a)
            flat_to_mj(ref_s[0], m, d)
            errs = []
            for t in range(T):
                if np.any(np.isnan(d.qpos)):
                    errs.append(100.0)  # penalty for divergence
                    break
                s = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if np.any(np.isnan(s)):
                    errs.append(100.0)
                    break
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
                d.ctrl[:] = np.clip(tau, -TAU_MAX, TAU_MAX)
                mujoco.mj_step(m, d)
                actual = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if np.any(np.isnan(actual)):
                    errs.append(100.0)
                    break
                errs.append(float(np.mean((actual - ref_s[t+1])**2)))
            mses.append(np.mean(errs))
        results[fam] = float(np.mean(mses))
    results["AGGREGATE"] = float(
        np.mean([results[k] for k in ["uniform", "step", "chirp"]]))
    return results


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-traj", type=int, default=2000)
    parser.add_argument("--traj-len", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = "outputs/humanoid"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"STAGE 4: FULL HUMANOID — {NU} actuated DOF")
    print(f"  state_dim={SD}, limbs={LIMB_DOFS}")
    print("=" * 60)

    print("\nGenerating training data...")
    t0 = time.time()
    S, SN, A, F = gen_data(args.n_traj, args.traj_len)
    print(f"  Time: {time.time()-t0:.0f}s")

    print("\nGenerating benchmark refs...")
    bench_fams = gen_benchmark(15, args.traj_len)

    # RawMLP internally does 2*state_dim for input
    # For no-contact: input=[state, target]=2*SD → RawMLP(SD, NU)
    # For contact: input=[state, target, flags]=2*SD+15 → use plain MLP
    raw_no_contact = RawMLP(SD, NU, hidden=1024, layers=4)

    # For contact variants, use a plain MLP since input isn't 2*state_dim
    contact_input_dim = 2 * SD + N_BODY_GEOMS  # 54+54+15=123
    raw_contact_net = nn.Sequential(
        nn.Linear(contact_input_dim, 1024), nn.ELU(),
        nn.Linear(1024, 1024), nn.ELU(),
        nn.Linear(1024, 1024), nn.ELU(),
        nn.Linear(1024, NU),
    )
    # Wrap in a simple class with set_norm
    class PlainMLP(nn.Module):
        def __init__(self, net, in_dim):
            super().__init__()
            self.net = net
            self.register_buffer("mu", torch.zeros(in_dim))
            self.register_buffer("sigma", torch.ones(in_dim))
        def set_norm(self, m, s):
            self.mu.copy_(torch.from_numpy(m).float())
            self.sigma.copy_(torch.from_numpy(s).float())
        def forward(self, x):
            return self.net((x - self.mu) / self.sigma)

    configs = [
        ("raw_mlp", "none", raw_no_contact),
        ("raw_mlp_contact", "perlink",
         PlainMLP(raw_contact_net, contact_input_dim)),
        ("limb_contact", "perlink",
         LimbFactoredMLP(SD, LIMB_DOFS, n_contact=N_BODY_GEOMS,
                         hidden=256, layers=3)),
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
        if train_time > 7200:
            print(f"  ⚠ Budget exceeded: {train_time:.0f}s > 7200s")

        bench_r = benchmark_humanoid(nn_model, bench_fams, device,
                                     contact_mode)
        bench_r["val_loss"] = val_loss
        bench_r["params"] = npar
        bench_r["train_time"] = train_time
        all_results[label] = bench_r

        print(f"  AGG={bench_r['AGGREGATE']:.4e}  time={train_time:.0f}s")

    print(f"\n{'='*60}")
    print("STAGE 4 HUMANOID SUMMARY")
    print(f"{'='*60}")
    for name, r in all_results.items():
        print(f"  {name:<25s} AGG={r['AGGREGATE']:.4e}  "
              f"params={r['params']:,}")

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_dir}/results.json")


if __name__ == "__main__":
    main()
