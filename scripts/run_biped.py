#!/usr/bin/env python3
"""Factored architecture on a 2D biped — branching topology test.

The biped has:
  - Torso: 3 unactuated DOF (slide-x, slide-z, hinge-pitch)
  - 2 legs × 2 joints (hip, knee) = 4 actuated DOF
  - Total: 7 DOF, 4 actuators
  - Ground contact on feet

This tests fundamentally different challenges from serial chains:
  1. Branching kinematic tree
  2. Unactuated floating base (nq > nu)
  3. Multi-point contact
  4. Must control torso indirectly through ground reaction forces

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.run_biped
"""
import argparse, json, os, time
import mujoco
import numpy as np
import torch

from scripts.models_structured import FactoredMLP, RawMLP, ContactFactoredMLP
from scripts.run_structured import train_model
from trackzero.config import Config


BIPED_XML = """
<mujoco model="planar_biped">
  <option timestep="0.005" gravity="0 0 -9.81" integrator="Euler"/>
  <default>
    <joint damping="0.5" armature="0.01"/>
    <geom type="capsule" condim="3" friction="1.0 0.5 0.005"
          rgba="0.4 0.6 0.8 1"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0"
          size="10 10 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="torso" pos="0 0 1.2">
      <joint name="root_x" type="slide" axis="1 0 0" damping="0"/>
      <joint name="root_z" type="slide" axis="0 0 1" damping="0"/>
      <joint name="root_pitch" type="hinge" axis="0 1 0" damping="0"/>
      <geom fromto="0 0 -0.15 0 0 0.15" size="0.06" mass="5"/>
      <!-- Left leg -->
      <body name="upper_leg_L" pos="0 0 -0.15">
        <joint name="hip_L" type="hinge" axis="0 1 0"
               range="-1.57 0.52" limited="true"/>
        <geom fromto="0 0 0 0 0 -0.35" size="0.04" mass="2"/>
        <body name="lower_leg_L" pos="0 0 -0.35">
          <joint name="knee_L" type="hinge" axis="0 1 0"
                 range="-2.09 0" limited="true"/>
          <geom fromto="0 0 0 0 0 -0.35" size="0.035" mass="1.5"/>
          <body name="foot_L" pos="0 0 -0.35">
            <geom type="sphere" size="0.05" mass="0.5"
                  rgba="0.8 0.3 0.3 1"/>
          </body>
        </body>
      </body>
      <!-- Right leg -->
      <body name="upper_leg_R" pos="0 0 -0.15">
        <joint name="hip_R" type="hinge" axis="0 1 0"
               range="-1.57 0.52" limited="true"/>
        <geom fromto="0 0 0 0 0 -0.35" size="0.04" mass="2"/>
        <body name="lower_leg_R" pos="0 0 -0.35">
          <joint name="knee_R" type="hinge" axis="0 1 0"
                 range="-2.09 0" limited="true"/>
          <geom fromto="0 0 0 0 0 -0.35" size="0.035" mass="1.5"/>
          <body name="foot_R" pos="0 0 -0.35">
            <geom type="sphere" size="0.05" mass="0.5"
                  rgba="0.8 0.3 0.3 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hip_L" ctrlrange="-30 30" ctrllimited="true"/>
    <motor joint="knee_L" ctrlrange="-30 30" ctrllimited="true"/>
    <motor joint="hip_R" ctrlrange="-30 30" ctrllimited="true"/>
    <motor joint="knee_R" ctrlrange="-30 30" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


def gen_biped_data(n_traj, traj_len, tau_max, seed=42):
    """Generate training data from biped random rollouts."""
    rng = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_string(BIPED_XML)
    data = mujoco.MjData(model)
    nq, nv, nu = model.nq, model.nv, model.nu
    print(f"  Biped: nq={nq}, nv={nv}, nu={nu}")

    all_s, all_sn, all_a, all_c = [], [], [], []
    for _ in range(n_traj):
        mujoco.mj_resetData(model, data)
        # Randomize initial pose: small perturbation from standing
        data.qpos[0] = rng.uniform(-0.5, 0.5)   # x
        data.qpos[1] = rng.uniform(0.8, 1.5)     # z
        data.qpos[2] = rng.uniform(-0.3, 0.3)    # pitch
        data.qpos[3] = rng.uniform(-1.0, 0.3)    # hip_L
        data.qpos[4] = rng.uniform(-1.5, 0.0)    # knee_L
        data.qpos[5] = rng.uniform(-1.0, 0.3)    # hip_R
        data.qpos[6] = rng.uniform(-1.5, 0.0)    # knee_R
        data.qvel[:] = rng.uniform(-2, 2, size=nv)
        mujoco.mj_forward(model, data)

        for _ in range(traj_len):
            s = np.concatenate([data.qpos.copy(), data.qvel.copy()])
            tau = rng.uniform(-tau_max, tau_max, size=nu)
            c = float(data.ncon > 0)
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            sn = np.concatenate([data.qpos.copy(), data.qvel.copy()])
            all_s.append(s)
            all_sn.append(sn)
            all_a.append(tau)
            all_c.append(c)

    S = np.array(all_s, dtype=np.float32)
    SN = np.array(all_sn, dtype=np.float32)
    A = np.array(all_a, dtype=np.float32)
    C = np.array(all_c, dtype=np.float32).reshape(-1, 1)
    print(f"  {len(S):,} pairs, sd={2*nq}, ad={nu}, "
          f"contact fraction={C.mean():.1%}")
    return S, SN, A, C, nq, nv, nu


def gen_biped_benchmark(n_per, traj_len, tau_max, seed=9999):
    """Generate benchmark reference trajectories."""
    rng = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_string(BIPED_XML)
    data = mujoco.MjData(model)
    nq, nv, nu = model.nq, model.nv, model.nu
    dt = model.opt.timestep
    families = {}

    for fam in ["uniform", "step", "chirp"]:
        trajs = []
        for _ in range(n_per):
            mujoco.mj_resetData(model, data)
            data.qpos[0] = rng.uniform(-0.5, 0.5)
            data.qpos[1] = rng.uniform(0.8, 1.5)
            data.qpos[2] = rng.uniform(-0.3, 0.3)
            data.qpos[3] = rng.uniform(-1.0, 0.3)
            data.qpos[4] = rng.uniform(-1.5, 0.0)
            data.qpos[5] = rng.uniform(-1.0, 0.3)
            data.qpos[6] = rng.uniform(-1.5, 0.0)
            data.qvel[:] = rng.uniform(-1, 1, size=nv)
            mujoco.mj_forward(model, data)
            T = traj_len
            actions = np.zeros((T, nu))
            if fam == "uniform":
                actions = rng.uniform(-tau_max, tau_max, size=(T, nu))
            elif fam == "step":
                for j in range(nu):
                    v = rng.uniform(-tau_max, tau_max)
                    sw = rng.integers(T // 4, 3 * T // 4)
                    actions[:sw, j] = v
                    actions[sw:, j] = rng.uniform(-tau_max, tau_max)
            elif fam == "chirp":
                t_arr = np.arange(T) * dt
                for j in range(nu):
                    f0, f1 = rng.uniform(0.1, 1), rng.uniform(2, 10)
                    amp = rng.uniform(0.5, 1.0) * tau_max
                    ph = t_arr * (f0 + (f1 - f0) * t_arr / (2 * t_arr[-1]))
                    actions[:, j] = amp * np.sin(2 * np.pi * ph)

            states = np.zeros((T + 1, nq + nv))
            states[0] = np.concatenate([data.qpos.copy(), data.qvel.copy()])
            for t in range(T):
                data.ctrl[:] = np.clip(actions[t], -tau_max, tau_max)
                mujoco.mj_step(model, data)
                states[t + 1] = np.concatenate(
                    [data.qpos.copy(), data.qvel.copy()]
                )
            trajs.append((states, actions))
        families[fam] = trajs
    return families, nq, nv, nu


def benchmark_biped(model_nn, families, device, tau_max,
                    use_flag=False, raw_mlp_flag=False):
    """Benchmark with closed-loop rollout on biped.
    
    raw_mlp_flag: if True, input is [s, c, sn, c] instead of [s, sn, c].
    """
    mj = mujoco.MjModel.from_xml_string(BIPED_XML)
    d = mujoco.MjData(mj)
    nq = mj.nq
    results = {}

    for fam, trajs in families.items():
        mses = []
        for ref_s, ref_a in trajs:
            T = len(ref_a)
            mujoco.mj_resetData(mj, d)
            d.qpos[:] = ref_s[0, :nq]
            d.qvel[:] = ref_s[0, nq:]
            mujoco.mj_forward(mj, d)
            errs = []
            for t in range(T):
                s = np.concatenate([d.qpos.copy(), d.qvel.copy()])
                c_val = np.array([float(d.ncon > 0)], dtype=np.float32)
                if use_flag:
                    if raw_mlp_flag:
                        parts = [s, c_val, ref_s[t + 1], c_val]
                    else:
                        parts = [s, ref_s[t + 1], c_val]
                else:
                    parts = [s, ref_s[t + 1]]
                inp = np.concatenate(parts).astype(np.float32)
                with torch.no_grad():
                    tau = model_nn(
                        torch.tensor(inp).unsqueeze(0).to(device)
                    ).cpu().numpy()[0]
                d.ctrl[:] = np.clip(tau, -tau_max, tau_max)
                mujoco.mj_step(mj, d)
                actual = np.concatenate([d.qpos.copy(), d.qvel.copy()])
                errs.append(np.mean((actual - ref_s[t + 1]) ** 2))
            mses.append(np.mean(errs))
        results[fam] = float(np.mean(mses))
    results["AGGREGATE"] = float(
        np.mean([results[k] for k in ["uniform", "step", "chirp"]])
    )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-traj", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tau_max = 30.0  # biped has higher torque limits
    out_dir = "outputs/biped"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("BIPED EXPERIMENT: Factored vs Baselines")
    print("  7 DOF (3 unactuated + 4 actuated), branching topology")
    print("=" * 60)

    # Generate data
    print("\nGenerating training data...")
    t0 = time.time()
    S, SN, A, C, nq, nv, nu = gen_biped_data(
        args.n_traj, 500, tau_max
    )
    print(f"  Data generated in {time.time()-t0:.1f}s")

    sd = nq + nv  # state dim = 14

    # Generate benchmark
    print("\nGenerating benchmark...")
    families, _, _, _ = gen_biped_benchmark(50, 500, tau_max)

    # Test configs: no_flag and contact_aware for each arch
    configs = [
        ("raw_mlp", False),
        ("raw_mlp_aware", True),
        ("factored", False),
        ("factored_aware", True),
    ]

    all_results = {}
    for label, use_flag in configs:
        print(f"\n{'='*60}")
        print(f"Config: {label}")
        print(f"{'='*60}")

        # Default input construction
        if use_flag:
            X = np.concatenate([S, SN, C], axis=1)
        else:
            X = np.concatenate([S, SN], axis=1)
        Y = A

        if "factored" in label:
            if use_flag:
                nn_model = ContactFactoredMLP(
                    sd, nu, hidden=512, layers=4
                )
            else:
                nn_model = FactoredMLP(sd, nu, hidden=512, layers=4)
        else:
            # RawMLP doubles state_dim internally, so pass sd (not 2*sd)
            # For contact-aware: append flag to BOTH halves → input=2*(sd+1)
            if use_flag:
                X = np.concatenate([S, C, SN, C], axis=1)
                nn_model = RawMLP(sd + 1, nu, hidden=512, layers=4)
            else:
                nn_model = RawMLP(sd, nu, hidden=512, layers=4)

        npar = sum(p.numel() for p in nn_model.parameters())
        print(f"  Parameters: {npar:,}")

        nn_model, val_loss = train_model(
            nn_model, X, Y, device, epochs=args.epochs, bs=4096
        )

        is_raw = "raw_mlp" in label
        bench = benchmark_biped(
            nn_model, families, device, tau_max,
            use_flag, raw_mlp_flag=(is_raw and use_flag)
        )
        bench["val_loss"] = val_loss
        bench["params"] = npar
        all_results[label] = bench

        print(f"  AGG={bench['AGGREGATE']:.4e}  "
              f"uni={bench['uniform']:.4e}  "
              f"step={bench['step']:.4e}  "
              f"chirp={bench['chirp']:.4e}")

    # Summary
    print(f"\n{'='*60}")
    print("BIPED SUMMARY")
    print(f"{'='*60}")
    for name, r in all_results.items():
        print(f"  {name:<25s} AGG={r['AGGREGATE']:.4e}")

    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
