#!/usr/bin/env python3
"""Factored architecture + contact dynamics.

Tests whether factored MLP rescues contact-aware input at higher DOF.
Previously, contact-aware binary flag helped 9.2× at 2-DOF but was
useless at 3-5 DOF with flat MLP. With factored architecture breaking
the DOF scaling barrier, contact-aware input might work again.

Conditions per DOF:
  1. factored, no contact
  2. factored, with contact, no flag
  3. factored, with contact, contact-aware flag

Usage:
    CUDA_VISIBLE_DEVICES=3 python -m scripts.run_factored_contact --n-links 3
"""
import argparse, json, os, time
import mujoco
import numpy as np
import torch
import torch.nn as nn

from trackzero.config import Config
from scripts.models_structured import FactoredMLP, ContactFactoredMLP
from scripts.run_structured import (
    build_chain_xml, generate_benchmark, train_model, benchmark_model,
)


def build_chain_contact_xml(n_links):
    """N-link chain with ground plane."""
    cfg = Config()
    p, s = cfg.pendulum, cfg.simulation
    L, m = p.link_length, p.link_mass
    ix, iy, iz = p.link_inertia
    damp, tau, g, dt = p.joint_damping, p.tau_max, p.gravity, s.dt
    floor_z = -(n_links * L * 0.6)

    body_lines, close_lines = [], []
    for i in range(1, n_links + 1):
        pre = "    " * (i + 1)
        pos = "0 0 0" if i == 1 else f"0 0 -{L}"
        body_lines.append(f'{pre}<body name="link{i}" pos="{pos}">')
        body_lines.append(f'{pre}  <joint name="joint{i}" type="hinge"/>')
        body_lines.append(
            f'{pre}  <inertial pos="0 0 -{L/2}" mass="{m}"'
            f' diaginertia="{ix} {iy} {iz}"/>'
        )
        body_lines.append(f'{pre}  <geom name="geom{i}" fromto="0 0 0 0 0 -{L}"/>')
        close_lines.append(f'{pre}</body>')

    bodies = "\n".join(body_lines) + "\n" + "\n".join(reversed(close_lines))
    actuators = "\n".join(
        f'    <motor name="motor{i}" joint="joint{i}" ctrllimited="true"'
        f' ctrlrange="{-tau} {tau}"/>'
        for i in range(1, n_links + 1)
    )
    return f"""\
<mujoco model="chain_{n_links}link_contact">
  <option gravity="0 0 -{g}" timestep="{dt}" integrator="RK4"/>
  <default>
    <joint axis="0 1 0" damping="{damp}" limited="false"/>
    <geom type="capsule" size="0.02" rgba="0.4 0.6 0.8 1"
          condim="3" friction="0.5 0.5 0.005"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 {floor_z}"
          size="5 5 0.01" rgba="0.9 0.9 0.9 1"
          condim="3" friction="0.5 0.5 0.005"/>
{bodies}
  </worldbody>
  <actuator>
{actuators}
  </actuator>
</mujoco>
"""


def gen_contact_data(xml, n_traj, traj_len, tau_max, seed=42):
    """Random rollout data with contact flags."""
    rng = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    nq, nv, nu = model.nq, model.nv, model.nu

    all_s, all_sn, all_a, all_c = [], [], [], []
    for _ in range(n_traj):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = rng.uniform(-np.pi, np.pi, size=nq)
        data.qvel[:] = rng.uniform(-5, 5, size=nv)
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
    cfrac = C.mean()
    print(f"  {len(S):,} pairs, contact fraction={cfrac:.1%}")
    return S, SN, A, C


def gen_contact_refs(xml, n_per, traj_len, tau_max, seed=9999):
    """Benchmark refs with contact info."""
    rng = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    nq, nv, nu = model.nq, model.nv, model.nu
    dt = model.opt.timestep
    families = {}

    for fam in ["uniform", "step", "chirp"]:
        trajs = []
        for _ in range(n_per):
            mujoco.mj_resetData(model, data)
            data.qpos[:] = rng.uniform(-np.pi, np.pi, size=nq)
            data.qvel[:] = rng.uniform(-3, 3, size=nv)
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
                    f0 = rng.uniform(0.1, 1)
                    f1 = rng.uniform(2, 10)
                    amp = rng.uniform(0.5, 1.0) * tau_max
                    ph = t_arr * (f0 + (f1 - f0) * t_arr / (2 * t_arr[-1]))
                    actions[:, j] = amp * np.sin(2 * np.pi * ph)

            states = np.zeros((T + 1, nq + nv))
            states[0] = np.concatenate([data.qpos.copy(), data.qvel.copy()])
            for t in range(T):
                data.ctrl[:] = np.clip(actions[t], -tau_max, tau_max)
                mujoco.mj_step(model, data)
                states[t + 1] = np.concatenate([data.qpos.copy(), data.qvel.copy()])
            trajs.append((states, actions))
        families[fam] = trajs
    return families


def benchmark_contact(model, xml, families, device, tau_max, use_flag):
    """Benchmark with contact-aware flag during rollout."""
    mj = mujoco.MjModel.from_xml_string(xml)
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
                parts = [s, ref_s[t + 1]]
                if use_flag:
                    parts.append(np.array([float(d.ncon > 0)], dtype=np.float32))
                inp = np.concatenate(parts).astype(np.float32)
                with torch.no_grad():
                    tau = model(
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
    parser.add_argument("--n-links", type=int, required=True)
    parser.add_argument("--n-traj", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    tau_max = cfg.pendulum.tau_max
    n = args.n_links
    nq = n
    sd = 2 * n

    out_dir = f"outputs/factored_contact_{n}link"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"FACTORED + CONTACT: {n}-link chain")
    print("=" * 60)

    # No-contact baseline
    xml_nc = build_chain_xml(n)
    xml_c = build_chain_contact_xml(n)

    configs = [
        ("no_contact", xml_nc, False, False),
        ("contact_baseline", xml_c, True, False),
        ("contact_aware", xml_c, True, True),
    ]

    all_results = {}
    for label, xml, has_contact, use_flag in configs:
        print(f"\n{'='*60}")
        print(f"Config: {label}")
        print(f"{'='*60}")

        # Generate data
        if has_contact:
            S, SN, A, C = gen_contact_data(
                xml, args.n_traj, 500, tau_max
            )
            if use_flag:
                X = np.concatenate([S, SN, C], axis=1)
                in_dim = 2 * sd + 1
            else:
                X = np.concatenate([S, SN], axis=1)
                in_dim = 2 * sd
        else:
            from scripts.run_structured import generate_data
            S, SN, A = generate_data(xml, args.n_traj, 500, tau_max)
            X = np.concatenate([S, SN], axis=1)
            in_dim = 2 * sd

        Y = A

        # Build factored model with appropriate architecture
        if use_flag:
            model = ContactFactoredMLP(sd, nq, hidden=512, layers=4)
        else:
            model = FactoredMLP(sd, nq, hidden=512, layers=4)

        npar = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {npar:,}")

        model, val_loss = train_model(model, X, Y, device, epochs=args.epochs)

        # Benchmark
        if has_contact:
            bench_fams = gen_contact_refs(xml, 50, 500, tau_max)
            bench = benchmark_contact(
                model, xml, bench_fams, device, tau_max, use_flag
            )
        else:
            bench_fams = generate_benchmark(xml, 50, 500, tau_max)
            bench = benchmark_model(model, xml, bench_fams, device, tau_max)

        bench["val_loss"] = val_loss
        bench["params"] = npar
        all_results[label] = bench

        print(f"  AGG={bench['AGGREGATE']:.4e}  "
              f"uni={bench['uniform']:.4e}  "
              f"step={bench['step']:.4e}  "
              f"chirp={bench['chirp']:.4e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {n}-link factored + contact")
    print(f"{'='*60}")
    for name, r in all_results.items():
        print(f"  {name:<25s} AGG={r['AGGREGATE']:.4e}")

    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
