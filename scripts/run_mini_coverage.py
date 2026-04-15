#!/usr/bin/env python3
"""Mini-humanoid coverage control: does the coverage problem exist at 12 DOF?

Compares random-only vs diverse-torque training on the mini-humanoid.
If diverse patterns improve 12-DOF tracking significantly, the coverage
phase transition is gradual. If not, there's a sharp DOF threshold
between 12 and 21 DOF.

Usage:
    CUDA_VISIBLE_DEVICES=X .venv/bin/python -m scripts.run_mini_coverage
"""
import json, os, time
import numpy as np
import mujoco
import torch
import torch.nn as nn

from scripts.run_mini_humanoid import (
    MINI_HUMANOID_XML, N_BODY_GEOMS,
    mj_to_flat, flat_to_mj, get_contact_flags,
    gen_data, gen_benchmark,
)
from scripts.models_structured import _build_mlp
from scripts.run_structured import train_model

OUT = "outputs/mini_coverage"
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get tau_max from model
_m = mujoco.MjModel.from_xml_string(MINI_HUMANOID_XML)
TAU_MAX_NP = np.array([_m.actuator_ctrlrange[i, 1]
                        for i in range(_m.nu)], dtype=np.float32)
NU = _m.nu
DT = _m.opt.timestep
FLAT_DIM = 36  # mini-humanoid flat state dim


class PlainMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=1024, layers=4):
        super().__init__()
        self.net = _build_mlp(in_dim, out_dim, hidden, layers)
        self.register_buffer("norm_m", torch.zeros(in_dim))
        self.register_buffer("norm_s", torch.ones(in_dim))

    def set_norm(self, m, s):
        self.norm_m = torch.tensor(m, dtype=torch.float32).to(
            next(self.parameters()).device)
        self.norm_s = torch.tensor(s, dtype=torch.float32).to(
            next(self.parameters()).device)

    def forward(self, x):
        return self.net((x - self.norm_m) / (self.norm_s + 1e-8))


def gen_diverse_data(n_traj, traj_len, seed=42):
    """Generate data with diverse torque patterns."""
    rng = np.random.RandomState(seed)
    m = mujoco.MjModel.from_xml_string(MINI_HUMANOID_XML)
    d = mujoco.MjData(m)
    nu = m.nu
    patterns = ["white", "ou", "brownian", "sine",
                "step", "chirp", "bang_bang", "ramp"]

    all_S, all_SN, all_A, all_F = [], [], [], []
    nan_resets = 0

    for i in range(n_traj):
        if i % 500 == 0 and i > 0:
            print(f"    diverse: {i}/{n_traj}")
        pat = patterns[i % len(patterns)]
        scale = 0.15
        torques = _gen_pattern(pat, traj_len, nu, DT, rng,
                               TAU_MAX_NP, scale)

        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)

        for t in range(traj_len):
            s = mj_to_flat(d.qpos.copy(), d.qvel.copy())
            flags = get_contact_flags(d)
            tau = np.clip(torques[t], -TAU_MAX_NP, TAU_MAX_NP)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            sn = mj_to_flat(d.qpos.copy(), d.qvel.copy())
            if np.any(np.isnan(sn)) or np.any(np.isnan(s)):
                nan_resets += 1
                mujoco.mj_resetData(m, d)
                mujoco.mj_forward(m, d)
                continue
            all_S.append(s)
            all_SN.append(sn)
            all_A.append(tau.astype(np.float32))
            all_F.append(flags)

    print(f"    {len(all_S):,} pairs, nan_resets={nan_resets}")
    return (np.array(all_S, dtype=np.float32),
            np.array(all_SN, dtype=np.float32),
            np.array(all_A, dtype=np.float32),
            np.array(all_F, dtype=np.float32))


def _gen_pattern(pat, T, nu, dt, rng, tau_max, scale):
    """Generate one torque trajectory for a given pattern."""
    if pat == "white":
        return rng.uniform(-1, 1, (T, nu)) * tau_max * scale
    elif pat == "ou":
        tau = np.zeros(nu)
        out = np.zeros((T, nu))
        for t in range(T):
            tau += 0.15 * (-tau) * dt + 0.3 * np.sqrt(dt) * rng.randn(nu)
            out[t] = np.clip(tau * tau_max, -tau_max * scale,
                             tau_max * scale)
        return out
    elif pat == "brownian":
        inc = rng.randn(T, nu) * 0.01
        return np.clip(np.cumsum(inc, 0) * tau_max,
                       -tau_max * scale, tau_max * scale)
    elif pat == "sine":
        freq = rng.uniform(0.1, 10.0, nu)
        phase = rng.uniform(0, 2 * np.pi, nu)
        amp = rng.uniform(0.3, 1.0, nu) * tau_max * scale
        t = np.arange(T)[:, None] * dt
        return amp * np.sin(2 * np.pi * freq * t + phase)
    elif pat == "step":
        n_s = rng.randint(2, 8)
        bd = sorted(rng.randint(0, T, n_s))
        bd = [0] + list(bd) + [T]
        out = np.zeros((T, nu))
        for j in range(len(bd) - 1):
            out[bd[j]:bd[j+1]] = rng.uniform(-1, 1, nu) * tau_max * scale
        return out
    elif pat == "chirp":
        t = np.arange(T) * dt
        f0, f1 = rng.uniform(0.1, 2), rng.uniform(3, 15)
        freq = np.linspace(f0, f1, T)
        phase = rng.uniform(0, 2 * np.pi, nu)
        amp = rng.uniform(0.5, 1.0, nu) * tau_max * scale
        out = np.zeros((T, nu))
        for j in range(nu):
            out[:, j] = amp[j] * np.sin(
                2 * np.pi * np.cumsum(freq) * dt + phase[j])
        return out
    elif pat == "bang_bang":
        out = np.zeros((T, nu))
        for j in range(nu):
            sw = sorted(rng.randint(0, T, rng.randint(3, 10)))
            sw = [0] + list(sw) + [T]
            for k in range(len(sw) - 1):
                out[sw[k]:sw[k+1], j] = rng.choice([-1, 1]) * \
                    tau_max[j] * scale
        return out
    else:  # ramp
        start = rng.uniform(-1, 1, nu) * tau_max * scale
        end = rng.uniform(-1, 1, nu) * tau_max * scale
        alpha = np.linspace(0, 1, T)[:, None]
        return start + alpha * (end - start)


def eval_h1(model_nn, families, dev):
    """H=1 benchmark eval for mini-humanoid."""
    m = mujoco.MjModel.from_xml_string(MINI_HUMANOID_XML)
    d = mujoco.MjData(m)
    fam_errors = {}
    for fam, trajs in families.items():
        errors = []
        for ref_s, ref_a in trajs:
            se = []
            T = min(500, len(ref_a))
            for t in range(T):
                if np.any(np.isnan(ref_s[t])) or \
                   np.any(np.isnan(ref_s[t + 1])):
                    break
                flat_to_mj(ref_s[t], m, d)
                mujoco.mj_forward(m, d)
                flags = get_contact_flags(d)
                inp = np.concatenate([ref_s[t], ref_s[t + 1], flags])
                with torch.no_grad():
                    tau = model_nn(
                        torch.tensor(inp, dtype=torch.float32)
                        .unsqueeze(0).to(dev)).cpu().numpy()[0]
                flat_to_mj(ref_s[t], m, d)
                d.ctrl[:] = np.clip(tau, -TAU_MAX_NP, TAU_MAX_NP)
                mujoco.mj_step(m, d)
                actual = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if np.any(np.isnan(actual)):
                    se.append(100.0)
                else:
                    se.append(float(np.mean(
                        (actual - ref_s[t + 1]) ** 2)))
            if se:
                errors.append(float(np.mean(se)))
        fam_errors[fam] = float(np.mean(errors)) if errors else 1e6
    return fam_errors


def eval_h500(model_nn, families, dev):
    """H=500 rollout eval."""
    m = mujoco.MjModel.from_xml_string(MINI_HUMANOID_XML)
    d = mujoco.MjData(m)
    fam_errors = {}
    for fam, trajs in families.items():
        errors = []
        for ref_s, ref_a in trajs:
            flat_to_mj(ref_s[0], m, d)
            mujoco.mj_forward(m, d)
            traj_err = 0.0
            T = min(500, len(ref_a))
            for t in range(T):
                s_now = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if np.any(np.isnan(s_now)):
                    traj_err += 100.0 * (T - t)
                    break
                flags = get_contact_flags(d)
                inp = np.concatenate([s_now, ref_s[t + 1], flags])
                with torch.no_grad():
                    tau = model_nn(
                        torch.tensor(inp, dtype=torch.float32)
                        .unsqueeze(0).to(dev)).cpu().numpy()[0]
                d.ctrl[:] = np.clip(tau, -TAU_MAX_NP, TAU_MAX_NP)
                mujoco.mj_step(m, d)
                actual = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if np.any(np.isnan(actual)):
                    traj_err += 100.0 * (T - t)
                    break
                traj_err += float(np.mean(
                    (actual - ref_s[t + 1]) ** 2))
            errors.append(traj_err / T)
        fam_errors[fam] = float(np.mean(errors))
    return fam_errors


def train_eval(S, SN, A, F, label, families, dev):
    """Train and evaluate."""
    in_dim = 2 * FLAT_DIM + N_BODY_GEOMS
    X = np.concatenate([S, SN, F], axis=1)
    model = PlainMLP(in_dim, NU).to(dev)
    print(f"  Training {label} ({len(S):,} pairs)...")
    train_model(model, X, A, dev, epochs=200, bs=2048)
    model.eval()
    h1 = eval_h1(model, families, dev)
    h500 = eval_h500(model, families, dev)
    h1_agg = float(np.mean(list(h1.values())))
    h500_agg = float(np.mean(list(h500.values())))
    print(f"  {label}: H1={h1_agg:.4e}  H500={h500_agg:.4e}")
    return {"H1": h1, "H1_AGG": h1_agg,
            "H500": h500, "H500_AGG": h500_agg,
            "pairs": len(S)}


def main():
    print("=" * 60)
    print("MINI-HUMANOID COVERAGE CONTROL (12 DOF)")
    print("=" * 60)

    families = gen_benchmark(20, 500, seed=99)
    results = {}

    # Random-only baseline (use existing gen_data)
    print("\n[1] Random-only baseline (2K traj)...")
    t0 = time.time()
    S, SN, A, F = gen_data(2000, 500, seed=42)
    print(f"    Generated in {time.time()-t0:.0f}s")
    results["random_only"] = train_eval(
        S, SN, A, F, "random_only", families, device)

    # Diverse patterns (8 types)
    print("\n[2] Diverse patterns (2K traj, 8 types)...")
    t0 = time.time()
    S2, SN2, A2, F2 = gen_diverse_data(2000, 500, seed=42)
    print(f"    Generated in {time.time()-t0:.0f}s")
    results["diverse"] = train_eval(
        S2, SN2, A2, F2, "diverse", families, device)

    # Summary
    print("\n" + "=" * 60)
    print("MINI-HUMANOID COVERAGE RESULTS")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name:>15}: H1={r['H1_AGG']:.4e} "
              f"H500={r['H500_AGG']:.4e} ({r['pairs']:,} pairs)")

    print("\nFor reference (full humanoid):")
    print("  random_only:  H1=3,278    H500=1.37e+13")
    print("  diverse:      H1=69.6     H500=5.72e+09")
    print("  (47× improvement with diverse patterns)")

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT}/results.json")


if __name__ == "__main__":
    main()
