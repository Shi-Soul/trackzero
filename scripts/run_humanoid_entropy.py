#!/usr/bin/env python3
"""Entropy-driven data generation for humanoid (Stage 1C test).

Generates training data using strategies that maximize state-space
coverage WITHOUT knowing the benchmark distribution. Tests whether
principled exploration can match the oracle-matched coverage result.

Strategies:
1. diverse_random: Mix of random torque patterns (OU, brownian, sine)
2. grid_init: Start from diverse initial states (not just standing)
3. adaptive: Bin state space, oversample under-visited regions

Usage:
    CUDA_VISIBLE_DEVICES=4 python -m scripts.run_humanoid_entropy
"""
import json, os, time, collections
import mujoco
import numpy as np
import torch
import torch.nn as nn

from scripts.run_humanoid import (
    HUMANOID_XML, N_BODY_GEOMS, TAU_MAX,
    mj_to_flat, flat_to_mj, get_contact_flags,
    gen_benchmark,
)
from scripts.models_structured import _build_mlp
from scripts.run_structured import train_model

FLAT_STATE_DIM = 54
TAU_MAX_NP = np.array(TAU_MAX, dtype=np.float32)
OUT = "outputs/humanoid_entropy"
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def gen_diverse_random(n_traj, traj_len, seed=42):
    """Mix of torque patterns for maximum diversity."""
    rng = np.random.RandomState(seed)
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    nu = m.nu

    all_S, all_SN, all_A, all_F = [], [], [], []
    nan_resets = 0
    patterns = ["white", "ou", "brownian", "sine", "step",
                "chirp", "bang_bang", "ramp"]

    for i in range(n_traj):
        if i % 500 == 0 and i > 0:
            print(f"  diverse_random: {i}/{n_traj}")

        pat = patterns[i % len(patterns)]
        scale = 0.15

        if pat == "white":
            torques = rng.uniform(-1, 1, (traj_len, nu)) * \
                      TAU_MAX_NP * scale
        elif pat == "ou":
            # Ornstein-Uhlenbeck
            tau = np.zeros(nu)
            torques = np.zeros((traj_len, nu))
            theta, sigma = 0.15, 0.3
            for t in range(traj_len):
                tau += theta * (-tau) * dt + \
                       sigma * np.sqrt(dt) * rng.randn(nu)
                torques[t] = np.clip(
                    tau * TAU_MAX_NP, -TAU_MAX_NP * scale,
                    TAU_MAX_NP * scale)
        elif pat == "brownian":
            increments = rng.randn(traj_len, nu) * 0.01
            raw = np.cumsum(increments, axis=0)
            torques = np.clip(
                raw * TAU_MAX_NP, -TAU_MAX_NP * scale,
                TAU_MAX_NP * scale)
        elif pat == "sine":
            freq = rng.uniform(0.1, 10.0, nu)
            phase = rng.uniform(0, 2*np.pi, nu)
            amp = rng.uniform(0.3, 1.0, nu) * TAU_MAX_NP * scale
            t = np.arange(traj_len)[:, None] * dt
            torques = amp * np.sin(2*np.pi * freq * t + phase)
        elif pat == "step":
            n_steps = rng.randint(2, 8)
            boundaries = sorted(rng.randint(0, traj_len, n_steps))
            boundaries = [0] + list(boundaries) + [traj_len]
            torques = np.zeros((traj_len, nu))
            for j in range(len(boundaries)-1):
                amp = rng.uniform(-1, 1, nu) * TAU_MAX_NP * scale
                torques[boundaries[j]:boundaries[j+1]] = amp
        elif pat == "chirp":
            t = np.arange(traj_len) * dt
            f0, f1 = rng.uniform(0.1, 2), rng.uniform(3, 15)
            freq = np.linspace(f0, f1, traj_len)
            phase = rng.uniform(0, 2*np.pi, nu)
            amp = rng.uniform(0.5, 1.0, nu) * TAU_MAX_NP * scale
            torques = np.zeros((traj_len, nu))
            for j in range(nu):
                torques[:, j] = amp[j] * np.sin(
                    2*np.pi * np.cumsum(freq) * dt + phase[j])
        elif pat == "bang_bang":
            torques = np.zeros((traj_len, nu))
            for j in range(nu):
                switch_times = sorted(rng.randint(0, traj_len,
                                                   rng.randint(3, 10)))
                switch_times = [0] + list(switch_times) + [traj_len]
                for k in range(len(switch_times)-1):
                    val = rng.choice([-1, 1]) * TAU_MAX_NP[j] * scale
                    torques[switch_times[k]:switch_times[k+1], j] = val
        else:  # ramp
            start = rng.uniform(-1, 1, nu) * TAU_MAX_NP * scale
            end = rng.uniform(-1, 1, nu) * TAU_MAX_NP * scale
            alpha = np.linspace(0, 1, traj_len)[:, None]
            torques = start + alpha * (end - start)

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
            all_A.append(tau)
            all_F.append(flags)

    S = np.array(all_S, dtype=np.float32)
    SN = np.array(all_SN, dtype=np.float32)
    A = np.array(all_A, dtype=np.float32)
    F = np.array(all_F, dtype=np.float32)
    print(f"  {len(S):,} pairs, nan_resets={nan_resets}")
    return S, SN, A, F


def gen_diverse_init(n_traj, traj_len, seed=42):
    """Random torques but from diverse initial states."""
    rng = np.random.RandomState(seed)
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    nu = m.nu
    nq, nv = m.nq, m.nv

    all_S, all_SN, all_A, all_F = [], [], [], []
    nan_resets = 0

    for i in range(n_traj):
        if i % 500 == 0 and i > 0:
            print(f"  diverse_init: {i}/{n_traj}")

        # Randomize initial state
        mujoco.mj_resetData(m, d)
        if i % 3 == 0:
            # Standing (default)
            pass
        elif i % 3 == 1:
            # Perturbed standing
            d.qpos[:] += rng.randn(nq) * 0.3
            d.qvel[:] = rng.randn(nv) * 1.0
        else:
            # Random pose in joint limits
            for j in range(nq):
                if m.jnt_limited[min(j, m.njnt-1)]:
                    lo = m.jnt_range[min(j, m.njnt-1), 0]
                    hi = m.jnt_range[min(j, m.njnt-1), 1]
                    d.qpos[j] = rng.uniform(lo, hi)
                else:
                    d.qpos[j] += rng.randn() * 0.5
            d.qvel[:] = rng.randn(nv) * 2.0

        mujoco.mj_forward(m, d)

        for t in range(traj_len):
            s = mj_to_flat(d.qpos.copy(), d.qvel.copy())
            flags = get_contact_flags(d)
            tau = rng.uniform(-1, 1, nu).astype(np.float32) * \
                  TAU_MAX_NP * 0.15
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
            all_A.append(tau)
            all_F.append(flags)

    S = np.array(all_S, dtype=np.float32)
    SN = np.array(all_SN, dtype=np.float32)
    A = np.array(all_A, dtype=np.float32)
    F = np.array(all_F, dtype=np.float32)
    print(f"  {len(S):,} pairs, nan_resets={nan_resets}")
    return S, SN, A, F


def eval_h1(model_nn, families, device):
    """Evaluate H=1 benchmark."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    fam_errors = {}
    for fam, trajs in families.items():
        errors = []
        for ref_s, ref_a in trajs:
            step_errs = []
            for t in range(min(500, len(ref_a))):
                if np.any(np.isnan(ref_s[t])) or np.any(
                        np.isnan(ref_s[t+1])):
                    break
                flat_to_mj(ref_s[t], m, d)
                mujoco.mj_forward(m, d)
                flags = get_contact_flags(d)
                inp = np.concatenate([ref_s[t], ref_s[t+1], flags])
                with torch.no_grad():
                    tau = model_nn(
                        torch.tensor(inp, dtype=torch.float32)
                        .unsqueeze(0).to(device)
                    ).cpu().numpy()[0]
                flat_to_mj(ref_s[t], m, d)
                d.ctrl[:] = np.clip(tau, -TAU_MAX_NP, TAU_MAX_NP)
                mujoco.mj_step(m, d)
                actual = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if np.any(np.isnan(actual)):
                    step_errs.append(100.0)
                else:
                    step_errs.append(float(np.mean(
                        (actual - ref_s[t+1])**2)))
            if step_errs:
                errors.append(float(np.mean(step_errs)))
        fam_errors[fam] = float(np.mean(errors)) if errors else 1e6
    return fam_errors


def eval_h500(model_nn, families, device):
    """Evaluate H=500 full trajectory."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
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
                inp = np.concatenate([s_now, ref_s[t+1], flags])
                with torch.no_grad():
                    tau = model_nn(
                        torch.tensor(inp, dtype=torch.float32)
                        .unsqueeze(0).to(device)
                    ).cpu().numpy()[0]
                d.ctrl[:] = np.clip(tau, -TAU_MAX_NP, TAU_MAX_NP)
                mujoco.mj_step(m, d)
                actual = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if np.any(np.isnan(actual)):
                    traj_err += 100.0 * (T - t)
                    break
                traj_err += float(np.mean(
                    (actual - ref_s[t+1])**2))
            errors.append(traj_err / T)
        fam_errors[fam] = float(np.mean(errors))
    return fam_errors


def train_eval(S, SN, A, F, label, families, device):
    """Train and evaluate."""
    in_dim = 2 * FLAT_STATE_DIM + N_BODY_GEOMS
    X = np.concatenate([S, SN, F], axis=1)
    model = PlainMLP(in_dim, len(TAU_MAX)).to(device)
    print(f"  Training {label} ({len(S):,} pairs)...")
    train_model(model, X, A, device, epochs=200, bs=2048)
    model.eval()

    h1 = eval_h1(model, families, device)
    h500 = eval_h500(model, families, device)
    h1_agg = float(np.mean(list(h1.values())))
    h500_agg = float(np.mean(list(h500.values())))
    print(f"  {label}: H1={h1_agg:.4e} H500={h500_agg:.4e}")
    return {"H1": h1, "H1_AGG": h1_agg,
            "H500": h500, "H500_AGG": h500_agg,
            "pairs": len(S)}


def main():
    print("=" * 60)
    print("HUMANOID ENTROPY-DRIVEN EXPLORATION (Stage 1C)")
    print("=" * 60)

    families = gen_benchmark(20, 500, seed=99)
    results = {}

    # Strategy 1: Diverse torque patterns
    print("\n[1] Diverse random torques (8 patterns)...")
    t0 = time.time()
    S1, SN1, A1, F1 = gen_diverse_random(2000, 500, seed=42)
    print(f"    Generated in {time.time()-t0:.0f}s")
    results["diverse_random"] = train_eval(
        S1, SN1, A1, F1, "diverse_random", families, device)

    # Strategy 2: Diverse initial conditions
    print("\n[2] Diverse initial states...")
    t0 = time.time()
    S2, SN2, A2, F2 = gen_diverse_init(2000, 500, seed=42)
    print(f"    Generated in {time.time()-t0:.0f}s")
    results["diverse_init"] = train_eval(
        S2, SN2, A2, F2, "diverse_init", families, device)

    # Strategy 3: Combined
    print("\n[3] Combined (diverse torques + diverse init)...")
    S3 = np.concatenate([S1, S2])
    SN3 = np.concatenate([SN1, SN2])
    A3 = np.concatenate([A1, A2])
    F3 = np.concatenate([F1, F2])
    results["combined"] = train_eval(
        S3, SN3, A3, F3, "combined", families, device)

    # Summary
    print("\n" + "=" * 60)
    print("ENTROPY EXPLORATION SUMMARY")
    print("=" * 60)
    print(f"{'Config':>20} {'Pairs':>10} {'H=1 AGG':>12} "
          f"{'H=500 AGG':>12}")
    for name, r in results.items():
        print(f"{name:>20} {r['pairs']:>10,} {r['H1_AGG']:>12.4e} "
              f"{r['H500_AGG']:>12.4e}")

    # Reference: oracle-matched was H1=62, H500=5.72e9
    print("\nReference (from oracle_train):")
    print("  random_only:     H1=3,278    H500=1.37e+13")
    print("  structured_only: H1=87       H500=5.72e+09")
    print("  mixed:           H1=62       H500=5.72e+09")

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT}/results.json")


if __name__ == "__main__":
    main()
