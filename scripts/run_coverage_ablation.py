#!/usr/bin/env python3
"""Ablation: which torque patterns drive coverage?

Tests subsets of diverse patterns to identify whether the improvement
comes from including benchmark-matching patterns (step/chirp) or from
general diversity. This is the key control experiment for Stage 1C.

Conditions:
  A) no_bench: all patterns EXCEPT step/chirp (6 patterns)
  B) bench_only: ONLY step/chirp (2 patterns)
  C) smooth_only: sine/ramp/OU (3 smooth patterns)
  D) discontinuous_only: bang_bang/step/brownian (3 patterns)

Usage:
    CUDA_VISIBLE_DEVICES=X .venv/bin/python -m scripts.run_coverage_ablation
"""
import json, os, time
import numpy as np
import mujoco
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
OUT = "outputs/coverage_ablation"
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


def gen_torque_pattern(pat, traj_len, nu, dt, rng, scale=0.15):
    """Generate one torque trajectory for a given pattern type."""
    if pat == "white":
        return rng.uniform(-1, 1, (traj_len, nu)) * TAU_MAX_NP * scale
    elif pat == "ou":
        tau = np.zeros(nu)
        torques = np.zeros((traj_len, nu))
        theta, sigma = 0.15, 0.3
        for t in range(traj_len):
            tau += theta * (-tau) * dt + \
                   sigma * np.sqrt(dt) * rng.randn(nu)
            torques[t] = np.clip(tau * TAU_MAX_NP,
                                 -TAU_MAX_NP * scale, TAU_MAX_NP * scale)
        return torques
    elif pat == "brownian":
        inc = rng.randn(traj_len, nu) * 0.01
        raw = np.cumsum(inc, axis=0)
        return np.clip(raw * TAU_MAX_NP,
                       -TAU_MAX_NP * scale, TAU_MAX_NP * scale)
    elif pat == "sine":
        freq = rng.uniform(0.1, 10.0, nu)
        phase = rng.uniform(0, 2 * np.pi, nu)
        amp = rng.uniform(0.3, 1.0, nu) * TAU_MAX_NP * scale
        t = np.arange(traj_len)[:, None] * dt
        return amp * np.sin(2 * np.pi * freq * t + phase)
    elif pat == "step":
        n_steps = rng.randint(2, 8)
        bd = sorted(rng.randint(0, traj_len, n_steps))
        bd = [0] + list(bd) + [traj_len]
        torques = np.zeros((traj_len, nu))
        for j in range(len(bd) - 1):
            a = rng.uniform(-1, 1, nu) * TAU_MAX_NP * scale
            torques[bd[j]:bd[j + 1]] = a
        return torques
    elif pat == "chirp":
        t = np.arange(traj_len) * dt
        f0, f1 = rng.uniform(0.1, 2), rng.uniform(3, 15)
        freq = np.linspace(f0, f1, traj_len)
        phase = rng.uniform(0, 2 * np.pi, nu)
        amp = rng.uniform(0.5, 1.0, nu) * TAU_MAX_NP * scale
        torques = np.zeros((traj_len, nu))
        for j in range(nu):
            torques[:, j] = amp[j] * np.sin(
                2 * np.pi * np.cumsum(freq) * dt + phase[j])
        return torques
    elif pat == "bang_bang":
        torques = np.zeros((traj_len, nu))
        for j in range(nu):
            sw = sorted(rng.randint(0, traj_len, rng.randint(3, 10)))
            sw = [0] + list(sw) + [traj_len]
            for k in range(len(sw) - 1):
                v = rng.choice([-1, 1]) * TAU_MAX_NP[j] * scale
                torques[sw[k]:sw[k + 1], j] = v
        return torques
    elif pat == "ramp":
        start = rng.uniform(-1, 1, nu) * TAU_MAX_NP * scale
        end = rng.uniform(-1, 1, nu) * TAU_MAX_NP * scale
        alpha = np.linspace(0, 1, traj_len)[:, None]
        return start + alpha * (end - start)
    else:
        raise ValueError(f"Unknown pattern: {pat}")


def gen_data_from_patterns(patterns, n_traj, traj_len, seed=42):
    """Generate training data using specified torque patterns."""
    rng = np.random.RandomState(seed)
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    nu = m.nu

    all_S, all_SN, all_A, all_F = [], [], [], []
    nan_resets = 0

    for i in range(n_traj):
        if i % 500 == 0 and i > 0:
            print(f"    {i}/{n_traj}")
        pat = patterns[i % len(patterns)]
        torques = gen_torque_pattern(pat, traj_len, nu, dt, rng)

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

    S = np.array(all_S, dtype=np.float32)
    SN = np.array(all_SN, dtype=np.float32)
    A = np.array(all_A, dtype=np.float32)
    F = np.array(all_F, dtype=np.float32)
    print(f"    {len(S):,} pairs, nan_resets={nan_resets}")
    return S, SN, A, F


def eval_h1(model_nn, families, dev):
    """Evaluate single-step benchmark accuracy."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    fam_errors = {}
    for fam, trajs in families.items():
        errors = []
        for ref_s, ref_a in trajs:
            se = []
            for t in range(min(500, len(ref_a))):
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
    """Evaluate full-trajectory rollout."""
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
    """Train model and evaluate on benchmark."""
    in_dim = 2 * FLAT_STATE_DIM + N_BODY_GEOMS
    X = np.concatenate([S, SN, F], axis=1)
    model = PlainMLP(in_dim, len(TAU_MAX)).to(dev)
    print(f"  Training {label} ({len(S):,} pairs)...")
    train_model(model, X, A, dev, epochs=200, bs=2048)
    model.eval()
    h1 = eval_h1(model, families, dev)
    h500 = eval_h500(model, families, dev)
    h1_agg = float(np.mean(list(h1.values())))
    h500_agg = float(np.mean(list(h500.values())))
    print(f"  {label}: H1={h1_agg:.4e} H500={h500_agg:.4e}")
    return {"H1": h1, "H1_AGG": h1_agg,
            "H500": h500, "H500_AGG": h500_agg,
            "pairs": len(S)}


# Ablation conditions
ABLATIONS = {
    "no_bench": ["white", "ou", "brownian", "sine", "bang_bang", "ramp"],
    "bench_only": ["step", "chirp"],
    "smooth_only": ["sine", "ramp", "ou"],
    "discontinuous": ["bang_bang", "step", "brownian"],
    "all_8": ["white", "ou", "brownian", "sine", "step",
              "chirp", "bang_bang", "ramp"],
}


def main():
    print("=" * 60)
    print("COVERAGE ABLATION (Stage 1C)")
    print("=" * 60)

    families = gen_benchmark(20, 500, seed=99)
    results = {}

    for name, patterns in ABLATIONS.items():
        print(f"\n[{name}] patterns={patterns}")
        t0 = time.time()
        S, SN, A, F = gen_data_from_patterns(
            patterns, 2000, 500, seed=42)
        print(f"    Generated in {time.time() - t0:.0f}s")
        results[name] = train_eval(
            S, SN, A, F, name, families, device)
        results[name]["patterns"] = patterns

    print("\n" + "=" * 60)
    print("COVERAGE ABLATION SUMMARY")
    print("=" * 60)
    print(f"{'Condition':>20} {'Patterns':>8} {'Pairs':>10} "
          f"{'H=1 AGG':>12} {'H=500 AGG':>12}")
    for name, r in results.items():
        n_pat = len(ABLATIONS[name])
        print(f"{name:>20} {n_pat:>8} {r['pairs']:>10,} "
              f"{r['H1_AGG']:>12.4e} {r['H500_AGG']:>12.4e}")

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT}/results.json")


if __name__ == "__main__":
    main()
