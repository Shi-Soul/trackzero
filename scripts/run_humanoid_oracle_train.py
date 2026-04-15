#!/usr/bin/env python3
"""Oracle upper-bound: train on benchmark-distribution data.

The definitive test of the coverage hypothesis. We generate training
data using the SAME torque patterns as the benchmark (step, chirp,
uniform structured), then evaluate on the benchmark.

If this works: the supervised approach IS viable at 21 DOF,
just needs targeted coverage. Implies Stage 1C/1D methods matter.

If this fails: there's a deeper problem beyond coverage.

Usage:
    CUDA_VISIBLE_DEVICES=6 python -m scripts.run_humanoid_oracle_train
"""
import json, os, time
import mujoco
import numpy as np
import torch
import torch.nn as nn

from scripts.run_humanoid import (
    HUMANOID_XML, N_BODY_GEOMS, TAU_MAX,
    mj_to_flat, flat_to_mj, get_contact_flags,
    gen_data, gen_benchmark,
)
from scripts.models_structured import _build_mlp
from scripts.run_structured import train_model

FLAT_STATE_DIM = 54
TAU_MAX_NP = np.array(TAU_MAX, dtype=np.float32)
OUT = "outputs/humanoid_oracle_train"
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


def gen_structured_data(n_traj, traj_len, seed=42):
    """Generate data using structured torque patterns (step, chirp, mixed).

    These are the same patterns used in the benchmark, generating
    training data from the benchmark distribution.
    """
    rng = np.random.RandomState(seed)
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    nu = m.nu

    all_S, all_SN, all_A, all_F = [], [], [], []
    nan_resets = 0
    patterns = ["uniform", "step", "chirp"]

    for i in range(n_traj):
        if i % 500 == 0 and i > 0:
            print(f"  gen_structured: {i}/{n_traj}")

        # Pick pattern
        pat = patterns[i % 3]

        # Generate torque sequence
        if pat == "uniform":
            # Random constant torque per joint
            amp = rng.uniform(-1, 1, size=nu) * TAU_MAX_NP * 0.15
            torques = np.tile(amp, (traj_len, 1))
        elif pat == "step":
            # Step function: change at random time
            t_switch = rng.randint(traj_len // 4, 3 * traj_len // 4)
            amp1 = rng.uniform(-1, 1, size=nu) * TAU_MAX_NP * 0.15
            amp2 = rng.uniform(-1, 1, size=nu) * TAU_MAX_NP * 0.15
            torques = np.zeros((traj_len, nu))
            torques[:t_switch] = amp1
            torques[t_switch:] = amp2
        else:  # chirp
            # Chirp: increasing frequency
            t = np.arange(traj_len) * dt
            freq = np.linspace(0.5, 5.0, traj_len)
            phase = rng.uniform(0, 2*np.pi, size=nu)
            amp = rng.uniform(0.5, 1.0, size=nu) * TAU_MAX_NP * 0.15
            torques = np.zeros((traj_len, nu))
            for j in range(nu):
                torques[:, j] = amp[j] * np.sin(
                    2*np.pi * np.cumsum(freq) * dt + phase[j])

        # Reset to standing
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


def eval_h1(model_nn, families, device):
    """Evaluate H=1 (single-step) benchmark."""
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


def eval_full(model_nn, families, device, horizon=500):
    """Evaluate full trajectory tracking."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    fam_errors = {}
    for fam, trajs in families.items():
        errors = []
        for ref_s, ref_a in trajs:
            flat_to_mj(ref_s[0], m, d)
            mujoco.mj_forward(m, d)
            traj_err = 0.0
            T = min(horizon, len(ref_a))
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


def main():
    print("=" * 60)
    print("HUMANOID ORACLE TRAINING (BENCHMARK DISTRIBUTION)")
    print("=" * 60)

    # Generate benchmark
    print("\n[1] Generating benchmark (20 per family, 500 steps)...")
    families = gen_benchmark(20, 500, seed=99)

    results = {}

    # Config 1: Random-only (baseline)
    print("\n[2] Baseline: random-torque data only...")
    t0 = time.time()
    S, SN, A, F = gen_data(2000, 500, seed=42)
    print(f"    {len(S):,} pairs in {time.time()-t0:.0f}s")

    in_dim = 2 * FLAT_STATE_DIM + N_BODY_GEOMS
    X = np.concatenate([S, SN, F], axis=1)
    model = PlainMLP(in_dim, len(TAU_MAX)).to(device)
    train_model(model, X, A, device, epochs=200, bs=2048)
    model.eval()

    h1 = eval_h1(model, families, device)
    h500 = eval_full(model, families, device, 500)
    h1_agg = float(np.mean(list(h1.values())))
    h500_agg = float(np.mean(list(h500.values())))
    print(f"    H=1 AGG={h1_agg:.4e}, H=500 AGG={h500_agg:.4e}")
    results["random_only"] = {
        "H1": h1, "H1_AGG": h1_agg,
        "H500": h500, "H500_AGG": h500_agg,
        "pairs": len(S)}

    # Config 2: Structured-only
    print("\n[3] Structured-torque data only (2K traj)...")
    t0 = time.time()
    sS, sSN, sA, sF = gen_structured_data(2000, 500, seed=42)
    print(f"    {len(sS):,} pairs in {time.time()-t0:.0f}s")

    X2 = np.concatenate([sS, sSN, sF], axis=1)
    model2 = PlainMLP(in_dim, len(TAU_MAX)).to(device)
    train_model(model2, X2, sA, device, epochs=200, bs=2048)
    model2.eval()

    h1_2 = eval_h1(model2, families, device)
    h500_2 = eval_full(model2, families, device, 500)
    h1_2_agg = float(np.mean(list(h1_2.values())))
    h500_2_agg = float(np.mean(list(h500_2.values())))
    print(f"    H=1 AGG={h1_2_agg:.4e}, H=500 AGG={h500_2_agg:.4e}")
    results["structured_only"] = {
        "H1": h1_2, "H1_AGG": h1_2_agg,
        "H500": h500_2, "H500_AGG": h500_2_agg,
        "pairs": len(sS)}

    # Config 3: Mixed (random + structured)
    print("\n[4] Mixed data (random + structured)...")
    mS = np.concatenate([S, sS])
    mSN = np.concatenate([SN, sSN])
    mA = np.concatenate([A, sA])
    mF = np.concatenate([F, sF])
    X3 = np.concatenate([mS, mSN, mF], axis=1)
    model3 = PlainMLP(in_dim, len(TAU_MAX)).to(device)
    train_model(model3, X3, mA, device, epochs=200, bs=2048)
    model3.eval()

    h1_3 = eval_h1(model3, families, device)
    h500_3 = eval_full(model3, families, device, 500)
    h1_3_agg = float(np.mean(list(h1_3.values())))
    h500_3_agg = float(np.mean(list(h500_3.values())))
    print(f"    H=1 AGG={h1_3_agg:.4e}, H=500 AGG={h500_3_agg:.4e}")
    results["mixed"] = {
        "H1": h1_3, "H1_AGG": h1_3_agg,
        "H500": h500_3, "H500_AGG": h500_3_agg,
        "pairs": len(mS)}

    # Summary
    print("\n" + "=" * 60)
    print("ORACLE TRAINING SUMMARY")
    print("=" * 60)
    print(f"{'Config':>20} {'Pairs':>10} {'H=1 AGG':>12} "
          f"{'H=500 AGG':>12}")
    for name, r in results.items():
        print(f"{name:>20} {r['pairs']:>10,} {r['H1_AGG']:>12.4e} "
              f"{r['H500_AGG']:>12.4e}")

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT}/results.json")


if __name__ == "__main__":
    main()
