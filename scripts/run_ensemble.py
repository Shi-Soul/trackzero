#!/usr/bin/env python3
"""Ensemble experiment: does averaging reduce H1?

Trains 5 models with different seeds on the SAME no_bench data.
Tests individual and ensemble-averaged predictions.

If ensemble >> individual: error is due to optimization noise.
If ensemble ≈ individual: error is systematic (data/architecture).

Budget: 1 GPU, ~40 min (shared data gen, 5 × training).

Usage:
    CUDA_VISIBLE_DEVICES=7 python -m scripts.run_ensemble
"""
import json, os, time
import numpy as np
import torch
import torch.nn as nn
import mujoco

from scripts.run_humanoid import (
    HUMANOID_XML, N_BODY_GEOMS, TAU_MAX,
    mj_to_flat, flat_to_mj, get_contact_flags,
    gen_benchmark,
)
from scripts.models_structured import _build_mlp
from scripts.run_structured import train_model

FLAT_DIM = 54
TAU_MAX_NP = np.array(TAU_MAX, dtype=np.float32)
NO_BENCH = ["white", "ou", "brownian", "sine", "bang_bang", "ramp"]

OUT = "outputs/ensemble"
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
            self.norm_m.device)
        self.norm_s = torch.tensor(s, dtype=torch.float32).to(
            self.norm_s.device)

    def forward(self, x):
        return self.net((x - self.norm_m) / (self.norm_s + 1e-8))


def gen_patterned(n_traj, traj_len, patterns, seed=42):
    """Generate data with specified torque patterns."""
    rng = np.random.RandomState(seed)
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    nu = m.nu
    all_S, all_SN, all_A, all_F = [], [], [], []

    for i in range(n_traj):
        if i % 500 == 0:
            print(f"  gen: {i}/{n_traj}")
        pat = patterns[i % len(patterns)]
        scale = 0.15

        if pat == "white":
            torques = rng.uniform(-1, 1, (traj_len, nu)) * TAU_MAX_NP * scale
        elif pat == "ou":
            tau = np.zeros(nu)
            torques = np.zeros((traj_len, nu))
            for t in range(traj_len):
                tau += 0.15 * (-tau) * dt + 0.3 * np.sqrt(dt) * rng.randn(nu)
                torques[t] = np.clip(tau * TAU_MAX_NP, -TAU_MAX_NP * scale,
                                     TAU_MAX_NP * scale)
        elif pat == "brownian":
            raw = np.cumsum(rng.randn(traj_len, nu) * 0.01, axis=0)
            torques = np.clip(raw * TAU_MAX_NP, -TAU_MAX_NP * scale,
                              TAU_MAX_NP * scale)
        elif pat == "sine":
            freq = rng.uniform(0.1, 10.0, nu)
            phase = rng.uniform(0, 2*np.pi, nu)
            amp = rng.uniform(0.3, 1.0, nu) * TAU_MAX_NP * scale
            t_arr = np.arange(traj_len)[:, None] * dt
            torques = amp * np.sin(2*np.pi * freq * t_arr + phase)
        elif pat == "bang_bang":
            torques = np.zeros((traj_len, nu))
            for j in range(nu):
                sw = sorted(rng.randint(0, traj_len, rng.randint(3, 10)))
                sw = [0] + list(sw) + [traj_len]
                for k in range(len(sw)-1):
                    torques[sw[k]:sw[k+1], j] = (
                        rng.choice([-1, 1]) * TAU_MAX_NP[j] * scale)
        else:
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
                mujoco.mj_resetData(m, d)
                mujoco.mj_forward(m, d)
                continue
            all_S.append(s)
            all_SN.append(sn)
            all_A.append(tau)
            all_F.append(flags)

    return (np.array(all_S, dtype=np.float32),
            np.array(all_SN, dtype=np.float32),
            np.array(all_A, dtype=np.float32),
            np.array(all_F, dtype=np.float32))


def train_with_seed(X, Y, dev, seed):
    """Train with specific random seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    in_dim = X.shape[1]
    model = PlainMLP(in_dim, len(TAU_MAX)).to(dev)
    model, vl = train_model(model, X, Y, dev, epochs=200)
    return model, vl


def eval_h1_single(model_nn, families, dev):
    """H=1 eval for single model."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    fam_errors = {}
    for fam, trajs in families.items():
        errors = []
        for ref_s, ref_a in trajs:
            step_errs = []
            for t in range(min(500, len(ref_a))):
                if np.any(np.isnan(ref_s[t])) or np.any(np.isnan(ref_s[t+1])):
                    break
                flat_to_mj(ref_s[t], m, d)
                mujoco.mj_forward(m, d)
                flags = get_contact_flags(d)
                inp = np.concatenate([ref_s[t], ref_s[t+1], flags])
                with torch.no_grad():
                    tau = model_nn(torch.tensor(inp, dtype=torch.float32)
                                   .unsqueeze(0).to(dev)).cpu().numpy()[0]
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


def eval_h1_ensemble(models, families, dev):
    """H=1 eval with ensemble averaging."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    fam_errors = {}
    for fam, trajs in families.items():
        errors = []
        for ref_s, ref_a in trajs:
            step_errs = []
            for t in range(min(500, len(ref_a))):
                if np.any(np.isnan(ref_s[t])) or np.any(np.isnan(ref_s[t+1])):
                    break
                flat_to_mj(ref_s[t], m, d)
                mujoco.mj_forward(m, d)
                flags = get_contact_flags(d)
                inp = np.concatenate([ref_s[t], ref_s[t+1], flags])
                inp_t = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(dev)
                # Average predictions from all models
                with torch.no_grad():
                    taus = [mdl(inp_t).cpu().numpy()[0] for mdl in models]
                tau = np.mean(taus, axis=0)
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


def main():
    print("=" * 60)
    print("ENSEMBLE EXPERIMENT: Optimization vs Systematic Error")
    print("=" * 60)

    families = gen_benchmark(10, 500, seed=99)
    results = {}
    t_total = time.time()

    # Generate data once
    print("\nGenerating no_bench data (2K traj)...")
    t0 = time.time()
    S, SN, A, F = gen_patterned(2000, 500, NO_BENCH, seed=42)
    print(f"Done in {time.time()-t0:.0f}s, {len(S):,} pairs")

    X = np.concatenate([S, SN, F], axis=1)

    # Train 5 models
    models = []
    for i in range(5):
        print(f"\n--- Model {i+1}/5 (seed={i*111}) ---")
        t0 = time.time()
        mdl, vl = train_with_seed(X, A, device, seed=i*111)
        models.append(mdl)
        h1 = eval_h1_single(mdl, families, device)
        h1_agg = float(np.mean(list(h1.values())))
        results[f"model_{i}"] = {
            "h1": h1, "h1_agg": h1_agg, "val_loss": vl,
            "time": time.time() - t0}
        print(f"  H1={h1_agg:.1f}, val_loss={vl:.5f}")

    # Ensemble averaging
    print("\n--- Ensemble (5 models) ---")
    t0 = time.time()
    h1_ens = eval_h1_ensemble(models, families, device)
    h1_ens_agg = float(np.mean(list(h1_ens.values())))
    results["ensemble_5"] = {
        "h1": h1_ens, "h1_agg": h1_ens_agg,
        "time": time.time() - t0}
    print(f"  H1={h1_ens_agg:.1f}")

    # Summary
    total_time = time.time() - t_total
    print(f"\nTOTAL: {total_time:.0f}s ({total_time/60:.1f}min)")
    print("\nSummary:")
    indiv_h1s = [results[f"model_{i}"]["h1_agg"] for i in range(5)]
    print(f"  Individual: mean={np.mean(indiv_h1s):.1f}, "
          f"std={np.std(indiv_h1s):.1f}, "
          f"range=[{np.min(indiv_h1s):.1f}, {np.max(indiv_h1s):.1f}]")
    print(f"  Ensemble:   {h1_ens_agg:.1f}")
    ratio = np.mean(indiv_h1s) / max(h1_ens_agg, 0.01)
    print(f"  Ratio: {ratio:.2f}×")
    if ratio > 1.5:
        print("  → Ensemble helps significantly → optimization noise matters")
    else:
        print("  → Ensemble barely helps → error is systematic")

    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT}/results.json")


if __name__ == "__main__":
    main()
