#!/usr/bin/env python3
"""Test coverage hypothesis: generate humanoid training data using ALL
torque patterns (uniform + step + chirp) to match benchmark distribution.

If single-step accuracy improves on benchmark, the humanoid failure is
a coverage problem, not a capacity problem.

Also tests: (a) what happens with higher torque scale (30% vs 15%),
(b) what happens with more diverse initial conditions.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m scripts.run_humanoid_coverage
"""
import json, os, time
import mujoco
import numpy as np
import torch
import torch.nn as nn

from scripts.run_humanoid import (
    HUMANOID_XML, N_BODY_GEOMS, GEOM_OFFSET, TAU_MAX,
    mj_to_flat, flat_to_mj, get_contact_flags, gen_benchmark,
)
from scripts.models_structured import _build_mlp
from scripts.run_structured import train_model

FLAT_STATE_DIM = 54
OUT = "outputs/humanoid_coverage"
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAU_MAX_NP = np.array(TAU_MAX, dtype=np.float32)


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


def gen_diverse_data(n_traj, traj_len, torque_scale=0.15, seed=42):
    """Generate data with diverse torque patterns matching benchmark.

    Each trajectory randomly chooses one of:
    - uniform random (33%)
    - step function (33%)
    - chirp/sinusoidal (33%)

    Also uses more diverse initial conditions.
    """
    rng = np.random.default_rng(seed)
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    all_s, all_sn, all_a, all_f = [], [], [], []
    nan_resets = 0
    pattern_counts = {"uniform": 0, "step": 0, "chirp": 0}

    for ti in range(n_traj):
        # Random init — more diverse than original
        mujoco.mj_resetData(m, d)
        d.qpos[:2] = rng.uniform(-0.5, 0.5, 2)
        d.qpos[2] = rng.uniform(0.3, 1.5)
        u = rng.standard_normal(4)
        d.qpos[3:7] = u / np.linalg.norm(u)
        d.qpos[7:] = rng.uniform(-0.6, 0.3, m.nq - 7)
        d.qvel[:] = rng.uniform(-2, 2, m.nv)
        mujoco.mj_forward(m, d)

        # Choose torque pattern
        pattern = rng.choice(["uniform", "step", "chirp"])
        pattern_counts[pattern] += 1

        # Generate pattern-specific torque
        if pattern == "step":
            cur_tau = (torque_scale * TAU_MAX_NP
                       * (2*(rng.random(m.nu) > 0.5) - 1))
        elif pattern == "chirp":
            base_freq = rng.uniform(0.05, 0.5)
            chirp_rate = rng.uniform(0.5, 3.0)
            phase = rng.uniform(0, 2*np.pi, m.nu)
        else:
            cur_tau = None

        for t in range(traj_len):
            if np.any(np.isnan(d.qpos)) or np.any(np.isnan(d.qvel)):
                nan_resets += 1
                mujoco.mj_resetData(m, d)
                d.qpos[2] = rng.uniform(0.3, 1.5)
                u = rng.standard_normal(4)
                d.qpos[3:7] = u / np.linalg.norm(u)
                mujoco.mj_forward(m, d)
                continue

            s = mj_to_flat(d.qpos.copy(), d.qvel.copy())
            flags = get_contact_flags(d)

            # Generate torque based on pattern
            if pattern == "uniform":
                tau = (rng.uniform(-torque_scale, torque_scale, m.nu)
                       * TAU_MAX_NP).astype(np.float32)
            elif pattern == "step":
                if t % 80 == 0:
                    cur_tau = (torque_scale * TAU_MAX_NP
                               * (2*(rng.random(m.nu) > 0.5) - 1))
                tau = cur_tau.astype(np.float32)
            else:  # chirp
                freq = base_freq + chirp_rate * t / traj_len
                tau = (torque_scale * TAU_MAX_NP
                       * np.sin(2*np.pi*freq*t*m.opt.timestep + phase)
                       ).astype(np.float32)

            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            if np.any(np.isnan(d.qpos)) or np.any(np.isnan(d.qvel)):
                nan_resets += 1
                mujoco.mj_resetData(m, d)
                d.qpos[2] = rng.uniform(0.3, 1.5)
                u = rng.standard_normal(4)
                d.qpos[3:7] = u / np.linalg.norm(u)
                mujoco.mj_forward(m, d)
                continue

            sn = mj_to_flat(d.qpos.copy(), d.qvel.copy())
            if (np.any(np.abs(s) > 50) or np.any(np.abs(sn) > 50)
                    or np.any(np.isnan(sn))):
                nan_resets += 1
                mujoco.mj_resetData(m, d)
                d.qpos[2] = rng.uniform(0.3, 1.5)
                u = rng.standard_normal(4)
                d.qpos[3:7] = u / np.linalg.norm(u)
                mujoco.mj_forward(m, d)
                continue

            all_s.append(s)
            all_sn.append(sn)
            all_a.append(tau)
            all_f.append(flags)

        if (ti + 1) % 500 == 0:
            print(f"  gen_data: {ti+1}/{n_traj}, nan={nan_resets}, "
                  f"pairs={len(all_s)}")

    S = np.array(all_s, dtype=np.float32)
    SN = np.array(all_sn, dtype=np.float32)
    A = np.array(all_a, dtype=np.float32)
    F = np.array(all_f, dtype=np.float32)
    print(f"  {len(S):,} pairs (patterns: {pattern_counts}), "
          f"nan_resets={nan_resets}")
    return S, SN, A, F


def benchmark_h1(model_nn, families, device):
    """Single-step benchmark (H=1): measures pure prediction accuracy
    on benchmark states with no compounding."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    results = {}
    for fam, trajs in families.items():
        mses = []
        for ref_s, ref_a in trajs:
            T = len(ref_a)
            errs = []
            for t in range(T):
                flat_to_mj(ref_s[t], m, d)
                mujoco.mj_forward(m, d)
                if np.any(np.isnan(ref_s[t])):
                    break
                s = ref_s[t]
                flags = get_contact_flags(d)
                inp = np.concatenate([s, ref_s[t+1], flags])
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
                    errs.append(100.0)
                else:
                    errs.append(float(np.mean(
                        (actual - ref_s[t+1])**2)))
            if errs:
                mses.append(np.mean(errs))
        results[fam] = float(np.mean(mses)) if mses else 100.0
    results["AGGREGATE"] = float(
        np.mean([results[k] for k in ["uniform", "step", "chirp"]]))
    return results


def benchmark_full(model_nn, families, device, horizon=500):
    """Full closed-loop benchmark."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    results = {}
    for fam, trajs in families.items():
        mses = []
        for ref_s, ref_a in trajs:
            T = min(horizon, len(ref_a))
            flat_to_mj(ref_s[0], m, d)
            errs = []
            for t in range(T):
                if np.any(np.isnan(d.qpos)):
                    errs.append(100.0)
                    break
                s = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if np.any(np.isnan(s)):
                    errs.append(100.0)
                    break
                flags = get_contact_flags(d)
                inp = np.concatenate([s, ref_s[t+1], flags])
                with torch.no_grad():
                    tau = model_nn(
                        torch.tensor(inp, dtype=torch.float32)
                        .unsqueeze(0).to(device)
                    ).cpu().numpy()[0]
                d.ctrl[:] = np.clip(tau, -TAU_MAX_NP, TAU_MAX_NP)
                mujoco.mj_step(m, d)
                actual = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if np.any(np.isnan(actual)):
                    errs.append(100.0)
                    break
                errs.append(float(np.mean(
                    (actual - ref_s[t+1])**2)))
            mses.append(np.mean(errs) if errs else 100.0)
        results[fam] = float(np.mean(mses))
    results["AGGREGATE"] = float(
        np.mean([results[k] for k in ["uniform", "step", "chirp"]]))
    return results


def main():
    print("=" * 60)
    print("HUMANOID COVERAGE EXPERIMENT")
    print("=" * 60)

    families = gen_benchmark(20, 500, seed=99)
    results = {}

    # ── Config 1: Original (uniform-only, 15% scale) ──
    print("\n[Config 1] Uniform-only, 15% scale (original)")
    from scripts.run_humanoid import gen_data as gen_data_orig
    S, SN, A, F = gen_data_orig(2000, 500, seed=42)
    X = np.concatenate([S, SN, F], axis=1)
    in_dim = X.shape[1]

    model = PlainMLP(in_dim, len(TAU_MAX)).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    t0 = time.time()
    train_model(model, X, A, epochs=200, bs=2048, device=device)
    train_t = time.time() - t0
    model.eval()
    h1 = benchmark_h1(model, families, device)
    full = benchmark_full(model, families, device)
    results["uniform_15pct"] = {
        "H1_AGG": h1["AGGREGATE"], "H1": h1,
        "full_AGG": full["AGGREGATE"], "full": full,
        "train_time": train_t, "n_pairs": len(S),
    }
    print(f"  H=1  AGG={h1['AGGREGATE']:.4e}")
    print(f"  H=500 AGG={full['AGGREGATE']:.4e}")

    # ── Config 2: Diverse patterns, 15% scale ──
    print("\n[Config 2] Diverse patterns (uni+step+chirp), 15% scale")
    S, SN, A, F = gen_diverse_data(2000, 500, torque_scale=0.15,
                                    seed=42)
    X = np.concatenate([S, SN, F], axis=1)
    model = PlainMLP(in_dim, len(TAU_MAX)).to(device)
    t0 = time.time()
    train_model(model, X, A, epochs=200, bs=2048, device=device)
    train_t = time.time() - t0
    model.eval()
    h1 = benchmark_h1(model, families, device)
    full = benchmark_full(model, families, device)
    results["diverse_15pct"] = {
        "H1_AGG": h1["AGGREGATE"], "H1": h1,
        "full_AGG": full["AGGREGATE"], "full": full,
        "train_time": train_t, "n_pairs": len(S),
    }
    print(f"  H=1  AGG={h1['AGGREGATE']:.4e}")
    print(f"  H=500 AGG={full['AGGREGATE']:.4e}")

    # ── Config 3: Diverse patterns, 30% scale ──
    print("\n[Config 3] Diverse patterns, 30% scale (more torque)")
    S, SN, A, F = gen_diverse_data(2000, 500, torque_scale=0.30,
                                    seed=42)
    X = np.concatenate([S, SN, F], axis=1)
    model = PlainMLP(in_dim, len(TAU_MAX)).to(device)
    t0 = time.time()
    train_model(model, X, A, epochs=200, bs=2048, device=device)
    train_t = time.time() - t0
    model.eval()
    h1 = benchmark_h1(model, families, device)
    full = benchmark_full(model, families, device)
    results["diverse_30pct"] = {
        "H1_AGG": h1["AGGREGATE"], "H1": h1,
        "full_AGG": full["AGGREGATE"], "full": full,
        "train_time": train_t, "n_pairs": len(S),
    }
    print(f"  H=1  AGG={h1['AGGREGATE']:.4e}")
    print(f"  H=500 AGG={full['AGGREGATE']:.4e}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("COVERAGE EXPERIMENT SUMMARY")
    print("=" * 60)
    for cfg, r in results.items():
        print(f"  {cfg:<20} H1={r['H1_AGG']:.3e}  "
              f"H500={r['full_AGG']:.3e}  "
              f"pairs={r['n_pairs']:,}")

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT}/results.json")


if __name__ == "__main__":
    main()
