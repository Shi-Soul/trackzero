#!/usr/bin/env python3
"""Humanoid data scaling: does more data help at 21 DOF?

At 2-10 DOF, data saturates at 2K trajectories (Finding 3, 10).
Hypothesis: at 21 DOF (54D state), the space is under-covered and
more data WILL help. This tests 2K vs 5K vs 10K trajectories.

Also measures H=1 (single-step) accuracy at different data sizes
to isolate coverage from compounding.

Usage:
    CUDA_VISIBLE_DEVICES=2 python -m scripts.run_humanoid_datascale
"""
import json, os, time
import mujoco
import numpy as np
import torch
import torch.nn as nn

from scripts.run_humanoid import (
    HUMANOID_XML, N_BODY_GEOMS, GEOM_OFFSET, TAU_MAX,
    mj_to_flat, flat_to_mj, get_contact_flags,
    gen_data, gen_benchmark,
)
from scripts.models_structured import _build_mlp
from scripts.run_structured import train_model

FLAT_STATE_DIM = 54
TAU_MAX_NP = np.array(TAU_MAX, dtype=np.float32)
OUT = "outputs/humanoid_datascale"
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


def benchmark_h1(model_nn, families, device):
    """Single-step benchmark (H=1): pure prediction accuracy."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    results = {}
    for fam, trajs in families.items():
        mses = []
        for ref_s, ref_a in trajs:
            T = len(ref_a)
            errs = []
            for t in range(T):
                if np.any(np.isnan(ref_s[t])):
                    break
                flat_to_mj(ref_s[t], m, d)
                mujoco.mj_forward(m, d)
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
    print("HUMANOID DATA SCALING EXPERIMENT")
    print("=" * 60)

    families = gen_benchmark(20, 500, seed=99)
    in_dim = 2 * FLAT_STATE_DIM + N_BODY_GEOMS
    results = {}

    for n_traj in [2000, 5000, 10000]:
        print(f"\n{'='*60}")
        print(f"N_TRAJ = {n_traj}")
        print(f"{'='*60}")

        t0 = time.time()
        S, SN, A, F = gen_data(n_traj, 500, seed=42)
        gen_time = time.time() - t0
        print(f"  Data: {len(S):,} pairs in {gen_time:.0f}s")

        X = np.concatenate([S, SN, F], axis=1)

        model = PlainMLP(in_dim, len(TAU_MAX)).to(device)
        t0 = time.time()
        train_model(model, X, A, epochs=200, bs=2048, device=device)
        train_time = time.time() - t0
        print(f"  Train: {train_time:.0f}s, "
              f"val={model.net[0].weight.data.abs().mean():.4f}")  # dummy

        model.eval()
        h1 = benchmark_h1(model, families, device)
        full = benchmark_full(model, families, device)
        print(f"  H=1  AGG={h1['AGGREGATE']:.4e}  "
              f"(u={h1['uniform']:.3e} s={h1['step']:.3e} "
              f"c={h1['chirp']:.3e})")
        print(f"  H=500 AGG={full['AGGREGATE']:.4e}")

        results[f"traj_{n_traj}"] = {
            "n_traj": n_traj,
            "n_pairs": len(S),
            "gen_time": gen_time,
            "train_time": train_time,
            "H1": h1,
            "H1_AGG": h1["AGGREGATE"],
            "full": full,
            "full_AGG": full["AGGREGATE"],
        }

    # ── Summary ──
    print("\n" + "=" * 60)
    print("DATA SCALING SUMMARY")
    print("=" * 60)
    for cfg, r in results.items():
        print(f"  {cfg}: pairs={r['n_pairs']:>9,}  "
              f"H1={r['H1_AGG']:.3e}  "
              f"H500={r['full_AGG']:.3e}")

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT}/results.json")


if __name__ == "__main__":
    main()
