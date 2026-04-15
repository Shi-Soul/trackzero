#!/usr/bin/env python3
"""Finding 30: Does TRACK-ZERO generalize to realistic motion amplitudes?

The current benchmark uses scale=0.15 of max torque — very small
perturbations from the initial pose. Real human gait involves joint
excursions of ±30-60°, which require MUCH larger torques. If our model
was trained on small-amplitude data, it may not generalize to real motion.

This tests whether the coverage breakthrough (no_bench) generalizes to
REALISTIC human-scale motion by generating gait-like references with
anatomically plausible amplitudes.

Reference types:
  bench_small: standard benchmark scale (0.15) — baseline
  gait_ref:    sinusoidal gait at realistic amplitudes (0.5-0.8 scale)
               with gait-frequency oscillations (~1-2 Hz)

Usage:
    CUDA_VISIBLE_DEVICES=3 python -m scripts.run_humanoid_finding30
"""
import json, os, time
import mujoco
import numpy as np
import torch
import torch.nn as nn

from scripts.run_humanoid import (
    HUMANOID_XML, N_BODY_GEOMS, GEOM_OFFSET, TAU_MAX,
    mj_to_flat, flat_to_mj, get_contact_flags,
)
from scripts.run_coverage_ablation import gen_data_from_patterns
from scripts.models_structured import _build_mlp
from scripts.run_structured import train_model

FLAT_STATE_DIM = 54
TAU_MAX_NP = np.array(TAU_MAX, dtype=np.float32)
OUT = "outputs/humanoid_finding30"
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIGS = {
    "random_only":    ["white"],
    "no_bench":       ["white", "ou", "brownian", "sine", "bang_bang", "ramp"],
    "oracle_matched": ["step", "chirp"],
}

N_REFS = 20    # trajectories per reference type
TRAJ_LEN = 500


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


def gen_gait_references(n_per_type, traj_len, seed=77):
    """Generate reference trajectories at realistic motion amplitudes.

    bench_small: standard benchmark (scale=0.15), small perturbations
    gait_large: gait-like sinusoidal at 0.5-0.8 scale, gait frequency
    """
    rng = np.random.RandomState(seed)
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    nu = m.nu

    def gen_one(scale, freq_range=(0.5, 2.0)):
        """Generate one trajectory with sinusoidal torques at given scale."""
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)

        # Randomize starting pose slightly (like benchmark)
        d.qpos[2] = rng.uniform(0.6, 1.2)
        u = rng.standard_normal(4)
        d.qpos[3:7] = u / np.linalg.norm(u)
        d.qpos[7:] = rng.uniform(-0.2, 0.1, m.nq - 7)
        d.qvel[:] = rng.uniform(-0.5, 0.5, m.nv)
        mujoco.mj_forward(m, d)

        # Sinusoidal torques at gait-like frequency
        freqs = rng.uniform(*freq_range, nu)
        phases = rng.uniform(0, 2 * np.pi, nu)
        amps = rng.uniform(scale * 0.8, scale * 1.0, nu) * TAU_MAX_NP

        states = [mj_to_flat(d.qpos.copy(), d.qvel.copy())]
        actions = []
        for t in range(traj_len):
            if np.any(np.isnan(d.qpos)):
                break
            tau = (amps * np.sin(2 * np.pi * freqs * t * dt + phases)
                   ).astype(np.float32)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            states.append(mj_to_flat(d.qpos.copy(), d.qvel.copy()))
            actions.append(tau)
        return np.array(states), np.array(actions)

    trajs = {
        "bench_small": [gen_one(0.15, freq_range=(0.1, 2.0))
                        for _ in range(n_per_type)],
        "gait_medium": [gen_one(0.40, freq_range=(0.5, 2.0))
                        for _ in range(n_per_type)],
        "gait_large":  [gen_one(0.75, freq_range=(0.5, 1.5))
                        for _ in range(n_per_type)],
    }
    return trajs


def eval_h1_on_refs(model_nn, refs, dev):
    """Evaluate single-step (H=1) MSE on reference set."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    errs = []
    for ref_type, trajs in refs.items():
        type_errs = []
        for ref_states, ref_actions in trajs:
            T = min(50, len(ref_actions))
            for t in range(T):
                s = ref_states[t]
                sn = ref_states[t + 1]
                flat_to_mj(s, m, d)
                flags = get_contact_flags(d)
                inp = np.concatenate([s, sn, flags]).astype(np.float32)
                with torch.no_grad():
                    tau = model_nn(
                        torch.tensor(inp).unsqueeze(0).to(dev)
                    ).cpu().numpy()[0]
                flat_to_mj(s, m, d)
                d.ctrl[:] = np.clip(tau, -TAU_MAX_NP, TAU_MAX_NP)
                mujoco.mj_step(m, d)
                s_pred = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                type_errs.append(float(np.mean((s_pred - sn) ** 2)))
        mean_err = float(np.mean(type_errs)) if type_errs else float('inf')
        errs.append((ref_type, mean_err))
    return errs


def main():
    print("=" * 60)
    print("FINDING 30: TRACK-ZERO on Realistic Motion Amplitudes")
    print("=" * 60)

    refs = gen_gait_references(N_REFS, TRAJ_LEN)
    print(f"Generated references: {list(refs.keys())}")
    for rname, trajs in refs.items():
        valid = [len(s) - 1 for s, _ in trajs]
        print(f"  {rname}: {len(trajs)} trajs, "
              f"avg_len={np.mean(valid):.0f} steps")

    results = {}

    for cfg_name, patterns in CONFIGS.items():
        print(f"\n{'='*50}")
        print(f"Config: {cfg_name}")
        t0 = time.time()
        S, SN, A, F = gen_data_from_patterns(patterns, 2000, 500, seed=42)
        print(f"  Data: {len(S):,} pairs in {time.time() - t0:.0f}s")

        in_dim = 2 * FLAT_STATE_DIM + N_BODY_GEOMS
        X = np.concatenate([S, SN, F], axis=1)
        model = PlainMLP(in_dim, len(TAU_MAX)).to(device)
        train_model(model, X, A, epochs=200, bs=2048, device=device)
        model.eval()

        errs = eval_h1_on_refs(model, refs, device)
        config_results = {}
        for rtype, e in errs:
            config_results[rtype] = e
            print(f"  {rtype}: H1={e:.4e}")

        results[cfg_name] = config_results

    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT}/results.json")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY: H1 MSE by config and reference amplitude")
    ref_types = ["bench_small", "gait_medium", "gait_large"]
    header = f"{'Config':<18s}" + "".join(f"  {r:>14s}" for r in ref_types)
    print(header)
    for cfg, r in results.items():
        row = f"{cfg:<18s}"
        for rtype in ref_types:
            row += f"  {r.get(rtype, float('nan')):>14.3e}"
        print(row)

    # Key comparison: does gait_large degrade vs bench_small?
    print("\nRatio gait_large / bench_small (>1 = harder):")
    for cfg, r in results.items():
        ratio = r.get("gait_large", 1) / max(r.get("bench_small", 1), 1e-10)
        print(f"  {cfg}: {ratio:.2f}×")


if __name__ == "__main__":
    main()
