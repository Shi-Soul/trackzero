#!/usr/bin/env python3
"""Analyze per-timestep error on humanoid benchmark trajectories.

Trains one model, then measures H=1 error at each timestep position
within benchmark trajectories. If error increases with t, it confirms
that later states (far from init) are out-of-distribution.

Also measures training data state statistics vs benchmark state
statistics to quantify distribution shift directly.

Usage:
    CUDA_VISIBLE_DEVICES=3 python -m scripts.run_humanoid_timestep
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
OUT = "outputs/humanoid_timestep"
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


def per_timestep_h1(model_nn, families, device, max_t=500):
    """Compute H=1 error at each timestep position."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)

    # Collect errors indexed by timestep
    timestep_errors = {fam: {} for fam in families}

    for fam, trajs in families.items():
        for ref_s, ref_a in trajs:
            T = min(max_t, len(ref_a))
            for t in range(T):
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
                    err = 100.0
                else:
                    err = float(np.mean((actual - ref_s[t+1])**2))
                if t not in timestep_errors[fam]:
                    timestep_errors[fam][t] = []
                timestep_errors[fam][t].append(err)

    # Average per timestep
    result = {}
    for fam in families:
        ts = sorted(timestep_errors[fam].keys())
        means = [float(np.mean(timestep_errors[fam][t])) for t in ts]
        result[fam] = {"timesteps": ts, "mean_errors": means}
    return result


def state_distribution_analysis(train_S, benchmark_families):
    """Compare training vs benchmark state distributions."""
    # Training stats
    train_mean = np.mean(train_S, axis=0)
    train_std = np.std(train_S, axis=0)
    train_min = np.min(train_S, axis=0)
    train_max = np.max(train_S, axis=0)

    # Benchmark stats per family
    bench_stats = {}
    for fam, trajs in benchmark_families.items():
        all_states = []
        for ref_s, ref_a in trajs:
            all_states.append(ref_s)
        all_s = np.concatenate(all_states, axis=0)
        bench_stats[fam] = {
            "mean": np.mean(all_s, axis=0),
            "std": np.std(all_s, axis=0),
            "min": np.min(all_s, axis=0),
            "max": np.max(all_s, axis=0),
            "n_states": len(all_s),
        }

    # Compute distribution distance per dimension
    dim_names = (["pos_x", "pos_y", "pos_z",
                  "rot_x", "rot_y", "rot_z"]
                 + [f"joint_{i}" for i in range(21)]
                 + [f"vel_{i}" for i in range(27)])

    analysis = {"dims": dim_names}
    for fam, bs in bench_stats.items():
        # Normalized distance: |mean_bench - mean_train| / std_train
        dist = np.abs(bs["mean"] - train_mean) / (train_std + 1e-8)
        # Out-of-range fraction: what % of bench states are outside
        # training [min, max]?
        # Approximate using means
        oor = np.maximum(0, bs["mean"] - train_max) + np.maximum(
            0, train_min - bs["mean"])
        analysis[fam] = {
            "mean_dist": float(np.mean(dist)),
            "max_dist": float(np.max(dist)),
            "worst_dim": dim_names[int(np.argmax(dist))],
            "worst_dist": float(np.max(dist)),
            "mean_oor": float(np.mean(oor)),
        }
        print(f"  {fam}: mean_shift={np.mean(dist):.2f}σ, "
              f"worst_dim={dim_names[int(np.argmax(dist))]} "
              f"({np.max(dist):.1f}σ)")

    return analysis


def main():
    print("=" * 60)
    print("HUMANOID PER-TIMESTEP ERROR ANALYSIS")
    print("=" * 60)

    # Generate data
    print("\n[1] Generating training data...")
    S, SN, A, F = gen_data(2000, 500, seed=42)
    print(f"    {len(S):,} pairs")

    # Train model
    in_dim = 2 * FLAT_STATE_DIM + N_BODY_GEOMS
    X = np.concatenate([S, SN, F], axis=1)
    model = PlainMLP(in_dim, len(TAU_MAX)).to(device)
    print(f"\n[2] Training ({sum(p.numel() for p in model.parameters()):,}"
          " params)...")
    train_model(model, X, A, epochs=200, bs=2048, device=device)
    model.eval()

    # Generate benchmark
    print("\n[3] Generating benchmark...")
    families = gen_benchmark(20, 500, seed=99)

    # Per-timestep error
    print("\n[4] Per-timestep H=1 error analysis...")
    ts_results = per_timestep_h1(model, families, device)

    # Print summary at key timesteps
    for fam in ["uniform", "step", "chirp"]:
        data = ts_results[fam]
        ts = data["timesteps"]
        errs = data["mean_errors"]
        print(f"\n  {fam}:")
        for check_t in [0, 10, 50, 100, 200, 400, 499]:
            if check_t in ts:
                idx = ts.index(check_t)
                print(f"    t={check_t:>4}: error={errs[idx]:.4e}")

    # State distribution analysis
    print("\n[5] State distribution shift analysis...")
    dist_analysis = state_distribution_analysis(S, families)

    # Also analyze benchmark states at different timesteps
    print("\n[6] Benchmark state drift over time...")
    for fam in ["uniform", "step", "chirp"]:
        trajs = families[fam]
        early_states = []  # t=0-10
        late_states = []   # t=400-500
        for ref_s, ref_a in trajs:
            early_states.append(ref_s[:11])
            if len(ref_s) > 400:
                late_states.append(ref_s[400:])
        early = np.concatenate(early_states)
        late = np.concatenate(late_states)
        train_std_safe = np.std(S, axis=0) + 1e-8

        early_shift = np.mean(np.abs(
            np.mean(early, axis=0) - np.mean(S, axis=0))
            / train_std_safe)
        late_shift = np.mean(np.abs(
            np.mean(late, axis=0) - np.mean(S, axis=0))
            / train_std_safe)
        print(f"  {fam}: early(0-10)={early_shift:.2f}σ, "
              f"late(400-500)={late_shift:.2f}σ")

    # Save results
    # Convert timestep lists to serializable format
    ts_save = {}
    for fam, data in ts_results.items():
        # Sample every 10 timesteps for compact storage
        ts = data["timesteps"]
        errs = data["mean_errors"]
        sampled = {str(t): e for t, e in zip(ts, errs)
                   if t % 10 == 0 or t < 10}
        ts_save[fam] = sampled

    results = {
        "per_timestep": ts_save,
        "distribution_shift": dist_analysis,
    }
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT}/results.json")


if __name__ == "__main__":
    main()
