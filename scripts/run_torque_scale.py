#!/usr/bin/env python3
"""Finding 29: Torque scale and training data quantity ablation.

Tests two key hyperparameters for humanoid data generation:
1. Torque scale: 5%, 15% (current), 30%, 50% of τ_max
   - Higher scale = more aggressive motions = more of state space
   - But also more NaN/instability
2. Data quantity with diverse patterns: 1K, 2K (current), 5K trajectories
   - Does more data help when patterns are already diverse?

All use 1024×4 MLP, 200 epochs, 6-pattern (no_bench) data.
Budget: 1 GPU, ≤ 2h total.

Usage:
    CUDA_VISIBLE_DEVICES=4 python -m scripts.run_torque_scale
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
from scripts.run_humanoid_entropy import eval_h1, eval_h500

FLAT_DIM = 54
TAU_MAX_NP = np.array(TAU_MAX, dtype=np.float32)
OUT = "outputs/torque_scale"
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


def gen_no_bench(n_traj, traj_len, scale, seed=42):
    """Generate diverse data with 6 non-benchmark patterns.

    Uses: white, OU, brownian, sine, bang_bang, ramp
    (excludes step, chirp which are benchmark patterns)
    """
    rng = np.random.RandomState(seed)
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    nu = m.nu

    all_S, all_SN, all_A, all_F = [], [], [], []
    nan_resets = 0
    patterns = ["white", "ou", "brownian", "sine", "bang_bang", "ramp"]

    for i in range(n_traj):
        if i % 500 == 0 and i > 0:
            print(f"    gen_no_bench: {i}/{n_traj} (scale={scale})")

        pat = patterns[i % len(patterns)]

        if pat == "white":
            torques = rng.uniform(-1, 1, (traj_len, nu)) * \
                      TAU_MAX_NP * scale
        elif pat == "ou":
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
            phase = rng.uniform(0, 2 * np.pi, nu)
            amp = rng.uniform(0.3, 1.0, nu) * TAU_MAX_NP * scale
            t_arr = np.arange(traj_len)[:, None] * dt
            torques = amp * np.sin(2 * np.pi * freq * t_arr + phase)
        elif pat == "bang_bang":
            torques = np.zeros((traj_len, nu))
            for j in range(nu):
                switch_times = sorted(
                    rng.randint(0, traj_len, rng.randint(3, 10)))
                switch_times = [0] + list(switch_times) + [traj_len]
                for k in range(len(switch_times) - 1):
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
    print(f"    {len(S):,} pairs, nan_resets={nan_resets}")
    return S, SN, A, F


def run():
    t0 = time.time()
    print("=== Finding 29: Torque Scale + Data Quantity Ablation ===")
    print(f"Device: {device}")

    # Benchmark
    print("\n--- Generating benchmark ---")
    families = gen_benchmark(n_per_fam=10, traj_len=500, seed=99)

    in_dim = 2 * FLAT_DIM + N_BODY_GEOMS  # 123
    n_joints = 21

    configs = [
        # (name, n_traj, scale)
        ("scale_05",  2000, 0.05),
        ("scale_15",  2000, 0.15),   # current default
        ("scale_30",  2000, 0.30),
        ("scale_50",  2000, 0.50),
        ("data_1k",   1000, 0.15),
        ("data_5k",   5000, 0.15),
    ]

    results = {}
    for name, n_traj, scale in configs:
        print(f"\n{'='*60}")
        print(f"  Config: {name} (n_traj={n_traj}, scale={scale})")
        print(f"{'='*60}")
        ct = time.time()

        # Generate data
        S, SN, A, F = gen_no_bench(n_traj, 500, scale, seed=42)
        X = np.concatenate([S, SN, F], axis=1)
        Y = A.copy()
        print(f"  Data: {len(X):,} pairs, input_dim={X.shape[1]}")

        # Create and train model
        model = PlainMLP(in_dim, n_joints, hidden=1024, layers=4)
        n_params = sum(p.numel() for p in model.parameters())
        model, val = train_model(model, X, Y, device)

        # Eval
        print("  Evaluating H=1...")
        h1 = eval_h1(model, families, device)
        h1_agg = float(np.mean(list(h1.values())))
        print(f"  H1: {h1} → AGG={h1_agg:.1f}")

        print("  Evaluating H=500...")
        h500 = eval_h500(model, families, device)
        h500_agg = float(np.mean(list(h500.values())))
        print(f"  H500: {h500} → AGG={h500_agg:.2e}")

        elapsed = time.time() - ct
        results[name] = {
            "h1": h1, "h1_agg": h1_agg,
            "h500": h500, "h500_agg": h500_agg,
            "val_loss": val, "params": n_params,
            "n_traj": n_traj, "scale": scale,
            "n_pairs": len(X), "elapsed_s": elapsed,
        }
        print(f"  Done in {elapsed:.0f}s")

        with open(f"{OUT}/results.json", "w") as f:
            json.dump(results, f, indent=2)

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"TOTAL: {total:.0f}s ({total/60:.1f}min)")
    print(f"\nSummary:")
    print(f"{'Config':<12} {'H1_AGG':>8} {'H500_AGG':>12} {'pairs':>8}")
    for name, r in results.items():
        print(f"{name:<12} {r['h1_agg']:>8.1f} {r['h500_agg']:>12.2e} "
              f"{r['n_pairs']:>8,}")


if __name__ == "__main__":
    run()
