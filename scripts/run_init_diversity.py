#!/usr/bin/env python3
"""Finding 32: Initial state diversity — does matching benchmark init help?

HYPOTHESIS: The benchmark generates trajectories from random initial
states (random height, quaternion, joint angles, velocities). Training
data always starts from the default standing pose. This distribution
shift on initial states may be a major source of H1 error.

Conditions:
  standing:    all trajectories start from default standing (current)
  random_init: trajectories start from random states matching benchmark
  mixed_init:  50% standing + 50% random init

Usage:
    CUDA_VISIBLE_DEVICES=4 python -m scripts.run_init_diversity
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
from scripts.run_humanoid_entropy import (
    gen_diverse_random, PlainMLP, eval_h1, eval_h500,
    FLAT_STATE_DIM,
)
from scripts.run_structured import train_model

TAU_MAX_NP = np.array(TAU_MAX, dtype=np.float32)
N_JOINTS = 21

OUT = "outputs/init_diversity"
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gen_diverse_random_init(n_traj, traj_len, seed=42, scale=0.15):
    """Same as gen_diverse_random but with random initial states."""
    rng = np.random.RandomState(seed)
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    nu = m.nu
    patterns = ["white", "ou", "brownian", "sine",
                "step", "chirp", "bang_bang", "ramp"]

    all_S, all_SN, all_A, all_F = [], [], [], []

    for i in range(n_traj):
        if i % 500 == 0 and i > 0:
            print(f"  random_init: {i}/{n_traj}")
        pat = patterns[i % len(patterns)]

        # Random initial state (matching benchmark distribution)
        mujoco.mj_resetData(m, d)
        d.qpos[2] = rng.uniform(0.6, 1.2)     # height
        u = rng.randn(4)
        d.qpos[3:7] = u / np.linalg.norm(u)   # random quaternion
        d.qpos[7:] = rng.uniform(-0.4, 0.1, m.nq - 7)  # joint angles
        d.qvel[:] = rng.uniform(-1, 1, m.nv)   # random velocities
        mujoco.mj_forward(m, d)

        # Generate torque sequence
        torques = _gen_pattern(pat, traj_len, nu, dt, scale, rng)

        for t in range(traj_len):
            s_t = mj_to_flat(d.qpos.copy(), d.qvel.copy())
            flags = get_contact_flags(d)
            d.ctrl[:] = torques[t]
            mujoco.mj_step(m, d)
            s_next = mj_to_flat(d.qpos.copy(), d.qvel.copy())

            if np.any(np.isnan(s_next)):
                mujoco.mj_resetData(m, d)
                d.qpos[2] = rng.uniform(0.6, 1.2)
                u = rng.randn(4)
                d.qpos[3:7] = u / np.linalg.norm(u)
                d.qpos[7:] = rng.uniform(-0.4, 0.1, m.nq - 7)
                d.qvel[:] = rng.uniform(-1, 1, m.nv)
                mujoco.mj_forward(m, d)
                continue

            all_S.append(s_t)
            all_SN.append(s_next)
            all_A.append(torques[t])
            all_F.append(flags)

    S = np.array(all_S, dtype=np.float32)
    SN = np.array(all_SN, dtype=np.float32)
    A = np.array(all_A, dtype=np.float32)
    F = np.array(all_F, dtype=np.float32)
    print(f"  {len(S):,} pairs")
    return S, SN, A, F


def _gen_pattern(pat, traj_len, nu, dt, scale, rng):
    """Generate torque pattern."""
    if pat == "white":
        return rng.uniform(-1, 1, (traj_len, nu)) * TAU_MAX_NP * scale
    elif pat == "ou":
        tau = np.zeros(nu)
        torques = np.zeros((traj_len, nu))
        for t in range(traj_len):
            tau += 0.15 * (-tau) * dt + \
                   0.3 * np.sqrt(dt) * rng.randn(nu)
            torques[t] = np.clip(
                tau * TAU_MAX_NP, -TAU_MAX_NP * scale,
                TAU_MAX_NP * scale)
        return torques
    elif pat == "brownian":
        inc = rng.randn(traj_len, nu) * 0.01
        raw = np.cumsum(inc, axis=0)
        return np.clip(raw * TAU_MAX_NP, -TAU_MAX_NP * scale,
                       TAU_MAX_NP * scale)
    elif pat == "sine":
        freq = rng.uniform(0.1, 10.0, nu)
        phase = rng.uniform(0, 2 * np.pi, nu)
        amp = rng.uniform(0.3, 1.0, nu) * TAU_MAX_NP * scale
        ts = np.arange(traj_len)[:, None] * dt
        return amp * np.sin(2 * np.pi * freq * ts + phase)
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
        ts = np.arange(traj_len) * dt
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
            period = rng.randint(10, 100)
            sign = 1.0
            for t in range(traj_len):
                if t % period == 0:
                    sign *= -1
                torques[t, j] = sign * TAU_MAX_NP[j] * scale
        return torques
    else:  # ramp
        torques = np.zeros((traj_len, nu))
        n_seg = rng.randint(2, 6)
        seg_len = traj_len // n_seg
        for seg in range(n_seg):
            sa = rng.uniform(-1, 1, nu) * TAU_MAX_NP * scale
            ea = rng.uniform(-1, 1, nu) * TAU_MAX_NP * scale
            s, e = seg * seg_len, min((seg + 1) * seg_len, traj_len)
            for t in range(s, e):
                alpha = (t - s) / max(e - s - 1, 1)
                torques[t] = sa * (1 - alpha) + ea * alpha
        return torques


def train_and_eval(S, SN, A, F, label, families, device):
    """Train model and evaluate."""
    X = np.concatenate([S, SN, F], axis=1)
    in_dim = X.shape[1]
    model = PlainMLP(in_dim, N_JOINTS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}, in_dim={in_dim}")

    t_start = time.time()
    model, val = train_model(model, X, A, device,
                             epochs=200, bs=4096)
    model.eval()

    print("  Evaluating H=1...")
    h1 = eval_h1(model, families, device)
    h1_agg = float(np.mean(list(h1.values())))
    print(f"  H1: {h1} → AGG={h1_agg:.1f}")

    print("  Evaluating H=500...")
    h500 = eval_h500(model, families, device)
    h500_agg = float(np.mean(list(h500.values())))
    print(f"  H500: {h500} → AGG={h500_agg:.2e}")

    elapsed = time.time() - t_start
    return {
        "h1": h1, "h1_agg": h1_agg,
        "h500": h500, "h500_agg": h500_agg,
        "val_loss": float(val),
        "params": n_params,
        "n_pairs": len(S),
        "elapsed_s": elapsed,
    }


def main():
    print("=" * 60)
    print("Finding 32: Initial state diversity")
    print("=" * 60)

    families = gen_benchmark(10, 500, seed=99)
    results = {}

    # Condition 1: standing init (baseline)
    print("\n--- standing (baseline) ---")
    S, SN, A, F = gen_diverse_random(2000, 500, seed=42)
    print(f"  Data: {len(S):,} pairs")
    results["standing"] = train_and_eval(
        S, SN, A, F, "standing", families, device)

    # Condition 2: random init
    print("\n--- random_init ---")
    S2, SN2, A2, F2 = gen_diverse_random_init(
        2000, 500, seed=42, scale=0.15)
    results["random_init"] = train_and_eval(
        S2, SN2, A2, F2, "random_init", families, device)

    # Condition 3: mixed (1000 standing + 1000 random)
    print("\n--- mixed_init ---")
    S_s, SN_s, A_s, F_s = gen_diverse_random(1000, 500, seed=42)
    S_r, SN_r, A_r, F_r = gen_diverse_random_init(
        1000, 500, seed=43, scale=0.15)
    S3 = np.concatenate([S_s, S_r])
    SN3 = np.concatenate([SN_s, SN_r])
    A3 = np.concatenate([A_s, A_r])
    F3 = np.concatenate([F_s, F_r])
    # Shuffle
    idx = np.random.RandomState(99).permutation(len(S3))
    S3, SN3, A3, F3 = S3[idx], SN3[idx], A3[idx], F3[idx]
    results["mixed_init"] = train_and_eval(
        S3, SN3, A3, F3, "mixed_init", families, device)

    # Save & summary
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("SUMMARY: Initial state diversity")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name:15s}: H1={r['h1_agg']:6.1f}  "
              f"H500={r['h500_agg']:10.1f}  "
              f"val={r['val_loss']:.1f}  "
              f"pairs={r['n_pairs']:,}")


if __name__ == "__main__":
    main()
