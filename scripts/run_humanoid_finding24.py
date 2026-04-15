#!/usr/bin/env python3
"""Finding 24: Coverage ablation — diversity vs pattern matching.

Research question: Does the 47× H=1 improvement of diverse_random
(H1_AGG=69.6) come from:
  (a) General state diversity — visiting states beyond near-standing, OR
  (b) Pattern matching — including step/chirp (which ARE the benchmark)?

Ablation design:
  blind6:    White, OU, brownian, sine, bang_bang, ramp (NO step/chirp)
  eval_like: Only step + chirp (benchmark patterns, 2 of 8)
  baseline:  White noise only (original random-only baseline)
  oracle:    step + chirp + uniform (known matched result for reference)

If blind6 ≈ diverse_random (69.6), the improvement is from general
diversity → TRACK-ZERO can generalize without knowing benchmark patterns.
If blind6 ≈ baseline (3,278), it was pattern matching all along.

Usage:
    CUDA_VISIBLE_DEVICES=X uv run python scripts/run_humanoid_finding24.py
"""

import json, os, time
import mujoco
import numpy as np
import torch
import torch.nn as nn

from scripts.run_humanoid import (
    HUMANOID_XML, N_BODY_GEOMS, TAU_MAX,
    mj_to_flat, get_contact_flags,
    gen_benchmark,
)
from scripts.models_structured import _build_mlp
from scripts.run_structured import train_model

FLAT_STATE_DIM = 54
TAU_MAX_NP = np.array(TAU_MAX, dtype=np.float32)
OUT = "outputs/humanoid_finding24"
os.makedirs(OUT, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_TRAJ = 2000
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


# ---------------------------------------------------------------------------
# Data generators — each selects a subset of torque patterns
# ---------------------------------------------------------------------------

def _gen_data(n_traj, traj_len, patterns, seed=42):
    """Generate training data using the given list of torque patterns."""
    rng = np.random.RandomState(seed)
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    nu = m.nu

    all_S, all_SN, all_A, all_F = [], [], [], []
    nan_resets = 0

    for i in range(n_traj):
        if i % 200 == 0 and i > 0:
            print(f"    {i}/{n_traj} ({len(all_S):,} pairs so far)")

        pat = patterns[i % len(patterns)]
        scale = 0.15

        if pat == "white":
            torques = rng.uniform(-1, 1, (traj_len, nu)) * TAU_MAX_NP * scale
        elif pat == "ou":
            tau_vec = np.zeros(nu)
            torques = np.zeros((traj_len, nu))
            for t in range(traj_len):
                tau_vec += 0.15 * (-tau_vec) * dt + \
                           0.3 * np.sqrt(dt) * rng.randn(nu)
                torques[t] = np.clip(tau_vec * TAU_MAX_NP,
                                     -TAU_MAX_NP * scale, TAU_MAX_NP * scale)
        elif pat == "brownian":
            raw = np.cumsum(rng.randn(traj_len, nu) * 0.01, axis=0)
            torques = np.clip(raw * TAU_MAX_NP,
                              -TAU_MAX_NP * scale, TAU_MAX_NP * scale)
        elif pat == "sine":
            freq = rng.uniform(0.1, 10.0, nu)
            phase = rng.uniform(0, 2 * np.pi, nu)
            amp = rng.uniform(0.3, 1.0, nu) * TAU_MAX_NP * scale
            t_arr = np.arange(traj_len)[:, None] * dt
            torques = amp * np.sin(2 * np.pi * freq * t_arr + phase)
        elif pat == "step":
            n_steps = rng.randint(2, 8)
            boundaries = sorted(rng.randint(0, traj_len, n_steps))
            boundaries = [0] + list(boundaries) + [traj_len]
            torques = np.zeros((traj_len, nu))
            for j in range(len(boundaries) - 1):
                amp = rng.uniform(-1, 1, nu) * TAU_MAX_NP * scale
                torques[boundaries[j]:boundaries[j + 1]] = amp
        elif pat == "chirp":
            t_arr = np.arange(traj_len) * dt
            f0, f1 = rng.uniform(0.1, 2), rng.uniform(3, 15)
            freq = np.linspace(f0, f1, traj_len)
            phase = rng.uniform(0, 2 * np.pi, nu)
            amp = rng.uniform(0.5, 1.0, nu) * TAU_MAX_NP * scale
            torques = np.zeros((traj_len, nu))
            for j in range(nu):
                torques[:, j] = amp[j] * np.sin(
                    2 * np.pi * np.cumsum(freq) * dt + phase[j])
        elif pat == "bang_bang":
            torques = np.zeros((traj_len, nu))
            for j in range(nu):
                sw = sorted(rng.randint(0, traj_len, rng.randint(3, 10)))
                sw = [0] + list(sw) + [traj_len]
                for k in range(len(sw) - 1):
                    torques[sw[k]:sw[k + 1], j] = (
                        rng.choice([-1, 1]) * TAU_MAX_NP[j] * scale
                    )
        elif pat == "ramp":
            start = rng.uniform(-1, 1, nu) * TAU_MAX_NP * scale
            end = rng.uniform(-1, 1, nu) * TAU_MAX_NP * scale
            alpha = np.linspace(0, 1, traj_len)[:, None]
            torques = start + alpha * (end - start)
        else:
            raise ValueError(f"Unknown pattern: {pat}")

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
    print(f"  {len(S):,} pairs, {nan_resets} NaN resets")
    return S, SN, A, F


ABLATIONS = {
    # Full diverse (original) — for sanity check reproducibility
    "diverse_all8": ["white", "ou", "brownian", "sine",
                     "step", "chirp", "bang_bang", "ramp"],
    # Blind exploration: no step or chirp (the benchmark patterns)
    "blind6":       ["white", "ou", "brownian", "sine",
                     "bang_bang", "ramp"],
    # Only benchmark-like patterns
    "eval_like2":   ["step", "chirp"],
    # Original baseline: white noise only
    "white_only":   ["white"],
}


def eval_policy(model, family_trajs, n_steps_horizon):
    """Evaluate model closed-loop at given horizon H.
    
    family_trajs: list of (states_array (T+1, 54), actions_array)
    flat state = [pos3, rotvec3, joints21, qvel27] = 54D
    joint positions are flat[6:27] (21 joints)
    """
    from scripts.run_humanoid import flat_to_mj

    m_mj = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d_mj = mujoco.MjData(m_mj)
    nu = m_mj.nu

    all_mse = []
    for ref_states, _ in family_trajs:
        T = len(ref_states) - 1
        cur_s = ref_states[0].copy()
        flat_to_mj(cur_s, m_mj, d_mj)
        mse_sum = 0.0

        for t in range(T):
            if t % n_steps_horizon == 0:
                # Reset to reference state
                flat_to_mj(ref_states[t], m_mj, d_mj)
                cur_s = mj_to_flat(d_mj.qpos.copy(), d_mj.qvel.copy())

            ref_next = ref_states[t + 1]
            flags = get_contact_flags(d_mj)
            x_in = np.concatenate([cur_s, ref_next, flags]).astype(np.float32)

            with torch.no_grad():
                xt = torch.from_numpy(x_in).unsqueeze(0).to(DEVICE)
                u = model(xt).squeeze(0).cpu().numpy()

            d_mj.ctrl[:] = np.clip(u, -TAU_MAX_NP, TAU_MAX_NP)
            mujoco.mj_step(m_mj, d_mj)
            cur_s = mj_to_flat(d_mj.qpos.copy(), d_mj.qvel.copy())

            # MSE on joint positions (flat[6:27], 21 joints)
            q_err = np.mean((cur_s[6:27] - ref_next[6:27]) ** 2)
            mse_sum += q_err

        all_mse.append(mse_sum / max(T, 1))
    return float(np.mean(all_mse))


def run_ablation(name: str, patterns: list, benchmark: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"Ablation: {name}  patterns={patterns}")
    print(f"{'='*60}")

    t0 = time.time()
    S, SN, A, F = _gen_data(N_TRAJ, TRAJ_LEN, patterns, seed=42)
    gen_time = time.time() - t0
    print(f"  Generated {len(S):,} pairs in {gen_time:.0f}s")

    # Build input: [state, ref_next, contact_flags]
    n_flags = N_BODY_GEOMS
    in_dim = FLAT_STATE_DIM + FLAT_STATE_DIM + n_flags
    out_dim = len(TAU_MAX)
    model = PlainMLP(in_dim, out_dim, hidden=1024, layers=4).to(DEVICE)

    # Input: [s, sn, flags] — use SN as ref_next target (oracle data)
    X = np.concatenate([S, SN, F], axis=1)
    Y = A

    # Normalize
    m_x = X.mean(0).astype(np.float32)
    s_x = np.maximum(X.std(0), 1e-6).astype(np.float32)
    model.set_norm(m_x, s_x)

    t1 = time.time()
    model, val_loss = train_model(model, X, Y, DEVICE, epochs=200, bs=4096)
    train_time = time.time() - t1
    print(f"  Training done in {train_time:.0f}s, val_loss={val_loss:.4f}")

    # Evaluate
    h1_results, h500_results = {}, {}
    for family, trajs in benchmark.items():
        h1_mse = eval_policy(model, trajs, n_steps_horizon=1)
        h500_mse = eval_policy(model, trajs, n_steps_horizon=500)
        h1_results[family] = h1_mse
        h500_results[family] = h500_mse
        print(f"  {family:12s}: H1={h1_mse:.3e}  H500={h500_mse:.3e}")

    h1_agg = float(np.mean(list(h1_results.values())))
    h500_agg = float(np.mean(list(h500_results.values())))
    print(f"  {'AGG':12s}: H1={h1_agg:.3e}  H500={h500_agg:.3e}")

    return {
        "patterns": patterns,
        "pairs": int(len(S)),
        "val_loss": float(val_loss),
        "H1": h1_results,
        "H1_AGG": h1_agg,
        "H500": h500_results,
        "H500_AGG": h500_agg,
    }


def main():
    print("Finding 24: Diversity vs Pattern Matching Ablation")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"N_TRAJ={N_TRAJ}, TRAJ_LEN={TRAJ_LEN}")

    # Generate benchmark once
    print("\nGenerating benchmark trajectories...")
    benchmark = gen_benchmark(n_per_fam=5, traj_len=500, seed=1000)
    print(f"  Benchmark: {list(benchmark.keys())}")

    results = {}
    for name, patterns in ABLATIONS.items():
        res = run_ablation(name, patterns, benchmark)
        results[name] = res
        # Save incrementally
        with open(os.path.join(OUT, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  [{name}] saved incrementally")

    # Summary
    print("\n" + "="*60)
    print("FINDING 24 SUMMARY")
    print("="*60)
    header = f"{'Config':<15s} {'Patterns':>10s} {'Pairs':>8s} {'H1 AGG':>10s} {'H500 AGG':>12s}"
    print(header)
    for name, r in results.items():
        print(f"{name:<15s} {len(r['patterns']):>10d} "
              f"{r['pairs']:>8,} {r['H1_AGG']:>10.3e} {r['H500_AGG']:>12.3e}")

    print(f"\nResults saved to: {OUT}/results.json")

    # Interpretation
    diverse_h1 = results.get("diverse_all8", {}).get("H1_AGG", None)
    blind6_h1 = results.get("blind6", {}).get("H1_AGG", None)
    eval_like_h1 = results.get("eval_like2", {}).get("H1_AGG", None)
    baseline_h1 = results.get("white_only", {}).get("H1_AGG", None)

    if all(x is not None for x in [blind6_h1, eval_like_h1, baseline_h1]):
        print("\nINTERPRETATION:")
        print(f"  blind6 / diverse_all8 ratio = {blind6_h1 / (diverse_h1 or 1):.2f}x")
        print(f"  eval_like2 / diverse_all8 ratio = {eval_like_h1 / (diverse_h1 or 1):.2f}x")
        if blind6_h1 < 3 * (diverse_h1 or 1):
            print("  -> Improvement is from GENERAL DIVERSITY (not pattern matching)")
            print("     TRACK-ZERO with blind exploration achieves humanoid coverage!")
        else:
            print("  -> Improvement is from PATTERN MATCHING (includes benchmark patterns)")
            print("     Need blind exploration strategy for Step 1E completion.")


if __name__ == "__main__":
    main()
