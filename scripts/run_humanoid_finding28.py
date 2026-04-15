#!/usr/bin/env python3
"""Finding 28: Does the coverage breakthrough transfer to long-horizon tracking?

The no_bench model achieves H1=50.7 (64× better than random). But
previous replan tests used only random data, which catastrophically fails
at K=5. This test asks: does no_bench + MPC replanning achieve stable
long-horizon tracking?

Designs:
  random_only: 2K traj, white noise (baseline from run_humanoid_replan.py)
  no_bench:    2K traj, diverse excluding step/chirp (Finding 24 winner)
  oracle_matched: 2K traj, step+chirp only (benchmark-matched)

Horizons: K=1,5,10,25,100 steps between replanning corrections.

K=1 = near-oracle MPC (correct every step from actual state)
K=500 = no replanning (baseline, same as H=500 closed-loop eval)

Usage:
    CUDA_VISIBLE_DEVICES=2 python -m scripts.run_humanoid_finding28
"""
import json, os, time
import mujoco
import numpy as np
import torch
import torch.nn as nn

from scripts.run_humanoid import (
    HUMANOID_XML, N_BODY_GEOMS, GEOM_OFFSET, TAU_MAX,
    mj_to_flat, flat_to_mj, get_contact_flags,
    gen_benchmark,
)
from scripts.run_coverage_ablation import gen_data_from_patterns
from scripts.models_structured import _build_mlp
from scripts.run_structured import train_model

FLAT_STATE_DIM = 54
TAU_MAX_NP = np.array(TAU_MAX, dtype=np.float32)
OUT = "outputs/humanoid_finding28"
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training configs
CONFIGS = {
    "random_only": ["white"],
    "no_bench":    ["white", "ou", "brownian", "sine", "bang_bang", "ramp"],
    "oracle_matched": ["step", "chirp"],
}

# MPC horizons to test
HORIZONS = [1, 5, 10, 25, 100, 500]


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


def oracle_replan(m, d, current_state, target_torques, n_steps):
    """Simulate forward from current state using target torques."""
    flat_to_mj(current_state, m, d)
    mujoco.mj_forward(m, d)
    ref_states = [current_state.copy()]
    for t in range(n_steps):
        tau = np.clip(target_torques[t], -TAU_MAX_NP, TAU_MAX_NP)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        s_next = mj_to_flat(d.qpos.copy(), d.qvel.copy())
        if np.any(np.isnan(s_next)):
            for _ in range(n_steps - t):
                ref_states.append(ref_states[-1].copy())
            break
        ref_states.append(s_next)
    return np.array(ref_states)


def rollout_with_replan(model_nn, ref_states_orig, ref_actions_orig, K, dev):
    """Roll out model with MPC-style replanning every K steps.

    At each K-step block: use oracle to compute local reference from
    current actual state, track that reference for K steps, measure
    error against the original reference trajectory.
    """
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    T = len(ref_actions_orig)

    flat_to_mj(ref_states_orig[0], m, d)
    mujoco.mj_forward(m, d)
    current_state = mj_to_flat(d.qpos.copy(), d.qvel.copy())

    total_error = 0.0
    n_valid = 0
    t = 0

    while t < T:
        local_len = min(K, T - t)
        local_torques = ref_actions_orig[t:t + local_len]
        local_ref = oracle_replan(m, d, current_state, local_torques, local_len)

        flat_to_mj(current_state, m, d)
        mujoco.mj_forward(m, d)

        for j in range(local_len):
            if j + 1 >= len(local_ref):
                break
            s_now = mj_to_flat(d.qpos.copy(), d.qvel.copy())
            s_target = local_ref[j + 1]
            flags = get_contact_flags(d)
            inp = np.concatenate([s_now, s_target, flags]).astype(np.float32)

            with torch.no_grad():
                tau = model_nn(
                    torch.tensor(inp).unsqueeze(0).to(dev)
                ).cpu().numpy()[0]

            d.ctrl[:] = np.clip(tau, -TAU_MAX_NP, TAU_MAX_NP)
            mujoco.mj_step(m, d)
            s_actual = mj_to_flat(d.qpos.copy(), d.qvel.copy())

            if np.any(np.isnan(s_actual)):
                return float('inf')

            orig_idx = t + j + 1
            if orig_idx < len(ref_states_orig):
                err = np.mean((s_actual - ref_states_orig[orig_idx]) ** 2)
                total_error += err
                n_valid += 1

        t += local_len
        current_state = mj_to_flat(d.qpos.copy(), d.qvel.copy())

    return total_error / max(n_valid, 1)


def eval_h1(model_nn, families, dev):
    """Single-step benchmark accuracy (54D MSE)."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    errs = []
    for fam, trajs in families.items():
        for ref_states, ref_actions in trajs:
            for t in range(min(50, len(ref_actions))):
                s = ref_states[t]
                sn = ref_states[t + 1]
                tau_ref = ref_actions[t]
                flat_to_mj(s, m, d)
                flags = get_contact_flags(d)
                inp = np.concatenate([s, sn, flags]).astype(np.float32)
                with torch.no_grad():
                    tau_pred = model_nn(
                        torch.tensor(inp).unsqueeze(0).to(dev)
                    ).cpu().numpy()[0]
                flat_to_mj(s, m, d)
                d.ctrl[:] = np.clip(tau_pred, -TAU_MAX_NP, TAU_MAX_NP)
                mujoco.mj_step(m, d)
                s_pred = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                errs.append(np.mean((s_pred - sn) ** 2))
    return float(np.mean(errs))


def train_config(name, patterns):
    print(f"\n{'=' * 50}")
    print(f"Training: {name}  patterns={patterns}")
    t0 = time.time()
    S, SN, A, F = gen_data_from_patterns(patterns, 2000, 500, seed=42)
    print(f"  Data: {len(S):,} pairs in {time.time() - t0:.0f}s")

    in_dim = 2 * FLAT_STATE_DIM + N_BODY_GEOMS
    X = np.concatenate([S, SN, F], axis=1)
    model = PlainMLP(in_dim, len(TAU_MAX)).to(device)
    train_model(model, X, A, epochs=200, bs=2048, device=device)
    model.eval()
    return model


def main():
    print("=" * 60)
    print("FINDING 28: Coverage breakthrough → Long-horizon tracking?")
    print("=" * 60)

    families = gen_benchmark(10, 500, seed=99)
    results = {}

    for cfg_name, patterns in CONFIGS.items():
        model = train_config(cfg_name, patterns)
        h1 = eval_h1(model, families, device)
        print(f"\n  {cfg_name} H1_AGG={h1:.4e}")

        horizon_results = {}
        for K in HORIZONS:
            fam_errs = {}
            for fam in ["uniform", "step", "chirp"]:
                errs = []
                for ref_states, ref_actions in families[fam]:
                    e = rollout_with_replan(
                        model, ref_states, ref_actions, K, device)
                    errs.append(e if e != float('inf') else 1e15)
                fam_errs[fam] = float(np.mean(errs))
            agg = float(np.mean(list(fam_errs.values())))
            horizon_results[f"K={K}"] = {"agg": agg, **fam_errs}
            print(f"    K={K:4d}: agg={agg:.4e}  "
                  f"uniform={fam_errs['uniform']:.3e}  "
                  f"step={fam_errs['step']:.3e}  "
                  f"chirp={fam_errs['chirp']:.3e}")

        results[cfg_name] = {"H1_AGG": h1, "horizons": horizon_results}

    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT}/results.json")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: H1 and K=1,5,25 AGG per config")
    print(f"{'Config':<18s}  {'H1':>10s}  {'K=1':>12s}  {'K=5':>12s}  {'K=25':>12s}")
    for name, r in results.items():
        k1 = r["horizons"].get("K=1", {}).get("agg", float('nan'))
        k5 = r["horizons"].get("K=5", {}).get("agg", float('nan'))
        k25 = r["horizons"].get("K=25", {}).get("agg", float('nan'))
        print(f"{name:<18s}  {r['H1_AGG']:>10.3e}  {k1:>12.3e}  {k5:>12.3e}  {k25:>12.3e}")


if __name__ == "__main__":
    main()
