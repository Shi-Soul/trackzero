#!/usr/bin/env python3
"""Ensemble disagreement for targeted data collection (Stage 1C).

Tests whether ensemble uncertainty can identify where the model fails
and guide data collection to those regions, WITHOUT knowing the
benchmark distribution.

Protocol:
  1. Train ensemble of 5 models on random data
  2. Identify high-disagreement states from random rollouts
  3. Collect MORE data near high-disagreement states
  4. Retrain on augmented dataset
  5. Compare against random baseline and oracle-matched

Usage:
    CUDA_VISIBLE_DEVICES=X .venv/bin/python -m scripts.run_ensemble_disagree
"""
import json, os, time
import numpy as np
import mujoco
import torch
import torch.nn as nn

from scripts.run_humanoid import (
    HUMANOID_XML, N_BODY_GEOMS, TAU_MAX,
    mj_to_flat, flat_to_mj, get_contact_flags,
    gen_benchmark,
)
from scripts.models_structured import _build_mlp
from scripts.run_structured import train_model

FLAT_STATE_DIM = 54
TAU_MAX_NP = np.array(TAU_MAX, dtype=np.float32)
OUT = "outputs/ensemble_disagree"
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


def gen_random_data(n_traj, traj_len, seed=42):
    """Generate random-torque training data."""
    rng = np.random.RandomState(seed)
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    nu = m.nu
    all_S, all_SN, all_A, all_F = [], [], [], []
    nan_resets = 0

    for i in range(n_traj):
        if i % 500 == 0 and i > 0:
            print(f"    gen_random: {i}/{n_traj}")
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        for t in range(traj_len):
            s = mj_to_flat(d.qpos.copy(), d.qvel.copy())
            flags = get_contact_flags(d)
            tau = (rng.uniform(-1, 1, nu) * TAU_MAX_NP * 0.15
                   ).astype(np.float32)
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

    print(f"    {len(all_S):,} pairs, nan_resets={nan_resets}")
    return (np.array(all_S, dtype=np.float32),
            np.array(all_SN, dtype=np.float32),
            np.array(all_A, dtype=np.float32),
            np.array(all_F, dtype=np.float32))


def train_ensemble(X, A, n_models=5, dev=None):
    """Train an ensemble of models with different seeds."""
    in_dim = X.shape[1]
    out_dim = A.shape[1]
    models = []
    for i in range(n_models):
        print(f"  Training ensemble member {i+1}/{n_models}...")
        torch.manual_seed(i * 1000)
        np.random.seed(i * 1000)
        model = PlainMLP(in_dim, out_dim, hidden=512, layers=3).to(dev)
        train_model(model, X, A, dev, epochs=100, bs=2048)
        model.eval()
        models.append(model)
    return models


def compute_disagreement(models, X_probe, dev):
    """Compute ensemble disagreement on probe states."""
    preds = []
    with torch.no_grad():
        X_t = torch.tensor(X_probe, dtype=torch.float32).to(dev)
        for model in models:
            pred = model(X_t).cpu().numpy()
            preds.append(pred)
    preds = np.array(preds)  # (n_models, n_samples, out_dim)
    # Disagreement = variance across ensemble members
    var = np.var(preds, axis=0)  # (n_samples, out_dim)
    return np.mean(var, axis=1)  # (n_samples,) mean across dims


def collect_near_states(high_disagree_S, n_traj, traj_len, seed=99):
    """Collect data starting from high-disagreement states."""
    rng = np.random.RandomState(seed)
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    nu = m.nu
    nq, nv = m.nq, m.nv
    all_S, all_SN, all_A, all_F = [], [], [], []
    nan_resets = 0

    for i in range(n_traj):
        if i % 200 == 0 and i > 0:
            print(f"    targeted: {i}/{n_traj}")
        # Pick a high-disagreement state as initial condition
        idx = rng.randint(len(high_disagree_S))
        init_s = high_disagree_S[idx]
        # Set sim to this state (+ small perturbation)
        flat_to_mj(init_s, m, d)
        d.qvel[:] += rng.randn(nv) * 0.1
        mujoco.mj_forward(m, d)

        for t in range(traj_len):
            s = mj_to_flat(d.qpos.copy(), d.qvel.copy())
            flags = get_contact_flags(d)
            tau = (rng.uniform(-1, 1, nu) * TAU_MAX_NP * 0.15
                   ).astype(np.float32)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            sn = mj_to_flat(d.qpos.copy(), d.qvel.copy())
            if np.any(np.isnan(sn)) or np.any(np.isnan(s)):
                nan_resets += 1
                mujoco.mj_resetData(m, d)
                mujoco.mj_forward(m, d)
                break
            all_S.append(s)
            all_SN.append(sn)
            all_A.append(tau)
            all_F.append(flags)

    print(f"    {len(all_S):,} targeted pairs, nan_resets={nan_resets}")
    return (np.array(all_S, dtype=np.float32),
            np.array(all_SN, dtype=np.float32),
            np.array(all_A, dtype=np.float32),
            np.array(all_F, dtype=np.float32))


def eval_h1(model_nn, families, dev):
    """Evaluate H=1 benchmark."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    fam_errors = {}
    for fam, trajs in families.items():
        errors = []
        for ref_s, ref_a in trajs:
            se = []
            for t in range(min(500, len(ref_a))):
                if np.any(np.isnan(ref_s[t])) or \
                   np.any(np.isnan(ref_s[t + 1])):
                    break
                flat_to_mj(ref_s[t], m, d)
                mujoco.mj_forward(m, d)
                flags = get_contact_flags(d)
                inp = np.concatenate([ref_s[t], ref_s[t + 1], flags])
                with torch.no_grad():
                    tau = model_nn(
                        torch.tensor(inp, dtype=torch.float32)
                        .unsqueeze(0).to(dev)).cpu().numpy()[0]
                flat_to_mj(ref_s[t], m, d)
                d.ctrl[:] = np.clip(tau, -TAU_MAX_NP, TAU_MAX_NP)
                mujoco.mj_step(m, d)
                actual = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if np.any(np.isnan(actual)):
                    se.append(100.0)
                else:
                    se.append(float(np.mean(
                        (actual - ref_s[t + 1]) ** 2)))
            if se:
                errors.append(float(np.mean(se)))
        fam_errors[fam] = float(np.mean(errors)) if errors else 1e6
    return fam_errors


def train_final_eval(X, A, label, families, dev):
    """Train a full-size model and evaluate."""
    in_dim = X.shape[1]
    out_dim = A.shape[1]
    model = PlainMLP(in_dim, out_dim, hidden=1024, layers=4).to(dev)
    print(f"  Training {label} ({X.shape[0]:,} pairs)...")
    train_model(model, X, A, dev, epochs=200, bs=2048)
    model.eval()
    h1 = eval_h1(model, families, dev)
    h1_agg = float(np.mean(list(h1.values())))
    print(f"  {label}: H1_AGG={h1_agg:.4e}")
    return {"H1": h1, "H1_AGG": h1_agg, "pairs": int(X.shape[0])}


def main():
    print("=" * 60)
    print("ENSEMBLE DISAGREEMENT (Stage 1C)")
    print("=" * 60)
    t_start = time.time()

    # Step 1: Generate base random data
    print("\n[Step 1] Generate base random data (2K traj)...")
    S, SN, A, F = gen_random_data(2000, 500, seed=42)
    X_base = np.concatenate([S, SN, F], axis=1)

    # Step 2: Train ensemble
    print("\n[Step 2] Train ensemble (5 models, 512×3)...")
    ensemble = train_ensemble(X_base, A, n_models=5, dev=device)

    # Step 3: Compute disagreement on base data
    print("\n[Step 3] Compute disagreement...")
    disagree = compute_disagreement(ensemble, X_base, device)
    print(f"  Disagreement stats: mean={disagree.mean():.4e}, "
          f"max={disagree.max():.4e}, "
          f"p95={np.percentile(disagree, 95):.4e}")

    # Find top-10% high-disagreement states
    thresh = np.percentile(disagree, 90)
    high_idx = np.where(disagree > thresh)[0]
    high_S = S[high_idx]
    print(f"  {len(high_S):,} high-disagreement states (>{thresh:.4e})")

    # Step 4: Collect targeted data near high-disagreement states
    print("\n[Step 4] Collect targeted data (1K traj from "
          "high-disagreement states)...")
    S_t, SN_t, A_t, F_t = collect_near_states(
        high_S, 1000, 100, seed=99)

    # Step 5: Evaluate
    families = gen_benchmark(20, 500, seed=99)
    results = {}

    # Baseline: random-only
    print("\n[Step 5a] Eval baseline (random-only)...")
    results["random_only"] = train_final_eval(
        X_base, A, "random_only", families, device)

    # Ensemble-augmented: base + targeted
    S_aug = np.concatenate([S, S_t])
    SN_aug = np.concatenate([SN, SN_t])
    A_aug = np.concatenate([A, A_t])
    F_aug = np.concatenate([F, F_t])
    X_aug = np.concatenate([S_aug, SN_aug, F_aug], axis=1)

    print("\n[Step 5b] Eval ensemble-augmented...")
    results["ensemble_augmented"] = train_final_eval(
        X_aug, A_aug, "ensemble_aug", families, device)

    # Summary
    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"ENSEMBLE DISAGREEMENT RESULTS ({elapsed / 60:.1f} min)")
    print(f"{'=' * 60}")
    for name, r in results.items():
        print(f"  {name:>25}: H1_AGG={r['H1_AGG']:.4e} "
              f"({r['pairs']:,} pairs)")

    print("\nReference:")
    print("  random_only (oracle_train): H1=3,278")
    print("  diverse_random (entropy):   H1=69.6")
    print("  mixed (oracle):             H1=62")

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT}/results.json")


if __name__ == "__main__":
    main()
