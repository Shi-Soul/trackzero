#!/usr/bin/env python3
"""State-augmented training for humanoid.

Instead of only training on states from random-torque rollouts, we
augment the training set with states sampled by adding noise to
existing training states. For each noisy state, we compute the oracle
action (simulate one step with the target next-state from the original
pair). This expands coverage without requiring a working policy.

Three augmentation strategies:
1. gaussian: Add N(0, σ) to each state dimension
2. interpolate: Lerp between random pairs of training states
3. boundary: Push states toward extreme joint limits / velocities

Usage:
    CUDA_VISIBLE_DEVICES=5 python -m scripts.run_humanoid_augment
"""
import json, os, time
import mujoco
import numpy as np
import torch
import torch.nn as nn

from scripts.run_humanoid import (
    HUMANOID_XML, N_BODY_GEOMS, TAU_MAX,
    mj_to_flat, flat_to_mj, get_contact_flags,
    gen_data, gen_benchmark,
)
from scripts.models_structured import _build_mlp
from scripts.run_structured import train_model

FLAT_STATE_DIM = 54
TAU_MAX_NP = np.array(TAU_MAX, dtype=np.float32)
OUT = "outputs/humanoid_augment"
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


def augment_gaussian(S, SN, A, F, n_aug, sigma_frac=0.5, seed=123):
    """Add Gaussian noise to training states and recompute oracle.

    For each augmented state s', we apply random action a from original
    data, simulate forward, get s'_next. This gives valid (s', a, s'_next)
    triples even though s' was never visited by a rollout.
    """
    rng = np.random.RandomState(seed)
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)

    # Compute per-dim std from training data
    state_std = np.std(S, axis=0) + 1e-8
    sigma = state_std * sigma_frac

    aug_S, aug_SN, aug_A, aug_F = [], [], [], []
    n_valid = 0
    n_tried = 0

    while n_valid < n_aug and n_tried < n_aug * 5:
        idx = rng.randint(len(S))
        noise = rng.randn(FLAT_STATE_DIM) * sigma
        s_noisy = S[idx] + noise

        # Set noisy state and apply original action
        flat_to_mj(s_noisy, m, d)
        mujoco.mj_forward(m, d)
        flags = get_contact_flags(d)

        # Apply same action
        a = A[idx].copy()
        d.ctrl[:] = np.clip(a, -TAU_MAX_NP, TAU_MAX_NP)
        mujoco.mj_step(m, d)
        s_next = mj_to_flat(d.qpos.copy(), d.qvel.copy())
        n_tried += 1

        if np.any(np.isnan(s_next)):
            continue

        aug_S.append(s_noisy)
        aug_SN.append(s_next)
        aug_A.append(a)
        aug_F.append(flags)
        n_valid += 1

    print(f"    gaussian: {n_valid}/{n_tried} valid "
          f"({100*n_valid/max(n_tried,1):.0f}%)")
    return (np.array(aug_S, dtype=np.float32),
            np.array(aug_SN, dtype=np.float32),
            np.array(aug_A, dtype=np.float32),
            np.array(aug_F, dtype=np.float32))


def augment_interpolate(S, SN, A, F, n_aug, seed=456):
    """Interpolate between random pairs of training states."""
    rng = np.random.RandomState(seed)
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)

    aug_S, aug_SN, aug_A, aug_F = [], [], [], []
    n_valid = 0
    n_tried = 0

    while n_valid < n_aug and n_tried < n_aug * 5:
        i, j = rng.randint(len(S), size=2)
        alpha = rng.uniform(0.1, 0.9)
        s_interp = S[i] * alpha + S[j] * (1 - alpha)
        a_interp = A[i] * alpha + A[j] * (1 - alpha)

        flat_to_mj(s_interp, m, d)
        mujoco.mj_forward(m, d)
        flags = get_contact_flags(d)

        d.ctrl[:] = np.clip(a_interp, -TAU_MAX_NP, TAU_MAX_NP)
        mujoco.mj_step(m, d)
        s_next = mj_to_flat(d.qpos.copy(), d.qvel.copy())
        n_tried += 1

        if np.any(np.isnan(s_next)):
            continue

        aug_S.append(s_interp)
        aug_SN.append(s_next)
        aug_A.append(a_interp)
        aug_F.append(flags)
        n_valid += 1

    print(f"    interpolate: {n_valid}/{n_tried} valid "
          f"({100*n_valid/max(n_tried,1):.0f}%)")
    return (np.array(aug_S, dtype=np.float32),
            np.array(aug_SN, dtype=np.float32),
            np.array(aug_A, dtype=np.float32),
            np.array(aug_F, dtype=np.float32))


def eval_h1(model_nn, families, device):
    """Evaluate H=1 (single-step) benchmark."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    fam_errors = {}
    for fam, trajs in families.items():
        errors = []
        for ref_s, ref_a in trajs:
            step_errs = []
            for t in range(min(500, len(ref_a))):
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
                    step_errs.append(100.0)
                else:
                    step_errs.append(float(np.mean(
                        (actual - ref_s[t+1])**2)))
            if step_errs:
                errors.append(float(np.mean(step_errs)))
        fam_errors[fam] = float(np.mean(errors)) if errors else 1e6
    return fam_errors


def train_and_eval(S, SN, A, F, label, families, device):
    """Train model on given data and evaluate."""
    in_dim = 2 * FLAT_STATE_DIM + N_BODY_GEOMS
    X = np.concatenate([S, SN, F], axis=1)
    model = PlainMLP(in_dim, len(TAU_MAX)).to(device)
    print(f"  Training {label} ({len(S):,} pairs)...")
    train_model(model, X, A, device, epochs=200, bs=2048)
    model.eval()

    h1 = eval_h1(model, families, device)
    h1_agg = float(np.mean(list(h1.values())))
    print(f"  {label}: H=1 AGG={h1_agg:.4e} "
          f"(u={h1.get('uniform',0):.2e} "
          f"s={h1.get('step',0):.2e} "
          f"c={h1.get('chirp',0):.2e})")
    return h1_agg, h1


def main():
    print("=" * 60)
    print("HUMANOID STATE-AUGMENTED TRAINING")
    print("=" * 60)

    # Generate base data
    print("\n[1] Generating base training data (2K traj)...")
    t0 = time.time()
    S, SN, A, F = gen_data(2000, 500, seed=42)
    n_base = len(S)
    print(f"    {n_base:,} pairs in {time.time()-t0:.0f}s")

    # Generate benchmark
    print("\n[2] Generating benchmark...")
    families = gen_benchmark(20, 500, seed=99)

    # Baseline: no augmentation
    print("\n[3] Baseline (no augmentation)...")
    base_agg, _ = train_and_eval(S, SN, A, F, "baseline",
                                  families, device)

    # Gaussian augmentation (50% of base size)
    n_aug = n_base // 2
    print(f"\n[4] Gaussian augmentation ({n_aug:,} new pairs)...")
    gS, gSN, gA, gF = augment_gaussian(S, SN, A, F, n_aug)
    allS = np.concatenate([S, gS])
    allSN = np.concatenate([SN, gSN])
    allA = np.concatenate([A, gA])
    allF = np.concatenate([F, gF])
    gauss_agg, _ = train_and_eval(allS, allSN, allA, allF,
                                   "gaussian_0.5σ", families, device)

    # Interpolation augmentation
    print(f"\n[5] Interpolation augmentation ({n_aug:,} new pairs)...")
    iS, iSN, iA, iF = augment_interpolate(S, SN, A, F, n_aug)
    allS = np.concatenate([S, iS])
    allSN = np.concatenate([SN, iSN])
    allA = np.concatenate([A, iA])
    allF = np.concatenate([F, iF])
    interp_agg, _ = train_and_eval(allS, allSN, allA, allF,
                                    "interpolate", families, device)

    # Combined: gaussian + interpolation
    print(f"\n[6] Combined augmentation...")
    allS = np.concatenate([S, gS, iS])
    allSN = np.concatenate([SN, gSN, iSN])
    allA = np.concatenate([A, gA, iA])
    allF = np.concatenate([F, gF, iF])
    combined_agg, _ = train_and_eval(allS, allSN, allA, allF,
                                      "combined", families, device)

    # Summary
    print("\n" + "=" * 60)
    print("AUGMENTATION SUMMARY")
    print("=" * 60)
    results = {
        "baseline": {"agg": base_agg, "pairs": n_base},
        "gaussian_0.5σ": {"agg": gauss_agg,
                          "pairs": n_base + len(gS)},
        "interpolate": {"agg": interp_agg,
                        "pairs": n_base + len(iS)},
        "combined": {"agg": combined_agg,
                     "pairs": n_base + len(gS) + len(iS)},
    }
    for name, r in results.items():
        ratio = base_agg / r["agg"] if r["agg"] > 0 else float('inf')
        print(f"  {name:>15}: H1_AGG={r['agg']:.4e} "
              f"({ratio:.2f}× vs baseline) pairs={r['pairs']:,}")

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT}/results.json")


if __name__ == "__main__":
    main()
