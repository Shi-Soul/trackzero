#!/usr/bin/env python3
"""Humanoid capacity scaling with diverse-pattern data.

Tests whether model capacity limits humanoid tracking quality
when coverage is controlled (diverse data).

Conditions:
  - 512×4  (~0.6M params, underpowered baseline)
  - 1024×4 (~2.2M params, current standard)
  - 2048×4 (~8.5M params, bigger)
  - 1024×6 (~4.2M params, deeper)

All trained on the same diverse-pattern data (2K traj, 8 patterns).
200 epochs, cosine LR, WD=1e-4.

Usage:
    CUDA_VISIBLE_DEVICES=X .venv/bin/python -m scripts.run_humanoid_capacity
"""
import json, os, time
import numpy as np
import torch
import torch.nn as nn

from scripts.run_humanoid import (
    HUMANOID_XML, mj_to_flat, flat_to_mj, get_contact_flags,
    gen_benchmark,
)
from scripts.run_humanoid_entropy import gen_diverse_random as gen_diverse_data
from scripts.models_structured import _build_mlp
from scripts.run_structured import train_model

OUT = "outputs/humanoid_capacity"
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import mujoco
_m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
TAU_MAX = np.array([_m.actuator_ctrlrange[i, 1]
                    for i in range(_m.nu)], dtype=np.float32)
NU = _m.nu
FLAT_DIM = 54
N_BODY_GEOMS = 15


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


def eval_h1(model_nn, families, dev):
    """H=1 benchmark eval."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    fam_errors = {}
    for fam, trajs in families.items():
        errors = []
        for ref_s, ref_a in trajs:
            se = []
            T = min(500, len(ref_a))
            for t in range(T):
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
                d.ctrl[:] = np.clip(tau, -TAU_MAX, TAU_MAX)
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


def train_eval(S, SN, A, F, hidden, layers, families, dev):
    """Train and evaluate a specific configuration."""
    in_dim = 2 * FLAT_DIM + N_BODY_GEOMS
    X = np.concatenate([S, SN, F], axis=1)
    model = PlainMLP(in_dim, NU, hidden=hidden, layers=layers).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    label = f"{hidden}x{layers} ({nparams/1e6:.1f}M)"
    print(f"  Training {label}...")
    t0 = time.time()
    train_model(model, X, A, dev, epochs=200, bs=2048)
    train_time = time.time() - t0
    model.eval()

    h1 = eval_h1(model, families, dev)
    h1_agg = float(np.mean(list(h1.values())))
    print(f"  {label}: H1={h1_agg:.4e} ({train_time:.0f}s)")
    return {"H1": h1, "H1_AGG": h1_agg,
            "params": nparams, "label": label,
            "train_time_s": train_time}


def main():
    print("=" * 60)
    print("HUMANOID CAPACITY SCALING (diverse data)")
    print("=" * 60)

    families = gen_benchmark(20, 500, seed=99)

    # Generate diverse data once
    print("\nGenerating diverse data (2K traj, 8 patterns)...")
    t0 = time.time()
    S, SN, A, F = gen_diverse_data(2000, 500, seed=42)
    print(f"Generated {len(S):,} pairs in {time.time()-t0:.0f}s")

    configs = [
        (512, 4),
        (1024, 4),
        (2048, 4),
        (1024, 6),
    ]

    results = {}
    for hidden, layers in configs:
        key = f"{hidden}x{layers}"
        print(f"\n[{key}]")
        results[key] = train_eval(
            S, SN, A, F, hidden, layers, families, device)

    # Summary
    print("\n" + "=" * 60)
    print("CAPACITY SCALING RESULTS")
    print("=" * 60)
    for key, r in results.items():
        print(f"  {r['label']:>25}: H1={r['H1_AGG']:.4e} "
              f"({r['train_time_s']:.0f}s)")

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT}/results.json")


if __name__ == "__main__":
    main()
