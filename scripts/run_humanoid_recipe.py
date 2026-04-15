#!/usr/bin/env python3
"""Humanoid capacity + training sweep with PROVEN eval.

Uses eval_h1 from run_humanoid_entropy (known correct).
Tests: model sizes × training epochs with diverse data.

Usage:
    CUDA_VISIBLE_DEVICES=X .venv/bin/python -m scripts.run_humanoid_recipe
"""
import json, os, time
import numpy as np
import torch
import torch.nn as nn

from scripts.run_humanoid import gen_benchmark
from scripts.run_humanoid_entropy import (
    gen_diverse_random, PlainMLP, eval_h1, eval_h500,
    FLAT_STATE_DIM, TAU_MAX_NP, N_BODY_GEOMS,
)
from scripts.run_structured import train_model

OUT = "outputs/humanoid_recipe"
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NU = len(TAU_MAX_NP)


def main():
    print("=" * 60)
    print("HUMANOID RECIPE SWEEP (capacity × epochs)")
    print("=" * 60)

    families = gen_benchmark(20, 500, seed=99)

    # Generate diverse data once
    print("\nGenerating diverse data (2K traj, 8 patterns)...")
    t0 = time.time()
    S, SN, A, F = gen_diverse_random(2000, 500, seed=42)
    print(f"Generated {len(S):,} pairs in {time.time()-t0:.0f}s")

    in_dim = 2 * FLAT_STATE_DIM + N_BODY_GEOMS
    X = np.concatenate([S, SN, F], axis=1)
    results = {}

    configs = [
        # (hidden, layers, epochs, label)
        (512, 4, 200, "512x4_200ep"),
        (1024, 4, 200, "1024x4_200ep"),
        (2048, 4, 200, "2048x4_200ep"),
        (1024, 4, 400, "1024x4_400ep"),
        (1024, 6, 200, "1024x6_200ep"),
    ]

    for hidden, layers, epochs, label in configs:
        print(f"\n[{label}]")
        model = PlainMLP(in_dim, NU, hidden=hidden, layers=layers).to(device)
        nparams = sum(p.numel() for p in model.parameters())
        print(f"  Params: {nparams/1e6:.1f}M")
        t0 = time.time()
        train_model(model, X, A, device, epochs=epochs, bs=2048)
        train_time = time.time() - t0
        model.eval()

        h1 = eval_h1(model, families, device)
        h1_agg = float(np.mean(list(h1.values())))
        print(f"  {label}: H1={h1_agg:.4e} ({train_time:.0f}s)")

        results[label] = {
            "H1": h1, "H1_AGG": h1_agg,
            "params": nparams, "hidden": hidden,
            "layers": layers, "epochs": epochs,
            "train_time_s": train_time,
        }

    # Summary
    print("\n" + "=" * 60)
    print("RECIPE SWEEP RESULTS")
    print("=" * 60)
    for key, r in results.items():
        print(f"  {key:>20}: H1={r['H1_AGG']:.4e} "
              f"({r['params']/1e6:.1f}M, {r['train_time_s']:.0f}s)")

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT}/results.json")


if __name__ == "__main__":
    main()
