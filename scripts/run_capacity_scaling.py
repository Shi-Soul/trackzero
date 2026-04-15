#!/usr/bin/env python3
"""Capacity scaling: test factored architecture at 10-DOF with different sizes.

Tests whether the 10-DOF error (AGG=0.196 with 512×4) is capacity-limited
or inherently hard. If larger models help, we need more capacity.
If not, the bottleneck is elsewhere (optimization, data distribution, etc.).

Usage:
    CUDA_VISIBLE_DEVICES=3 python -m scripts.run_capacity_scaling
"""
import json, os, time
import torch
import numpy as np
from scripts.models_structured import FactoredMLP
from scripts.run_structured import (
    build_chain_xml, generate_data, generate_benchmark,
    train_model, benchmark_model,
)
from trackzero.config import Config


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    tau_max = cfg.pendulum.tau_max
    n = 10
    nq = n
    sd = 2 * n

    out_dir = "outputs/capacity_scaling_10link"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("CAPACITY SCALING: 10-link factored")
    print("=" * 60)

    # Generate data and benchmark once
    xml = build_chain_xml(n)
    print("\nGenerating data (2K trajectories)...")
    S, SN, A = generate_data(xml, 2000, 500, tau_max)
    X = np.concatenate([S, SN], axis=1)
    Y = A

    print("Generating benchmark...")
    bench_fams = generate_benchmark(xml, 50, 500, tau_max)

    configs = [
        (256, 4),
        (512, 4),   # existing baseline for comparison
        (1024, 4),
        (1024, 6),
    ]

    all_results = {}
    for hidden, layers in configs:
        label = f"factored_{hidden}x{layers}"
        print(f"\n{'='*60}")
        print(f"Config: {label}")
        print(f"{'='*60}")

        model = FactoredMLP(sd, nq, hidden=hidden, layers=layers)
        npar = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {npar:,}")

        t0 = time.time()
        model, val_loss = train_model(model, X, Y, device, epochs=200)
        train_time = time.time() - t0

        bench = benchmark_model(model, xml, bench_fams, device, tau_max)
        bench["val_loss"] = val_loss
        bench["params"] = npar
        bench["train_time_s"] = train_time
        all_results[label] = bench

        print(f"  AGG={bench['AGGREGATE']:.4e}  "
              f"params={npar:,}  time={train_time:.0f}s")

    # Summary
    print(f"\n{'='*60}")
    print("CAPACITY SCALING SUMMARY (10-link factored)")
    print(f"{'='*60}")
    print(f"  {'Config':<22s} {'Params':>10s} {'AGG':>12s} {'Time':>8s}")
    for name, r in all_results.items():
        print(f"  {name:<22s} {r['params']:>10,} "
              f"{r['AGGREGATE']:>12.4e} {r['train_time_s']:>7.0f}s")

    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
