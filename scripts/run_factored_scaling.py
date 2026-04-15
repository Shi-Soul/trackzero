#!/usr/bin/env python3
"""Data scaling for factored architecture at high DOF.

Tests whether factored MLP can USE more data at higher DOF (unlike flat
MLP which saturates). If data helps factored but not flat MLP, this
confirms the representation bottleneck has been broken.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.run_factored_scaling --n-links 10
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn

from scripts.models_structured import FactoredMLP, RawMLP
from scripts.run_structured import (
    build_chain_xml, generate_data, generate_benchmark,
    train_model, benchmark_model,
)
from trackzero.config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-links", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    tau_max = cfg.pendulum.tau_max
    n = args.n_links
    nq = n
    sd = 2 * n

    out_dir = f"outputs/factored_scaling_{n}link"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"FACTORED DATA SCALING: {n}-link chain")
    print(f"Data sizes: [2000, 5000, 10000]")
    print(f"Architectures: factored (512×4), raw_mlp (512×4)")
    print("=" * 60)

    xml = build_chain_xml(n)

    # Generate max data upfront (10K), then slice
    print(f"\nGenerating 10000 training trajectories...")
    t0 = time.time()
    S_all, SN_all, A_all = generate_data(xml, 10000, 500, tau_max, seed=42)
    X_all = np.concatenate([S_all, SN_all], axis=1)
    Y_all = A_all
    print(f"  Generated in {time.time()-t0:.1f}s")

    print(f"\nGenerating benchmark...")
    families = generate_benchmark(xml, 50, 500, tau_max)

    data_sizes = [2000, 5000, 10000]
    archs = [("factored", FactoredMLP), ("raw_mlp", RawMLP)]
    all_results = {}

    # Each data size uses a contiguous slice of the full dataset
    # 2K = first 2K trajs × 500 steps = 1M pairs
    # 5K = first 5K trajs × 500 steps = 2.5M pairs
    # 10K = all 10K trajs × 500 steps = 5M pairs
    pairs_per_traj = 500

    for n_data in data_sizes:
        n_pairs = n_data * pairs_per_traj
        X = X_all[:n_pairs]
        Y = Y_all[:n_pairs]

        for arch_name, ArchCls in archs:
            key = f"{arch_name}_{n_data}"
            print(f"\n{'='*60}")
            print(f"{arch_name} with {n_data} trajectories ({n_pairs:,} pairs)")
            print(f"{'='*60}")

            model = ArchCls(sd, nq, hidden=512, layers=4)
            npar = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {npar:,}")

            model, val_loss = train_model(
                model, X, Y, device, epochs=args.epochs, bs=4096
            )

            bench = benchmark_model(model, xml, families, device, tau_max)
            bench["val_loss"] = val_loss
            bench["params"] = npar
            bench["n_data"] = n_data
            all_results[key] = bench

            print(f"  AGG={bench['AGGREGATE']:.4e}  "
                  f"uni={bench['uniform']:.4e}  "
                  f"step={bench['step']:.4e}  "
                  f"chirp={bench['chirp']:.4e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {n}-link data scaling")
    print(f"{'='*60}")
    hdr = f"{'Config':<25s} {'Params':>8s} {'AGG':>10s} {'uniform':>10s} {'step':>10s} {'chirp':>10s}"
    print(hdr)
    for key in sorted(all_results.keys()):
        r = all_results[key]
        print(f"{key:<25s} {r['params']:>8,} {r['AGGREGATE']:>10.4e} "
              f"{r['uniform']:>10.4e} {r['step']:>10.4e} {r['chirp']:>10.4e}")

    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
