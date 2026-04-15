#!/usr/bin/env python3
"""Test raw MLP + per-link contact on walker3D.

Missing comparison: we know raw_mlp beats factored on walker (0.795 vs
0.918), but does adding contact help the raw_mlp?

Usage:
    CUDA_VISIBLE_DEVICES=7 python -m scripts.run_rawmlp_contact
"""
import json, os, time
import numpy as np
import torch
import torch.nn as nn

from scripts.run_walker3d import (
    WALKER3D_XML, mj_to_flat, flat_to_mj, get_perlink_flags,
    gen_data, gen_benchmark, N_CONTACT_BODIES,
    benchmark_walker,
)
from scripts.models_structured import RawMLP
from scripts.run_structured import train_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = 24  # walker state dim
    nu = 6   # walker action dim
    out_dir = "outputs/rawmlp_contact_walker"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("RAW MLP + CONTACT on WALKER3D")
    print("=" * 60)

    print("\nGenerating data...")
    t0 = time.time()
    tau_max = 30.0
    S, SN, A, F = gen_data(2000, 500, tau_max)
    print(f"  Time: {time.time()-t0:.0f}s")

    print("Generating benchmark refs...")
    bench_fams = gen_benchmark(20, 500, tau_max)

    # Input for raw_mlp_contact: [state, target, flags] = 24+24+7 = 55
    contact_in = 2 * sd + N_CONTACT_BODIES
    raw_contact_net = nn.Sequential(
        nn.Linear(contact_in, 1024), nn.ELU(),
        nn.Linear(1024, 1024), nn.ELU(),
        nn.Linear(1024, 1024), nn.ELU(),
        nn.Linear(1024, nu),
    )

    class PlainMLP(nn.Module):
        def __init__(self, net, in_dim):
            super().__init__()
            self.net = net
            self.register_buffer("mu", torch.zeros(in_dim))
            self.register_buffer("sigma", torch.ones(in_dim))
        def set_norm(self, m, s):
            self.mu.copy_(torch.from_numpy(m).float())
            self.sigma.copy_(torch.from_numpy(s).float())
        def forward(self, x):
            return self.net((x - self.mu) / self.sigma)

    configs = [
        ("raw_mlp", "none", RawMLP(sd, nu, hidden=1024, layers=4)),
        ("raw_mlp_contact", "perlink",
         PlainMLP(raw_contact_net, contact_in)),
    ]

    all_results = {}
    for label, contact_mode, model in configs:
        print(f"\n{'='*60}")
        print(f"Config: {label}")
        print(f"{'='*60}")

        if contact_mode == "none":
            X = np.concatenate([S, SN], axis=1)
        else:
            X = np.concatenate([S, SN, F], axis=1)
        Y = A

        npar = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {npar:,}")

        t0 = time.time()
        model, val_loss = train_model(model, X, Y, device, epochs=200)
        train_time = time.time() - t0

        bench_r = benchmark_walker(model, bench_fams, device,
                                   tau_max, contact_mode)
        bench_r["val_loss"] = val_loss
        bench_r["params"] = npar
        bench_r["train_time"] = train_time
        all_results[label] = bench_r

        print(f"  AGG={bench_r['AGGREGATE']:.4e}  time={train_time:.0f}s")

    print(f"\n{'='*60}")
    print("RAW MLP + CONTACT SUMMARY")
    print(f"{'='*60}")
    for name, r in all_results.items():
        print(f"  {name:<25s} AGG={r['AGGREGATE']:.4e}  "
              f"params={r['params']:,}")
    print("\nPrior results (from walker3d):")
    print("  raw_mlp (prior)           AGG=7.9460e-01")
    print("  factored (prior)          AGG=9.1830e-01")
    print("  factored+contact (prior)  AGG=9.0500e-01")

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_dir}/results.json")


if __name__ == "__main__":
    main()
