#!/usr/bin/env python3
"""Per-link contact flags: richer contact representation.

Finding 6 showed that a single binary contact flag is insufficient at
3+ DOF. Hypothesis: per-link flags (which specific links are in contact)
provide enough information to handle contact at higher DOF.

Compares three contact representations with factored architecture:
  1. No flag (baseline) — factored model ignores contact
  2. Scalar flag — single binary "any contact" flag
  3. Per-link flags — N-dim binary vector (one per link)

Usage:
    CUDA_VISIBLE_DEVICES=4 python -m scripts.run_perlink_contact --n-links 5
"""
import argparse, json, os, time
import mujoco
import numpy as np
import torch

from trackzero.config import Config
from scripts.models_structured import FactoredMLP, ContactFactoredMLP
from scripts.run_structured import train_model
from scripts.run_factored_contact import (
    build_chain_contact_xml, gen_contact_refs,
)


def gen_perlink_data(xml, n_traj, traj_len, tau_max, seed=42):
    """Random rollout data with per-link contact flags."""
    rng = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    nq, nv, nu = model.nq, model.nv, model.nu
    n_links = nq

    # Map geom_id -> link_index (0-based)
    # geom 0 = floor, geom i = link i (1-indexed)
    # We want link_flags[k] = 1 if link k+1 is in contact
    link_geom_ids = list(range(1, n_links + 1))

    all_s, all_sn, all_a = [], [], []
    all_scalar, all_perlink = [], []

    for _ in range(n_traj):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = rng.uniform(-1.5, 1.5, size=nq)
        data.qvel[:] = rng.uniform(-5, 5, size=nv)
        mujoco.mj_forward(model, data)

        for _ in range(traj_len):
            s = np.concatenate([data.qpos.copy(), data.qvel.copy()])
            tau = rng.uniform(-tau_max, tau_max, size=nu)

            # Scalar flag
            scalar_c = float(data.ncon > 0)

            # Per-link flags
            link_flags = np.zeros(n_links, dtype=np.float32)
            for ci in range(data.ncon):
                g1 = data.contact[ci].geom1
                g2 = data.contact[ci].geom2
                for g in [g1, g2]:
                    if 1 <= g <= n_links:
                        link_flags[g - 1] = 1.0

            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            sn = np.concatenate([data.qpos.copy(), data.qvel.copy()])

            all_s.append(s)
            all_sn.append(sn)
            all_a.append(tau)
            all_scalar.append(scalar_c)
            all_perlink.append(link_flags)

    S = np.array(all_s, dtype=np.float32)
    SN = np.array(all_sn, dtype=np.float32)
    A = np.array(all_a, dtype=np.float32)
    C_scalar = np.array(all_scalar, dtype=np.float32).reshape(-1, 1)
    C_perlink = np.array(all_perlink, dtype=np.float32)

    cfrac = C_scalar.mean()
    per_frac = C_perlink.mean(axis=0)
    print(f"  {len(S):,} pairs, contact={cfrac:.1%}")
    print(f"  Per-link fractions: {np.array2string(per_frac, precision=2)}")
    return S, SN, A, C_scalar, C_perlink


def benchmark_perlink(model, xml, families, device, tau_max,
                      contact_mode, n_links):
    """Closed-loop benchmark with different contact representations."""
    mj = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(mj)
    nq = mj.nq
    results = {}

    for fam, trajs in families.items():
        mses = []
        for ref_s, ref_a in trajs:
            T = len(ref_a)
            mujoco.mj_resetData(mj, d)
            d.qpos[:] = ref_s[0, :nq]
            d.qvel[:] = ref_s[0, nq:]
            mujoco.mj_forward(mj, d)
            errs = []
            for t in range(T):
                s = np.concatenate([d.qpos.copy(), d.qvel.copy()])

                if contact_mode == "none":
                    inp_parts = [s, ref_s[t + 1]]
                elif contact_mode == "scalar":
                    c = np.array([float(d.ncon > 0)], dtype=np.float32)
                    inp_parts = [s, ref_s[t + 1], c]
                elif contact_mode == "perlink":
                    flags = np.zeros(n_links, dtype=np.float32)
                    for ci in range(d.ncon):
                        g1 = d.contact[ci].geom1
                        g2 = d.contact[ci].geom2
                        for g in [g1, g2]:
                            if 1 <= g <= n_links:
                                flags[g - 1] = 1.0
                    inp_parts = [s, ref_s[t + 1], flags]

                inp = np.concatenate(inp_parts).astype(np.float32)
                with torch.no_grad():
                    tau = model(
                        torch.tensor(inp).unsqueeze(0).to(device)
                    ).cpu().numpy()[0]
                d.ctrl[:] = np.clip(tau, -tau_max, tau_max)
                mujoco.mj_step(mj, d)
                actual = np.concatenate([d.qpos.copy(), d.qvel.copy()])
                errs.append(float(np.mean((actual - ref_s[t+1])**2)))
            mses.append(np.mean(errs))
        results[fam] = float(np.mean(mses))
    results["AGGREGATE"] = float(
        np.mean([results[k] for k in ["uniform", "step", "chirp"]]))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-links", type=int, required=True)
    parser.add_argument("--n-traj", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    tau_max = cfg.pendulum.tau_max
    n = args.n_links
    nq = n
    sd = 2 * n

    out_dir = f"outputs/perlink_contact_{n}link"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"PER-LINK CONTACT: {n}-link chain")
    print("=" * 60)

    xml_c = build_chain_contact_xml(n)

    # Generate data with both scalar and per-link flags
    print("\nGenerating data...")
    S, SN, A, C_scalar, C_perlink = gen_perlink_data(
        xml_c, args.n_traj, 500, tau_max)

    # Generate benchmark
    print("Generating benchmark...")
    bench_fams = gen_contact_refs(xml_c, 50, 500, tau_max)

    configs = [
        ("no_flag", "none", 0),
        ("scalar_flag", "scalar", 1),
        ("perlink_flags", "perlink", n),
    ]

    all_results = {}
    for label, contact_mode, n_contact in configs:
        print(f"\n{'='*60}")
        print(f"Config: {label}")
        print(f"{'='*60}")

        if contact_mode == "none":
            X = np.concatenate([S, SN], axis=1)
            model = FactoredMLP(sd, nq, hidden=512, layers=4)
        elif contact_mode == "scalar":
            X = np.concatenate([S, SN, C_scalar], axis=1)
            model = ContactFactoredMLP(sd, nq, n_contact=1,
                                       hidden=512, layers=4)
        elif contact_mode == "perlink":
            X = np.concatenate([S, SN, C_perlink], axis=1)
            model = ContactFactoredMLP(sd, nq, n_contact=n,
                                       hidden=512, layers=4)
        Y = A

        npar = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {npar:,}")

        model, val_loss = train_model(model, X, Y, device,
                                      epochs=args.epochs)

        bench = benchmark_perlink(
            model, xml_c, bench_fams, device, tau_max,
            contact_mode, n)
        bench["val_loss"] = val_loss
        bench["params"] = npar
        all_results[label] = bench

        print(f"  AGG={bench['AGGREGATE']:.4e}  "
              f"uni={bench['uniform']:.4e}  "
              f"step={bench['step']:.4e}  "
              f"chirp={bench['chirp']:.4e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"PER-LINK CONTACT SUMMARY: {n}-link")
    print(f"{'='*60}")
    for name, r in all_results.items():
        print(f"  {name:<20s} AGG={r['AGGREGATE']:.4e}")

    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
