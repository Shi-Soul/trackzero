#!/usr/bin/env python3
"""Hierarchical factored architecture for tree-structured bodies.

Finding 15 shows global factored HURTS on tree-structured bodies.
Hypothesis: applying factored decomposition within each kinematic
subtree (limb) while keeping subtrees independent should recover the
factored advantage without the cross-limb penalty.

Tests on 3D walker (6 actuated DOF = 2 legs × 3 joints):
  1. raw_mlp (baseline from run_walker3d)
  2. global_factored (global A matrix — known to be worse)
  3. limb_factored (separate A per leg, shared state input)
  4. limb_factored_contact (+ per-link contact flags)

Usage:
    CUDA_VISIBLE_DEVICES=4 python -m scripts.run_hierarchical
"""
import argparse, json, os, time
import mujoco
import numpy as np
import torch
import torch.nn as nn

from scripts.models_structured import _build_mlp
from scripts.run_structured import train_model
from scripts.run_walker3d import (
    WALKER3D_XML, mj_to_flat, flat_to_mj, get_perlink_flags,
    gen_data, gen_benchmark, N_CONTACT_BODIES,
)


class LimbFactoredMLP(nn.Module):
    """Hierarchical factored: separate gain matrix per kinematic subtree.

    For a walker with 2 legs (3 joints each):
      τ_left  = A_left(q,v)  @ [Δq;Δv] + b_left(q,v)
      τ_right = A_right(q,v) @ [Δq;Δv] + b_right(q,v)

    Each limb has its own gain and bias networks, but all see the full
    state. The key difference from global factored: A is block-diagonal
    (no cross-limb coupling terms), which matches the physics.
    """
    def __init__(self, state_dim, limb_dofs, n_contact=0,
                 hidden=512, layers=4):
        """
        Args:
            state_dim: physical state dimension (e.g., 24 for walker)
            limb_dofs: list of ints, actuated DOF per limb (e.g., [3, 3])
            n_contact: number of contact flag dimensions (0 = no flags)
        """
        super().__init__()
        self.sd = state_dim
        self.limb_dofs = limb_dofs
        self.nu = sum(limb_dofs)
        self.n_contact = n_contact
        cond_dim = state_dim + n_contact

        self.limb_gain_nets = nn.ModuleList()
        self.limb_bias_nets = nn.ModuleList()
        for nd in limb_dofs:
            self.limb_gain_nets.append(
                _build_mlp(cond_dim, nd * state_dim, hidden, layers))
            self.limb_bias_nets.append(
                _build_mlp(cond_dim, nd, hidden, layers))

        full_in = 2 * state_dim + n_contact
        self.register_buffer("mu", torch.zeros(full_in))
        self.register_buffer("sigma", torch.ones(full_in))

    def set_norm(self, m, s):
        self.mu.copy_(torch.from_numpy(m).float())
        self.sigma.copy_(torch.from_numpy(s).float())

    def forward(self, x):
        sd = self.sd
        x_norm = (x - self.mu) / self.sigma
        state_raw = x[:, :sd]
        target_raw = x[:, sd:2*sd]
        state_n = x_norm[:, :sd]

        if self.n_contact > 0:
            c_n = x_norm[:, 2*sd:]
            cond = torch.cat([state_n, c_n], dim=1)
        else:
            cond = state_n

        error = target_raw - state_raw
        taus = []
        for i, nd in enumerate(self.limb_dofs):
            A = self.limb_gain_nets[i](cond).reshape(-1, nd, sd)
            b = self.limb_bias_nets[i](cond)
            tau_limb = torch.bmm(A, error.unsqueeze(-1)).squeeze(-1) + b
            taus.append(tau_limb)
        return torch.cat(taus, dim=1)


def benchmark_hierarchical(model_nn, families, device, tau_max,
                           contact_mode):
    """Closed-loop benchmark."""
    m = mujoco.MjModel.from_xml_string(WALKER3D_XML)
    d = mujoco.MjData(m)
    results = {}

    for fam, trajs in families.items():
        mses = []
        for ref_s, ref_a in trajs:
            T = len(ref_a)
            flat_to_mj(ref_s[0], m, d)
            errs = []
            for t in range(T):
                s = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if contact_mode == "none":
                    inp = np.concatenate([s, ref_s[t + 1]])
                else:
                    flags = get_perlink_flags(d)
                    inp = np.concatenate([s, ref_s[t + 1], flags])

                with torch.no_grad():
                    tau = model_nn(
                        torch.tensor(inp, dtype=torch.float32)
                        .unsqueeze(0).to(device)
                    ).cpu().numpy()[0]
                d.ctrl[:] = np.clip(tau, -tau_max, tau_max)
                mujoco.mj_step(m, d)
                actual = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                errs.append(float(np.mean((actual - ref_s[t + 1]) ** 2)))
            mses.append(np.mean(errs))
        results[fam] = float(np.mean(mses))
    results["AGGREGATE"] = float(
        np.mean([results[k] for k in ["uniform", "step", "chirp"]]))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-traj", type=int, default=2000)
    parser.add_argument("--traj-len", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tau_max = 30.0
    sd = 24   # flat state dim
    nu = 6    # total actuated DOF
    limb_dofs = [3, 3]  # left leg, right leg

    out_dir = "outputs/hierarchical_walker"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("HIERARCHICAL FACTORED: Walker3D")
    print(f"  state_dim={sd}, limbs={limb_dofs}")
    print("=" * 60)

    # Generate data
    print("\nGenerating training data...")
    t0 = time.time()
    S, SN, A, F = gen_data(args.n_traj, args.traj_len, tau_max)
    print(f"  Time: {time.time()-t0:.0f}s")

    print("Generating benchmark...")
    bench_fams = gen_benchmark(20, args.traj_len, tau_max)

    configs = [
        ("limb_factored", "none", 0),
        ("limb_factored_contact", "perlink", N_CONTACT_BODIES),
    ]

    all_results = {}
    for label, contact_mode, nc in configs:
        print(f"\n{'='*60}")
        print(f"Config: {label}")
        print(f"{'='*60}")

        if contact_mode == "none":
            X = np.concatenate([S, SN], axis=1)
        else:
            X = np.concatenate([S, SN, F], axis=1)
        Y = A

        nn_model = LimbFactoredMLP(sd, limb_dofs, n_contact=nc,
                                   hidden=512, layers=4)
        npar = sum(p.numel() for p in nn_model.parameters())
        print(f"  Parameters: {npar:,}")

        t0 = time.time()
        nn_model, val_loss = train_model(nn_model, X, Y, device,
                                         epochs=args.epochs)
        train_time = time.time() - t0

        bench = benchmark_hierarchical(
            nn_model, bench_fams, device, tau_max, contact_mode)
        bench["val_loss"] = val_loss
        bench["params"] = npar
        bench["train_time"] = train_time
        all_results[label] = bench

        print(f"  AGG={bench['AGGREGATE']:.4e}  "
              f"uni={bench['uniform']:.4e}  "
              f"step={bench['step']:.4e}  "
              f"chirp={bench['chirp']:.4e}")

    # Summary with comparison to prior results
    print(f"\n{'='*60}")
    print("HIERARCHICAL vs GLOBAL COMPARISON")
    print(f"{'='*60}")
    prior = {"raw_mlp": 0.7946, "global_factored": 0.9183,
             "global_factored_contact": 0.9050}
    for name, agg in prior.items():
        print(f"  {name:<25s} AGG={agg:.4e} (prior)")
    for name, r in all_results.items():
        print(f"  {name:<25s} AGG={r['AGGREGATE']:.4e}")

    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
