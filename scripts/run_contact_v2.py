#!/usr/bin/env python3
"""Contact feature richness ablation (Stage 3B extension).

Tests whether richer contact information improves inverse dynamics.

Conditions:
1. binary_flags:  15D per-link binary contact flags (current)
2. qfrc_only:     27D joint-space constraint forces (from mj_forward)
3. flags_qfrc:    15D + 27D = 42D (both)
4. no_contact:    no contact info (ablation)

Uses no_bench data (6 patterns, scale=0.15, 2K traj, 200ep).
Budget: 1 GPU, ~90 min total.

Usage:
    CUDA_VISIBLE_DEVICES=3 python -m scripts.run_contact_v2
"""
import json, os, time
import numpy as np
import torch
import torch.nn as nn
import mujoco

from scripts.run_humanoid import (
    HUMANOID_XML, N_BODY_GEOMS, TAU_MAX,
    mj_to_flat, flat_to_mj, get_contact_flags,
    gen_benchmark,
)
from scripts.models_structured import _build_mlp
from scripts.run_structured import train_model

FLAT_DIM = 54
TAU_MAX_NP = np.array(TAU_MAX, dtype=np.float32)
NO_BENCH = ["white", "ou", "brownian", "sine", "bang_bang", "ramp"]

OUT = "outputs/contact_v2"
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data generation with physics features ────────────────────────────

def gen_with_physics(n_traj, traj_len, seed=42):
    """Generate data extracting all physics features at each step."""
    rng = np.random.RandomState(seed)
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    nu = m.nu

    all_S, all_SN, all_A = [], [], []
    all_flags, all_qfrc = [], []
    nan_resets = 0

    for i in range(n_traj):
        if i % 500 == 0:
            print(f"  gen: {i}/{n_traj}")
        pat = NO_BENCH[i % len(NO_BENCH)]
        scale = 0.15

        if pat == "white":
            torques = rng.uniform(-1, 1, (traj_len, nu)) * TAU_MAX_NP * scale
        elif pat == "ou":
            tau = np.zeros(nu)
            torques = np.zeros((traj_len, nu))
            for t in range(traj_len):
                tau += 0.15 * (-tau) * dt + 0.3 * np.sqrt(dt) * rng.randn(nu)
                torques[t] = np.clip(tau * TAU_MAX_NP, -TAU_MAX_NP * scale,
                                     TAU_MAX_NP * scale)
        elif pat == "brownian":
            raw = np.cumsum(rng.randn(traj_len, nu) * 0.01, axis=0)
            torques = np.clip(raw * TAU_MAX_NP, -TAU_MAX_NP * scale,
                              TAU_MAX_NP * scale)
        elif pat == "sine":
            freq = rng.uniform(0.1, 10.0, nu)
            phase = rng.uniform(0, 2*np.pi, nu)
            amp = rng.uniform(0.3, 1.0, nu) * TAU_MAX_NP * scale
            t_arr = np.arange(traj_len)[:, None] * dt
            torques = amp * np.sin(2*np.pi * freq * t_arr + phase)
        elif pat == "bang_bang":
            torques = np.zeros((traj_len, nu))
            for j in range(nu):
                sw = sorted(rng.randint(0, traj_len, rng.randint(3, 10)))
                sw = [0] + list(sw) + [traj_len]
                for k in range(len(sw)-1):
                    torques[sw[k]:sw[k+1], j] = (
                        rng.choice([-1, 1]) * TAU_MAX_NP[j] * scale)
        else:  # ramp
            start = rng.uniform(-1, 1, nu) * TAU_MAX_NP * scale
            end = rng.uniform(-1, 1, nu) * TAU_MAX_NP * scale
            alpha = np.linspace(0, 1, traj_len)[:, None]
            torques = start + alpha * (end - start)

        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)

        for t in range(traj_len):
            s = mj_to_flat(d.qpos.copy(), d.qvel.copy())
            flags = get_contact_flags(d)
            # Extract physics features BEFORE applying control
            qfrc = d.qfrc_constraint.copy().astype(np.float32)  # 27D

            tau = np.clip(torques[t], -TAU_MAX_NP, TAU_MAX_NP)
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
            all_flags.append(flags)
            all_qfrc.append(qfrc)

    S = np.array(all_S, dtype=np.float32)
    SN = np.array(all_SN, dtype=np.float32)
    A = np.array(all_A, dtype=np.float32)
    F = np.array(all_flags, dtype=np.float32)
    Q = np.array(all_qfrc, dtype=np.float32)
    print(f"  {len(S):,} pairs, nan_resets={nan_resets}")
    return S, SN, A, F, Q


# ── Model ────────────────────────────────────────────────────────────

class PlainMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=1024, layers=4):
        super().__init__()
        self.net = _build_mlp(in_dim, out_dim, hidden, layers)
        self.register_buffer("norm_m", torch.zeros(in_dim))
        self.register_buffer("norm_s", torch.ones(in_dim))

    def set_norm(self, m, s):
        self.norm_m = torch.tensor(m, dtype=torch.float32).to(
            self.norm_m.device)
        self.norm_s = torch.tensor(s, dtype=torch.float32).to(
            self.norm_s.device)

    def forward(self, x):
        return self.net((x - self.norm_m) / (self.norm_s + 1e-8))


# ── Evaluation ───────────────────────────────────────────────────────

def eval_h1(model_nn, families, dev, contact_mode):
    """H=1 eval with different contact feature configurations."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    fam_errors = {}
    for fam, trajs in families.items():
        errors = []
        for ref_s, ref_a in trajs:
            step_errs = []
            for t in range(min(500, len(ref_a))):
                if np.any(np.isnan(ref_s[t])) or np.any(np.isnan(ref_s[t+1])):
                    break
                flat_to_mj(ref_s[t], m, d)
                mujoco.mj_forward(m, d)
                flags = get_contact_flags(d)
                qfrc = d.qfrc_constraint.copy().astype(np.float32)

                # Build input based on contact mode
                base = np.concatenate([ref_s[t], ref_s[t+1]])  # 108D
                if contact_mode == "binary_flags":
                    inp = np.concatenate([base, flags])
                elif contact_mode == "qfrc_only":
                    inp = np.concatenate([base, qfrc])
                elif contact_mode == "flags_qfrc":
                    inp = np.concatenate([base, flags, qfrc])
                else:  # no_contact
                    inp = base

                with torch.no_grad():
                    tau = model_nn(torch.tensor(inp, dtype=torch.float32)
                                   .unsqueeze(0).to(dev)).cpu().numpy()[0]
                flat_to_mj(ref_s[t], m, d)
                d.ctrl[:] = np.clip(tau, -TAU_MAX_NP, TAU_MAX_NP)
                mujoco.mj_step(m, d)
                actual = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if np.any(np.isnan(actual)):
                    step_errs.append(100.0)
                else:
                    step_errs.append(float(
                        np.mean((actual - ref_s[t+1])**2)))
            if step_errs:
                errors.append(float(np.mean(step_errs)))
        fam_errors[fam] = float(np.mean(errors)) if errors else 1e6
    return fam_errors


# ── Conditions ───────────────────────────────────────────────────────

CONDS = {
    "no_contact": {
        "in_dim": 2 * FLAT_DIM,  # 108
        "build_x": lambda S, SN, F, Q: np.concatenate([S, SN], axis=1),
        "mode": "no_contact",
    },
    "binary_flags": {
        "in_dim": 2 * FLAT_DIM + N_BODY_GEOMS,  # 123
        "build_x": lambda S, SN, F, Q: np.concatenate([S, SN, F], axis=1),
        "mode": "binary_flags",
    },
    "qfrc_only": {
        "in_dim": 2 * FLAT_DIM + 27,  # 135
        "build_x": lambda S, SN, F, Q: np.concatenate([S, SN, Q], axis=1),
        "mode": "qfrc_only",
    },
    "flags_qfrc": {
        "in_dim": 2 * FLAT_DIM + N_BODY_GEOMS + 27,  # 150
        "build_x": lambda S, SN, F, Q: np.concatenate(
            [S, SN, F, Q], axis=1),
        "mode": "flags_qfrc",
    },
}


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CONTACT FEATURE RICHNESS ABLATION")
    print("=" * 60)

    families = gen_benchmark(10, 500, seed=99)
    results = {}
    t_total = time.time()

    # Generate data once (with all features)
    print("\nGenerating 2000 trajectories with physics features...")
    t0 = time.time()
    S, SN, A, F, Q = gen_with_physics(2000, 500, seed=42)
    print(f"Done in {time.time()-t0:.0f}s")

    for name, cfg in CONDS.items():
        print(f"\n--- {name} (in_dim={cfg['in_dim']}) ---")
        t0 = time.time()
        X = cfg["build_x"](S, SN, F, Q)
        model = PlainMLP(cfg["in_dim"], len(TAU_MAX)).to(device)
        model, vl = train_model(model, X, A, device, epochs=200)

        print(f"  Evaluating H=1...")
        h1 = eval_h1(model, families, device, cfg["mode"])
        h1_agg = float(np.mean(list(h1.values())))
        print(f"  H1: {h1} → AGG={h1_agg:.1f}")

        results[name] = {
            "h1": h1, "h1_agg": h1_agg,
            "val_loss": vl, "in_dim": cfg["in_dim"],
            "time": time.time() - t0}
        print(f"  Done in {time.time()-t0:.0f}s")

    total_time = time.time() - t_total
    print(f"\nTOTAL: {total_time:.0f}s ({total_time/60:.1f}min)")
    print("\nSummary:")
    print(f"{'Config':<20} {'H1_AGG':>10} {'val_loss':>10} {'in_dim':>8}")
    for name, r in results.items():
        print(f"{name:<20} {r['h1_agg']:>10.1f} {r['val_loss']:>10.5f} "
              f"{r['in_dim']:>8}")

    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT}/results.json")


if __name__ == "__main__":
    main()
