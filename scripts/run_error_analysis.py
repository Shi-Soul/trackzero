#!/usr/bin/env python3
"""Error decomposition: where does H1 error come from?

Breaks down the best model's prediction error by:
1. Position vs velocity (root_pos, root_vel, joint_pos, joint_vel)
2. Per-benchmark family (uniform, step, chirp)
3. Per-timestep (early vs late)
4. Contact vs no-contact states

Uses the optimized recipe: no_bench, scale=0.15, 1024×4, 200ep.
Trains ONE model, then does detailed error analysis.

Usage:
    CUDA_VISIBLE_DEVICES=4 python -m scripts.run_error_analysis
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

OUT = "outputs/error_analysis"
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
            self.norm_m.device)
        self.norm_s = torch.tensor(s, dtype=torch.float32).to(
            self.norm_s.device)

    def forward(self, x):
        return self.net((x - self.norm_m) / (self.norm_s + 1e-8))


def gen_patterned(n_traj, traj_len, patterns, seed=42):
    """Generate data with specified torque patterns."""
    rng = np.random.RandomState(seed)
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    nu = m.nu
    all_S, all_SN, all_A, all_F = [], [], [], []
    nan_resets = 0

    for i in range(n_traj):
        if i % 500 == 0:
            print(f"  gen: {i}/{n_traj}")
        pat = patterns[i % len(patterns)]
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
            all_F.append(flags)

    return (np.array(all_S, dtype=np.float32),
            np.array(all_SN, dtype=np.float32),
            np.array(all_A, dtype=np.float32),
            np.array(all_F, dtype=np.float32))


def detailed_h1_analysis(model_nn, families, dev):
    """Detailed per-step error analysis."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)

    # State decomposition indices
    # flat = [root_pos(3), root_rot(3), joints(21), root_vel(6), joint_vel(21)]
    IDX = {
        "root_pos": slice(0, 3),
        "root_rot": slice(3, 6),
        "joint_pos": slice(6, 27),
        "root_vel": slice(27, 33),
        "joint_vel": slice(33, 54),
    }

    analysis = {}
    for fam, trajs in families.items():
        fam_data = {"total": [], "components": {k: [] for k in IDX},
                    "per_timestep": {}, "contact_vs_free": {"contact": [],
                                                             "free": []}}
        for ref_s, ref_a in trajs:
            T = min(500, len(ref_a))
            for t in range(T):
                if np.any(np.isnan(ref_s[t])) or np.any(np.isnan(ref_s[t+1])):
                    break
                flat_to_mj(ref_s[t], m, d)
                mujoco.mj_forward(m, d)
                flags = get_contact_flags(d)
                has_contact = np.any(flags > 0.5)
                inp = np.concatenate([ref_s[t], ref_s[t+1], flags])
                with torch.no_grad():
                    tau = model_nn(torch.tensor(inp, dtype=torch.float32)
                                   .unsqueeze(0).to(dev)).cpu().numpy()[0]
                flat_to_mj(ref_s[t], m, d)
                d.ctrl[:] = np.clip(tau, -TAU_MAX_NP, TAU_MAX_NP)
                mujoco.mj_step(m, d)
                actual = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if np.any(np.isnan(actual)):
                    continue
                err = (actual - ref_s[t+1])**2
                total_err = float(np.mean(err))
                fam_data["total"].append(total_err)
                for comp, idx in IDX.items():
                    fam_data["components"][comp].append(
                        float(np.mean(err[idx])))
                # Per-timestep buckets
                bucket = t // 50 * 50
                if bucket not in fam_data["per_timestep"]:
                    fam_data["per_timestep"][bucket] = []
                fam_data["per_timestep"][bucket].append(total_err)
                # Contact vs free
                if has_contact:
                    fam_data["contact_vs_free"]["contact"].append(total_err)
                else:
                    fam_data["contact_vs_free"]["free"].append(total_err)

        # Summarize
        summary = {
            "total_mean": float(np.mean(fam_data["total"])) if fam_data["total"] else 0,
            "total_p95": float(np.percentile(fam_data["total"], 95)) if fam_data["total"] else 0,
            "total_p99": float(np.percentile(fam_data["total"], 99)) if fam_data["total"] else 0,
            "components": {k: float(np.mean(v)) if v else 0 
                          for k, v in fam_data["components"].items()},
        }
        # Per-timestep
        summary["per_timestep"] = {}
        for bucket, errs in sorted(fam_data["per_timestep"].items()):
            summary["per_timestep"][str(bucket)] = float(np.mean(errs))
        # Contact vs free
        for key in ["contact", "free"]:
            vals = fam_data["contact_vs_free"][key]
            summary[f"{key}_mean"] = float(np.mean(vals)) if vals else 0
            summary[f"{key}_count"] = len(vals)
        analysis[fam] = summary

    return analysis


def main():
    print("=" * 60)
    print("ERROR DECOMPOSITION ANALYSIS")
    print("=" * 60)

    families = gen_benchmark(10, 500, seed=99)
    t_total = time.time()

    # Train best-recipe model
    print("\nGenerating no_bench data (2K traj)...")
    t0 = time.time()
    S, SN, A, F = gen_patterned(2000, 500, NO_BENCH, seed=42)
    print(f"Done in {time.time()-t0:.0f}s, {len(S):,} pairs")

    print("\nTraining model...")
    in_dim = 2 * FLAT_DIM + N_BODY_GEOMS
    X = np.concatenate([S, SN, F], axis=1)
    model = PlainMLP(in_dim, len(TAU_MAX)).to(device)
    model, vl = train_model(model, X, A, device, epochs=200)

    # Detailed analysis
    print("\nRunning detailed H=1 error analysis...")
    analysis = detailed_h1_analysis(model, families, device)

    # Print summary
    print("\n" + "=" * 60)
    print("ERROR DECOMPOSITION RESULTS")
    print("=" * 60)

    all_total = []
    for fam, data in analysis.items():
        print(f"\n--- {fam} ---")
        print(f"  Total: mean={data['total_mean']:.2f}, "
              f"P95={data['total_p95']:.2f}, P99={data['total_p99']:.2f}")
        print(f"  Components:")
        for comp, val in data['components'].items():
            pct = 100 * val / max(data['total_mean'], 1e-10)
            print(f"    {comp:12s}: {val:.4f} ({pct:.1f}%)")
        print(f"  Contact: mean={data['contact_mean']:.2f} "
              f"(n={data['contact_count']})")
        print(f"  Free:    mean={data['free_mean']:.2f} "
              f"(n={data['free_count']})")
        print(f"  Per timestep:")
        for t, val in data["per_timestep"].items():
            print(f"    t={t:>3s}: {val:.2f}")
        all_total.append(data['total_mean'])

    agg = float(np.mean(all_total))
    print(f"\nOverall H1 AGG: {agg:.2f}")

    total_time = time.time() - t_total
    print(f"\nTOTAL: {total_time:.0f}s ({total_time/60:.1f}min)")

    results = {"analysis": analysis, "h1_agg": agg, "val_loss": vl}
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {OUT}/results.json")


if __name__ == "__main__":
    main()
