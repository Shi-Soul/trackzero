#!/usr/bin/env python3
"""Synthesis experiment: combine all Stage 4 improvements.

Tests whether coverage (no_bench) + architecture (accel_feat) + 
optimized recipe are additive.

Conditions:
1. random_only:        white noise, standard MLP
2. no_bench:           6-pattern diversity, standard MLP
3. no_bench_accel:     6-pattern + accel_feat architecture
4. no_bench_accel_reg: 6-pattern + accel_feat + dropout + WD=5e-4

All: 2K traj × 500 steps, scale=0.15, 1024×4, 200ep, cosine LR.
Budget: 1 GPU, ~90 min total.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m scripts.run_synthesis
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
VEL_START = 27
DT = 0.002
TAU_MAX_NP = np.array(TAU_MAX, dtype=np.float32)

OUT = "outputs/synthesis"
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data generation ──────────────────────────────────────────────────

NO_BENCH_PATTERNS = ["white", "ou", "brownian", "sine", "bang_bang", "ramp"]

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
            t = np.arange(traj_len)[:, None] * dt
            torques = amp * np.sin(2*np.pi * freq * t + phase)
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

    S = np.array(all_S, dtype=np.float32)
    SN = np.array(all_SN, dtype=np.float32)
    A = np.array(all_A, dtype=np.float32)
    F = np.array(all_F, dtype=np.float32)
    print(f"  {len(S):,} pairs, nan_resets={nan_resets}")
    return S, SN, A, F


# ── Models ───────────────────────────────────────────────────────────

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


class PlainMLPDropout(nn.Module):
    """MLP with dropout for regularization."""
    def __init__(self, in_dim, out_dim, hidden=1024, layers=4, drop=0.1):
        super().__init__()
        mods = []
        d = in_dim
        for _ in range(layers):
            mods.extend([nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(drop)])
            d = hidden
        mods.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*mods)
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

def eval_h1(model_nn, families, dev, mode="baseline"):
    """Evaluate H=1 per-step accuracy."""
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
                # Build input
                if mode == "accel_feat":
                    v_cur = ref_s[t][VEL_START:]
                    v_next = ref_s[t+1][VEL_START:]
                    accel = (v_next - v_cur) / DT
                    inp = np.concatenate([ref_s[t], ref_s[t+1], flags, accel])
                else:
                    inp = np.concatenate([ref_s[t], ref_s[t+1], flags])
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
                    step_errs.append(float(np.mean((actual - ref_s[t+1])**2)))
            if step_errs:
                errors.append(float(np.mean(step_errs)))
        fam_errors[fam] = float(np.mean(errors)) if errors else 1e6
    return fam_errors


def eval_h500(model_nn, families, dev, mode="baseline"):
    """Evaluate H=500 full trajectory tracking."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    fam_errors = {}
    for fam, trajs in families.items():
        errors = []
        for ref_s, ref_a in trajs:
            flat_to_mj(ref_s[0], m, d)
            mujoco.mj_forward(m, d)
            T = min(500, len(ref_a))
            traj_err = 0.0
            for t in range(T):
                s_now = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if np.any(np.isnan(s_now)):
                    traj_err += 100.0 * (T - t)
                    break
                flags = get_contact_flags(d)
                if mode == "accel_feat":
                    v_cur = s_now[VEL_START:]
                    v_next = ref_s[t+1][VEL_START:]
                    accel = (v_next - v_cur) / DT
                    inp = np.concatenate([s_now, ref_s[t+1], flags, accel])
                else:
                    inp = np.concatenate([s_now, ref_s[t+1], flags])
                with torch.no_grad():
                    tau = model_nn(torch.tensor(inp, dtype=torch.float32)
                                   .unsqueeze(0).to(dev)).cpu().numpy()[0]
                d.ctrl[:] = np.clip(tau, -TAU_MAX_NP, TAU_MAX_NP)
                mujoco.mj_step(m, d)
                actual = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if np.any(np.isnan(actual)):
                    traj_err += 100.0 * (T - t)
                    break
                traj_err += float(np.mean((actual - ref_s[t+1])**2))
            errors.append(traj_err / T)
        fam_errors[fam] = float(np.mean(errors))
    return fam_errors


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SYNTHESIS: Combining Stage 4 improvements")
    print("=" * 60)

    families = gen_benchmark(10, 500, seed=99)
    results = {}
    t_total = time.time()

    # Generate data
    print("\n[1] Generating random-only data (2K traj)...")
    t0 = time.time()
    S_rand, SN_rand, A_rand, F_rand = gen_patterned(
        2000, 500, ["white"], seed=42)
    print(f"    Done in {time.time()-t0:.0f}s")

    print("\n[2] Generating no_bench data (2K traj, 6 patterns)...")
    t0 = time.time()
    S_nb, SN_nb, A_nb, F_nb = gen_patterned(
        2000, 500, NO_BENCH_PATTERNS, seed=42)
    print(f"    Done in {time.time()-t0:.0f}s")

    # ── Condition 1: random_only ──
    print("\n--- Condition 1: random_only ---")
    in_dim = 2 * FLAT_DIM + N_BODY_GEOMS  # 123
    X1 = np.concatenate([S_rand, SN_rand, F_rand], axis=1)
    model1 = PlainMLP(in_dim, len(TAU_MAX)).to(device)
    model1, vl1 = train_model(model1, X1, A_rand, device, epochs=200)
    h1_1 = eval_h1(model1, families, device, "baseline")
    h500_1 = eval_h500(model1, families, device, "baseline")
    agg1 = float(np.mean(list(h1_1.values())))
    agg500_1 = float(np.mean(list(h500_1.values())))
    results["random_only"] = {
        "h1": h1_1, "h1_agg": agg1,
        "h500": h500_1, "h500_agg": agg500_1, "val_loss": vl1}
    print(f"  H1={agg1:.1f} H500={agg500_1:.2e}")

    # ── Condition 2: no_bench ──
    print("\n--- Condition 2: no_bench ---")
    X2 = np.concatenate([S_nb, SN_nb, F_nb], axis=1)
    model2 = PlainMLP(in_dim, len(TAU_MAX)).to(device)
    model2, vl2 = train_model(model2, X2, A_nb, device, epochs=200)
    h1_2 = eval_h1(model2, families, device, "baseline")
    h500_2 = eval_h500(model2, families, device, "baseline")
    agg2 = float(np.mean(list(h1_2.values())))
    agg500_2 = float(np.mean(list(h500_2.values())))
    results["no_bench"] = {
        "h1": h1_2, "h1_agg": agg2,
        "h500": h500_2, "h500_agg": agg500_2, "val_loss": vl2}
    print(f"  H1={agg2:.1f} H500={agg500_2:.2e}")

    # ── Condition 3: no_bench + accel_feat ──
    print("\n--- Condition 3: no_bench + accel_feat ---")
    v_cur = S_nb[:, VEL_START:]
    v_next = SN_nb[:, VEL_START:]
    accel = (v_next - v_cur) / DT
    X3 = np.concatenate([S_nb, SN_nb, F_nb, accel], axis=1)
    in_dim_af = in_dim + 27  # 150
    model3 = PlainMLP(in_dim_af, len(TAU_MAX)).to(device)
    model3, vl3 = train_model(model3, X3, A_nb, device, epochs=200)
    h1_3 = eval_h1(model3, families, device, "accel_feat")
    h500_3 = eval_h500(model3, families, device, "accel_feat")
    agg3 = float(np.mean(list(h1_3.values())))
    agg500_3 = float(np.mean(list(h500_3.values())))
    results["no_bench_accel"] = {
        "h1": h1_3, "h1_agg": agg3,
        "h500": h500_3, "h500_agg": agg500_3, "val_loss": vl3}
    print(f"  H1={agg3:.1f} H500={agg500_3:.2e}")

    # ── Condition 4: no_bench + accel_feat + dropout ──
    print("\n--- Condition 4: no_bench + accel_feat + dropout ---")
    model4 = PlainMLPDropout(in_dim_af, len(TAU_MAX), drop=0.1).to(device)
    model4, vl4 = train_model(model4, X3, A_nb, device, epochs=200)
    model4.eval()
    h1_4 = eval_h1(model4, families, device, "accel_feat")
    h500_4 = eval_h500(model4, families, device, "accel_feat")
    agg4 = float(np.mean(list(h1_4.values())))
    agg500_4 = float(np.mean(list(h500_4.values())))
    results["no_bench_accel_drop"] = {
        "h1": h1_4, "h1_agg": agg4,
        "h500": h500_4, "h500_agg": agg500_4, "val_loss": vl4}
    print(f"  H1={agg4:.1f} H500={agg500_4:.2e}")

    # Save
    total_time = time.time() - t_total
    print(f"\nTOTAL: {total_time:.0f}s ({total_time/60:.1f}min)")
    print("\nSummary:")
    print(f"{'Config':<25} {'H1_AGG':>10} {'H500_AGG':>12} {'val_loss':>10}")
    for name, r in results.items():
        print(f"{name:<25} {r['h1_agg']:>10.1f} {r['h500_agg']:>12.2e} "
              f"{r['val_loss']:>10.5f}")

    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT}/results.json")


if __name__ == "__main__":
    main()
