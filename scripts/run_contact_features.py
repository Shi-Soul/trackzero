#!/usr/bin/env python3
"""Finding 31: Contact feature richness ablation.

Tests whether richer contact information (beyond binary flags) improves
inverse dynamics at humanoid scale. The core hypothesis: binary contact
flags (15D) are insufficient for the model to learn contact dynamics.
Joint-space constraint forces (27D) provide the physics the model needs.

Conditions:
  flags:     [s_t, s_{t+1}, binary_flags(15D)] → τ     (baseline, 123D)
  qfc:       [s_t, s_{t+1}, qfrc_constraint(27D)] → τ  (135D)
  bias_qfc:  [s_t, s_{t+1}, qfrc_bias+qfc(54D)] → τ   (162D)
  flags_qfc: [s_t, s_{t+1}, flags+qfc(42D)] → τ        (150D)

Usage:
    CUDA_VISIBLE_DEVICES=5 python -m scripts.run_contact_features
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
N_JOINTS = 21
TAU_MAX_NP = np.array(TAU_MAX, dtype=np.float32)

OUT = "outputs/contact_features"
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
            next(self.parameters()).device)
        self.norm_s = torch.tensor(s, dtype=torch.float32).to(
            next(self.parameters()).device)

    def forward(self, x):
        return self.net((x - self.norm_m) / (self.norm_s + 1e-8))


def get_physics_features(d):
    """Extract physics features from MuJoCo data after mj_forward."""
    return {
        "flags": get_contact_flags(d).astype(np.float32),
        "qfc": d.qfrc_constraint.copy().astype(np.float32),
        "bias": d.qfrc_bias.copy().astype(np.float32),
    }


def gen_with_physics(n_traj, traj_len, seed=42, scale=0.15):
    """Generate trajectories with all physics features."""
    rng = np.random.RandomState(seed)
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    nu = m.nu
    patterns = ["white", "ou", "brownian", "sine",
                "step", "chirp", "bang_bang", "ramp"]

    all_S, all_SN, all_A = [], [], []
    all_flags, all_qfc, all_bias = [], [], []

    for i in range(n_traj):
        if i % 500 == 0 and i > 0:
            print(f"  gen: {i}/{n_traj}")
        pat = patterns[i % len(patterns)]
        torques = _gen_torques(pat, traj_len, nu, dt, scale, rng)

        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)

        for t in range(traj_len):
            s_t = mj_to_flat(d.qpos.copy(), d.qvel.copy())
            # Physics features from current state (before applying torque)
            mujoco.mj_forward(m, d)
            pf = get_physics_features(d)

            d.ctrl[:] = torques[t]
            mujoco.mj_step(m, d)
            s_next = mj_to_flat(d.qpos.copy(), d.qvel.copy())

            if np.any(np.isnan(s_next)):
                mujoco.mj_resetData(m, d)
                mujoco.mj_forward(m, d)
                continue

            all_S.append(s_t)
            all_SN.append(s_next)
            all_A.append(torques[t])
            all_flags.append(pf["flags"])
            all_qfc.append(pf["qfc"])
            all_bias.append(pf["bias"])

    print(f"  {len(all_S):,} pairs collected")
    return (np.array(all_S), np.array(all_SN), np.array(all_A),
            np.array(all_flags), np.array(all_qfc), np.array(all_bias))


def _gen_torques(pat, traj_len, nu, dt, scale, rng):
    """Generate torque sequence for a given pattern."""
    if pat == "white":
        return rng.uniform(-1, 1, (traj_len, nu)) * TAU_MAX_NP * scale
    elif pat == "ou":
        tau = np.zeros(nu)
        torques = np.zeros((traj_len, nu))
        for t in range(traj_len):
            tau += 0.15 * (-tau) * dt + \
                   0.3 * np.sqrt(dt) * rng.randn(nu)
            torques[t] = np.clip(
                tau * TAU_MAX_NP, -TAU_MAX_NP * scale,
                TAU_MAX_NP * scale)
        return torques
    elif pat == "brownian":
        inc = rng.randn(traj_len, nu) * 0.01
        raw = np.cumsum(inc, axis=0)
        return np.clip(raw * TAU_MAX_NP, -TAU_MAX_NP * scale,
                       TAU_MAX_NP * scale)
    elif pat == "sine":
        freq = rng.uniform(0.1, 10.0, nu)
        phase = rng.uniform(0, 2 * np.pi, nu)
        amp = rng.uniform(0.3, 1.0, nu) * TAU_MAX_NP * scale
        ts = np.arange(traj_len)[:, None] * dt
        return amp * np.sin(2 * np.pi * freq * ts + phase)
    elif pat == "step":
        n_steps = rng.randint(2, 8)
        bd = sorted(rng.randint(0, traj_len, n_steps))
        bd = [0] + list(bd) + [traj_len]
        torques = np.zeros((traj_len, nu))
        for j in range(len(bd) - 1):
            a = rng.uniform(-1, 1, nu) * TAU_MAX_NP * scale
            torques[bd[j]:bd[j + 1]] = a
        return torques
    elif pat == "chirp":
        ts = np.arange(traj_len) * dt
        f0, f1 = rng.uniform(0.1, 2), rng.uniform(3, 15)
        freq = np.linspace(f0, f1, traj_len)
        phase = rng.uniform(0, 2 * np.pi, nu)
        amp = rng.uniform(0.5, 1.0, nu) * TAU_MAX_NP * scale
        torques = np.zeros((traj_len, nu))
        for j in range(nu):
            torques[:, j] = amp[j] * np.sin(
                2 * np.pi * np.cumsum(freq) * dt + phase[j])
        return torques
    elif pat == "bang_bang":
        torques = np.zeros((traj_len, nu))
        for j in range(nu):
            period = rng.randint(10, 100)
            sign = 1.0
            for t in range(traj_len):
                if t % period == 0:
                    sign *= -1
                torques[t, j] = sign * TAU_MAX_NP[j] * scale
        return torques
    else:  # ramp
        torques = np.zeros((traj_len, nu))
        n_seg = rng.randint(2, 6)
        seg_len = traj_len // n_seg
        for seg in range(n_seg):
            sa = rng.uniform(-1, 1, nu) * TAU_MAX_NP * scale
            ea = rng.uniform(-1, 1, nu) * TAU_MAX_NP * scale
            s, e = seg * seg_len, min((seg + 1) * seg_len, traj_len)
            for t in range(s, e):
                alpha = (t - s) / max(e - s - 1, 1)
                torques[t] = sa * (1 - alpha) + ea * alpha
        return torques


# ── Eval functions ──────────────────────────────────────────────────

def eval_h1_feat(model, families, device, feat_fn):
    """H=1 with custom feature extraction."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    fam_errors = {}
    for fam, trajs in families.items():
        errors = []
        for ref_s, ref_a in trajs:
            step_errs = []
            for t in range(min(500, len(ref_a))):
                if np.any(np.isnan(ref_s[t])):
                    break
                flat_to_mj(ref_s[t], m, d)
                mujoco.mj_forward(m, d)
                feat = feat_fn(d)
                inp = np.concatenate([ref_s[t], ref_s[t + 1], feat])
                with torch.no_grad():
                    tau = model(
                        torch.tensor(inp, dtype=torch.float32)
                        .unsqueeze(0).to(device)
                    ).cpu().numpy()[0]
                flat_to_mj(ref_s[t], m, d)
                d.ctrl[:] = np.clip(tau, -TAU_MAX_NP, TAU_MAX_NP)
                mujoco.mj_step(m, d)
                actual = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if np.any(np.isnan(actual)):
                    step_errs.append(100.0)
                else:
                    step_errs.append(
                        float(np.mean((actual - ref_s[t + 1]) ** 2)))
            if step_errs:
                errors.append(float(np.mean(step_errs)))
        fam_errors[fam] = float(np.mean(errors)) if errors else 1e6
    return fam_errors


def eval_h500_feat(model, families, device, feat_fn):
    """H=500 rollout with custom feature extraction."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    fam_errors = {}
    for fam, trajs in families.items():
        errors = []
        for ref_s, ref_a in trajs:
            flat_to_mj(ref_s[0], m, d)
            mujoco.mj_forward(m, d)
            traj_err, T = 0.0, min(500, len(ref_a))
            for t in range(T):
                s_now = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if np.any(np.isnan(s_now)):
                    traj_err += 100.0 * (T - t)
                    break
                mujoco.mj_forward(m, d)
                feat = feat_fn(d)
                inp = np.concatenate([s_now, ref_s[t + 1], feat])
                with torch.no_grad():
                    tau = model(
                        torch.tensor(inp, dtype=torch.float32)
                        .unsqueeze(0).to(device)
                    ).cpu().numpy()[0]
                d.ctrl[:] = np.clip(tau, -TAU_MAX_NP, TAU_MAX_NP)
                mujoco.mj_step(m, d)
                actual = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if np.any(np.isnan(actual)):
                    traj_err += 100.0 * (T - t)
                    break
                traj_err += float(np.mean(
                    (actual - ref_s[t + 1]) ** 2))
            errors.append(traj_err / T)
        fam_errors[fam] = float(np.mean(errors))
    return fam_errors


# ── Feature condition builders ──────────────────────────────────────

CONDITION_DEFS = {
    "flags": {
        "build_x": lambda s, sn, fl, qfc, bias:
            np.concatenate([s, sn, fl], axis=1),
        "feat_fn": lambda d: get_contact_flags(d).astype(np.float32),
        "feat_dim": N_BODY_GEOMS,
    },
    "qfc": {
        "build_x": lambda s, sn, fl, qfc, bias:
            np.concatenate([s, sn, qfc], axis=1),
        "feat_fn": lambda d: d.qfrc_constraint.copy().astype(np.float32),
        "feat_dim": 27,
    },
    "bias_qfc": {
        "build_x": lambda s, sn, fl, qfc, bias:
            np.concatenate([s, sn, bias, qfc], axis=1),
        "feat_fn": lambda d: np.concatenate([
            d.qfrc_bias.copy(), d.qfrc_constraint.copy()
        ]).astype(np.float32),
        "feat_dim": 54,
    },
    "flags_qfc": {
        "build_x": lambda s, sn, fl, qfc, bias:
            np.concatenate([s, sn, fl, qfc], axis=1),
        "feat_fn": lambda d: np.concatenate([
            get_contact_flags(d), d.qfrc_constraint.copy()
        ]).astype(np.float32),
        "feat_dim": 42,
    },
}


def main():
    print("=" * 60)
    print("Finding 31: Contact feature richness ablation")
    print("=" * 60)

    # Generate data with all features
    print("\nGenerating 2000 trajectories with physics features...")
    t0 = time.time()
    S, SN, A, flags, qfc, bias = gen_with_physics(
        2000, 500, seed=42, scale=0.15)
    gen_time = time.time() - t0
    print(f"  Generated {len(S):,} pairs in {gen_time:.0f}s")
    print(f"  Feature shapes: flags={flags.shape[1]}, "
          f"qfc={qfc.shape[1]}, bias={bias.shape[1]}")

    # Benchmark
    families = gen_benchmark(10, 500, seed=99)
    results = {}

    for cond_name, cond in CONDITION_DEFS.items():
        print(f"\n{'=' * 60}")
        print(f"  Config: {cond_name}")
        print(f"{'=' * 60}")

        X = cond["build_x"](S, SN, flags, qfc, bias)
        in_dim = X.shape[1]
        expected = 2 * FLAT_DIM + cond["feat_dim"]
        print(f"  in_dim={in_dim} (expected={expected}), "
              f"pairs={len(X):,}")

        model = PlainMLP(in_dim, N_JOINTS).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params:,}")

        t_start = time.time()
        model, val = train_model(model, X, A, device,
                                 epochs=200, bs=4096)
        model.eval()

        print("  Evaluating H=1...")
        h1 = eval_h1_feat(model, families, device, cond["feat_fn"])
        h1_agg = float(np.mean(list(h1.values())))
        print(f"  H1: {h1} → AGG={h1_agg:.1f}")

        print("  Evaluating H=500...")
        h500 = eval_h500_feat(model, families, device, cond["feat_fn"])
        h500_agg = float(np.mean(list(h500.values())))
        print(f"  H500: {h500} → AGG={h500_agg:.2e}")

        elapsed = time.time() - t_start
        results[cond_name] = {
            "h1": h1, "h1_agg": h1_agg,
            "h500": h500, "h500_agg": h500_agg,
            "val_loss": float(val),
            "params": n_params,
            "input_dim": in_dim,
            "feat_dim": cond["feat_dim"],
            "elapsed_s": elapsed,
        }
        print(f"  Done in {elapsed:.0f}s")

        with open(f"{OUT}/results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY: Contact feature ablation")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name:12s}: H1={r['h1_agg']:6.1f}  "
              f"H500={r['h500_agg']:10.1f}  "
              f"val={r['val_loss']:.1f}  "
              f"feat_dim={r['feat_dim']}")


if __name__ == "__main__":
    main()
