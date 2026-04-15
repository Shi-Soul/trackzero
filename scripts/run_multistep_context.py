#!/usr/bin/env python3
"""Finding 30: Multi-step reference conditioning at humanoid scale.

Tests whether providing a window of future reference states helps
torque prediction. Directly tests Stage 2C hypothesis: "conditioning
on a window of future reference states lets the policy anticipate
upcoming dynamics and plan ahead."

Conditions:
  k=1: [s_t, s_{t+1}, flags] → τ           (baseline, 123D input)
  k=2: [s_t, s_{t+1}, s_{t+2}, flags] → τ  (177D input)
  k=4: [s_t, ..., s_{t+4}, flags] → τ       (285D input)
  k=8: [s_t, ..., s_{t+8}, flags] → τ       (501D input)

Usage:
    CUDA_VISIBLE_DEVICES=3 python -m scripts.run_multistep_context
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

OUT = "outputs/multistep_context"
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


# ── Data generation (returns raw trajectories) ──────────────────────

def gen_raw_trajectories(n_traj, traj_len, seed=42, scale=0.15):
    """Generate raw trajectory data with diverse torque patterns."""
    rng = np.random.RandomState(seed)
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    nu = m.nu
    patterns = ["white", "ou", "brownian", "sine",
                "step", "chirp", "bang_bang", "ramp"]
    trajectories = []

    for i in range(n_traj):
        if i % 500 == 0 and i > 0:
            print(f"  gen_raw: {i}/{n_traj}")
        pat = patterns[i % len(patterns)]

        # Generate torque sequence
        if pat == "white":
            torques = rng.uniform(-1, 1, (traj_len, nu)) * \
                      TAU_MAX_NP * scale
        elif pat == "ou":
            tau = np.zeros(nu)
            torques = np.zeros((traj_len, nu))
            theta, sigma = 0.15, 0.3
            for t in range(traj_len):
                tau += theta * (-tau) * dt + \
                       sigma * np.sqrt(dt) * rng.randn(nu)
                torques[t] = np.clip(
                    tau * TAU_MAX_NP, -TAU_MAX_NP * scale,
                    TAU_MAX_NP * scale)
        elif pat == "brownian":
            inc = rng.randn(traj_len, nu) * 0.01
            raw = np.cumsum(inc, axis=0)
            torques = np.clip(
                raw * TAU_MAX_NP, -TAU_MAX_NP * scale,
                TAU_MAX_NP * scale)
        elif pat == "sine":
            freq = rng.uniform(0.1, 10.0, nu)
            phase = rng.uniform(0, 2 * np.pi, nu)
            amp = rng.uniform(0.3, 1.0, nu) * TAU_MAX_NP * scale
            ts = np.arange(traj_len)[:, None] * dt
            torques = amp * np.sin(2 * np.pi * freq * ts + phase)
        elif pat == "step":
            n_steps = rng.randint(2, 8)
            bd = sorted(rng.randint(0, traj_len, n_steps))
            bd = [0] + list(bd) + [traj_len]
            torques = np.zeros((traj_len, nu))
            for j in range(len(bd) - 1):
                a = rng.uniform(-1, 1, nu) * TAU_MAX_NP * scale
                torques[bd[j]:bd[j + 1]] = a
        elif pat == "chirp":
            ts = np.arange(traj_len) * dt
            f0 = rng.uniform(0.1, 2)
            f1 = rng.uniform(3, 15)
            freq = np.linspace(f0, f1, traj_len)
            phase = rng.uniform(0, 2 * np.pi, nu)
            amp = rng.uniform(0.5, 1.0, nu) * TAU_MAX_NP * scale
            torques = np.zeros((traj_len, nu))
            for j in range(nu):
                torques[:, j] = amp[j] * np.sin(
                    2 * np.pi * np.cumsum(freq) * dt + phase[j])
        elif pat == "bang_bang":
            torques = np.zeros((traj_len, nu))
            for j in range(nu):
                period = rng.randint(10, 100)
                sign = 1.0
                for t in range(traj_len):
                    if t % period == 0:
                        sign *= -1
                    torques[t, j] = sign * TAU_MAX_NP[j] * scale
        elif pat == "ramp":
            torques = np.zeros((traj_len, nu))
            n_seg = rng.randint(2, 6)
            seg_len = traj_len // n_seg
            for seg in range(n_seg):
                start_amp = rng.uniform(-1, 1, nu) * TAU_MAX_NP * scale
                end_amp = rng.uniform(-1, 1, nu) * TAU_MAX_NP * scale
                s, e = seg * seg_len, min((seg + 1) * seg_len, traj_len)
                for t_idx in range(s, e):
                    alpha = (t_idx - s) / max(e - s - 1, 1)
                    torques[t_idx] = start_amp * (1 - alpha) + \
                                     end_amp * alpha
        else:
            torques = rng.uniform(-1, 1, (traj_len, nu)) * \
                      TAU_MAX_NP * scale

        # Simulate trajectory
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        states = np.zeros((traj_len + 1, FLAT_DIM))
        actions = np.zeros((traj_len, nu))
        contacts = np.zeros((traj_len, N_BODY_GEOMS))
        states[0] = mj_to_flat(d.qpos.copy(), d.qvel.copy())

        valid = True
        for t in range(traj_len):
            d.ctrl[:] = torques[t]
            contacts[t] = get_contact_flags(d)
            mujoco.mj_step(m, d)
            s = mj_to_flat(d.qpos.copy(), d.qvel.copy())
            if np.any(np.isnan(s)):
                mujoco.mj_resetData(m, d)
                mujoco.mj_forward(m, d)
                s = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                valid = False
            states[t + 1] = s
            actions[t] = torques[t]
            if not valid:
                valid = True  # reset and continue

        trajectories.append((states, actions, contacts))

    return trajectories


def make_window_pairs(trajectories, window_k):
    """Extract windowed input-output pairs from raw trajectories."""
    X_list, A_list = [], []
    for states, actions, contacts in trajectories:
        T = len(actions)
        max_t = T - window_k  # last valid start index
        for t in range(max(max_t, 0)):
            s_t = states[t]
            # Future reference: [s_{t+1}, ..., s_{t+k}]
            future = states[t + 1: t + window_k + 1].flatten()
            flags = contacts[t]
            x = np.concatenate([s_t, future, flags])
            X_list.append(x)
            A_list.append(actions[t])
    X = np.array(X_list, dtype=np.float32)
    A = np.array(A_list, dtype=np.float32)
    return X, A


# ── Evaluation with window ──────────────────────────────────────────

def eval_h1_window(model, families, device, window_k):
    """H=1 tracking with multi-step reference window."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    fam_errors = {}
    for fam, trajs in families.items():
        errors = []
        for ref_s, ref_a in trajs:
            step_errs = []
            T = min(500, len(ref_a))
            for t in range(T - window_k + 1):
                if np.any(np.isnan(ref_s[t])):
                    break
                # Set simulator to reference state
                flat_to_mj(ref_s[t], m, d)
                mujoco.mj_forward(m, d)
                flags = get_contact_flags(d)
                # Build window: [s_t, s_{t+1}, ..., s_{t+k}, flags]
                future = ref_s[t + 1: t + window_k + 1].flatten()
                inp = np.concatenate([ref_s[t], future, flags])
                with torch.no_grad():
                    tau = model(
                        torch.tensor(inp, dtype=torch.float32)
                        .unsqueeze(0).to(device)
                    ).cpu().numpy()[0]
                # Apply torque and step
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


def eval_h500_window(model, families, device, window_k):
    """H=500 rollout with multi-step reference window."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    fam_errors = {}
    for fam, trajs in families.items():
        errors = []
        for ref_s, ref_a in trajs:
            flat_to_mj(ref_s[0], m, d)
            mujoco.mj_forward(m, d)
            traj_err = 0.0
            T = min(500, len(ref_a))
            for t in range(T - window_k + 1):
                s_now = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if np.any(np.isnan(s_now)):
                    traj_err += 100.0 * (T - t)
                    break
                flags = get_contact_flags(d)
                # Future from REFERENCE (not rollout)
                future = ref_s[t + 1: t + window_k + 1].flatten()
                inp = np.concatenate([s_now, future, flags])
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
            errors.append(traj_err / max(T - window_k + 1, 1))
        fam_errors[fam] = float(np.mean(errors))
    return fam_errors


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Finding 30: Multi-step reference conditioning")
    print("=" * 60)

    # Generate raw trajectories once
    print("\nGenerating 2000 trajectories...")
    t0 = time.time()
    trajs = gen_raw_trajectories(2000, 500, seed=42, scale=0.15)
    print(f"  Generated in {time.time() - t0:.0f}s")

    # Benchmark
    families = gen_benchmark(10, 500, seed=99)

    results = {}
    for window_k in [1, 2, 4, 8]:
        label = f"k={window_k}"
        print(f"\n{'=' * 60}")
        print(f"  Config: {label}")
        print(f"{'=' * 60}")

        # Extract windowed pairs
        X, A = make_window_pairs(trajs, window_k)
        in_dim = X.shape[1]
        expected_dim = FLAT_DIM + FLAT_DIM * window_k + N_BODY_GEOMS
        assert in_dim == expected_dim, \
            f"in_dim={in_dim}, expected={expected_dim}"
        print(f"  Pairs: {len(X):,}, in_dim: {in_dim}")

        # Train
        model = PlainMLP(in_dim, N_JOINTS).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params:,}")
        t_start = time.time()
        model, val = train_model(model, X, A, device,
                                 epochs=200, bs=4096)
        model.eval()

        # Eval H1
        print("  Evaluating H=1...")
        h1 = eval_h1_window(model, families, device, window_k)
        h1_agg = float(np.mean(list(h1.values())))
        print(f"  H1: {h1} → AGG={h1_agg:.1f}")

        # Eval H500
        print("  Evaluating H=500...")
        h500 = eval_h500_window(model, families, device, window_k)
        h500_agg = float(np.mean(list(h500.values())))
        print(f"  H500: {h500} → AGG={h500_agg:.2e}")

        elapsed = time.time() - t_start
        results[label] = {
            "h1": h1, "h1_agg": h1_agg,
            "h500": h500, "h500_agg": h500_agg,
            "val_loss": float(val),
            "params": n_params,
            "input_dim": in_dim,
            "window_k": window_k,
            "n_pairs": len(X),
            "elapsed_s": elapsed,
        }
        print(f"  Done in {elapsed:.0f}s")

        # Save incremental results
        with open(f"{OUT}/results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Multi-step context ablation")
    print("=" * 60)
    for label, r in results.items():
        print(f"  {label:8s}: H1={r['h1_agg']:6.1f}  "
              f"H500={r['h500_agg']:10.1f}  "
              f"val={r['val_loss']:.1f}  "
              f"dim={r['input_dim']}  "
              f"pairs={r['n_pairs']:,}")


if __name__ == "__main__":
    main()
