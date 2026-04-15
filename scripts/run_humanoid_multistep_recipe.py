#!/usr/bin/env python3
"""Stage 4 follow-up: test README hypothesis 5 on the best blind recipe.

Hypothesis:
  Conditioning on a short future reference window significantly
  outperforms single-step conditioning at humanoid scale.

Protocol:
  - Training data: 2K humanoid trajectories, 500 steps each
  - Data policy: blind no_bench coverage
    (white, ou, brownian, sine, bang_bang, ramp)
  - Model: 1024x4 MLP with dropout=0.1
  - Inputs:
      k=1: [s_t, s_{t+1}, contact_t, qdd_t]
      k=4: [s_t, s_{t+1:t+4}, contact_t, qdd_t]
    where qdd_t = (v_{t+1} - v_t) / dt
  - Eval: canonical humanoid benchmark
    gen_benchmark(20, 500, seed=99)

This keeps the experiment inside the proposal's Stage 2C / Stage 4 scope:
future-reference conditioning, evaluated on the canonical humanoid benchmark.

Usage:
  CUDA_VISIBLE_DEVICES=7 uv run python -m scripts.run_humanoid_multistep_recipe
"""

import json
import os
import time

import mujoco
import numpy as np
import torch
import torch.nn as nn

from scripts.models_structured import _build_mlp
from scripts.run_humanoid import (
    HUMANOID_XML,
    N_BODY_GEOMS,
    TAU_MAX,
    flat_to_mj,
    gen_benchmark,
    get_contact_flags,
    mj_to_flat,
)
from scripts.run_structured import train_model


FLAT_DIM = 54
VEL_START = 27
DT = 0.002
TAU_MAX_NP = np.array(TAU_MAX, dtype=np.float32)
NO_BENCH = ["white", "ou", "brownian", "sine", "bang_bang", "ramp"]

OUT = "outputs/humanoid_multistep_recipe"
os.makedirs(OUT, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PlainMLPDropout(nn.Module):
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
            self.norm_m.device
        )
        self.norm_s = torch.tensor(s, dtype=torch.float32).to(
            self.norm_s.device
        )

    def forward(self, x):
        return self.net((x - self.norm_m) / (self.norm_s + 1e-8))


def _make_torques(rng, pat, traj_len, nu, dt, scale=0.15):
    if pat == "white":
        return rng.uniform(-1, 1, (traj_len, nu)) * TAU_MAX_NP * scale
    if pat == "ou":
        tau = np.zeros(nu)
        torques = np.zeros((traj_len, nu))
        for t in range(traj_len):
            tau += 0.15 * (-tau) * dt + 0.3 * np.sqrt(dt) * rng.randn(nu)
            torques[t] = np.clip(
                tau * TAU_MAX_NP, -TAU_MAX_NP * scale, TAU_MAX_NP * scale
            )
        return torques
    if pat == "brownian":
        raw = np.cumsum(rng.randn(traj_len, nu) * 0.01, axis=0)
        return np.clip(raw * TAU_MAX_NP, -TAU_MAX_NP * scale, TAU_MAX_NP * scale)
    if pat == "sine":
        freq = rng.uniform(0.1, 10.0, nu)
        phase = rng.uniform(0, 2 * np.pi, nu)
        amp = rng.uniform(0.3, 1.0, nu) * TAU_MAX_NP * scale
        t_arr = np.arange(traj_len)[:, None] * dt
        return amp * np.sin(2 * np.pi * freq * t_arr + phase)
    if pat == "bang_bang":
        torques = np.zeros((traj_len, nu))
        for j in range(nu):
            sw = sorted(rng.randint(0, traj_len, rng.randint(3, 10)))
            sw = [0] + list(sw) + [traj_len]
            for k in range(len(sw) - 1):
                torques[sw[k] : sw[k + 1], j] = (
                    rng.choice([-1, 1]) * TAU_MAX_NP[j] * scale
                )
        return torques
    if pat == "ramp":
        start = rng.uniform(-1, 1, nu) * TAU_MAX_NP * scale
        end = rng.uniform(-1, 1, nu) * TAU_MAX_NP * scale
        alpha = np.linspace(0, 1, traj_len)[:, None]
        return start + alpha * (end - start)
    raise ValueError(f"Unknown pattern: {pat}")


def gen_raw_trajectories(n_traj, traj_len, seed=42):
    """Generate no_bench trajectories for future-window training."""
    rng = np.random.RandomState(seed)
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    nu = m.nu

    trajs = []
    nan_count = 0
    for i in range(n_traj):
        if i % 500 == 0:
            print(f"  gen: {i}/{n_traj}")
        pat = NO_BENCH[i % len(NO_BENCH)]
        torques = _make_torques(rng, pat, traj_len, nu, dt)

        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        states = [mj_to_flat(d.qpos.copy(), d.qvel.copy())]
        actions = []
        flags = []

        for t in range(traj_len):
            flags.append(get_contact_flags(d))
            tau = np.clip(torques[t], -TAU_MAX_NP, TAU_MAX_NP)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            sn = mj_to_flat(d.qpos.copy(), d.qvel.copy())
            if np.any(np.isnan(sn)):
                nan_count += 1
                break
            states.append(sn)
            actions.append(tau)

        if len(actions) > 10:
            trajs.append(
                (
                    np.array(states, dtype=np.float32),
                    np.array(actions, dtype=np.float32),
                    np.array(flags[: len(actions)], dtype=np.float32),
                )
            )

    print(f"  kept {len(trajs)} trajectories, nan_count={nan_count}")
    return trajs


def make_window_pairs(trajs, k):
    """Build [s_t, s_{t+1:t+k}, flags_t, qdd_t] -> tau_t pairs."""
    all_x, all_y = [], []
    for states, actions, flags in trajs:
        t_max = len(actions)
        for t in range(t_max):
            ref_states = []
            for off in range(k + 1):
                idx = min(t + off, len(states) - 1)
                ref_states.append(states[idx])
            v_cur = ref_states[0][VEL_START:]
            v_next = ref_states[1][VEL_START:]
            accel = (v_next - v_cur) / DT
            x = np.concatenate([np.concatenate(ref_states), flags[t], accel])
            all_x.append(x)
            all_y.append(actions[t])
    x_arr = np.array(all_x, dtype=np.float32)
    y_arr = np.array(all_y, dtype=np.float32)
    print(f"  k={k}: {len(x_arr):,} pairs, in_dim={x_arr.shape[1]}")
    return x_arr, y_arr


def _window_input(cur_state, ref_states, flags, t, k):
    future = [cur_state]
    for off in range(1, k + 1):
        idx = min(t + off, len(ref_states) - 1)
        future.append(ref_states[idx])
    v_cur = cur_state[VEL_START:]
    v_next = future[1][VEL_START:]
    accel = (v_next - v_cur) / DT
    return np.concatenate([np.concatenate(future), flags, accel])


def eval_h1_window(model_nn, families, device, k):
    """Per-step accuracy with actual current state and future reference window."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    fam_errors = {}
    for fam, trajs in families.items():
        errors = []
        for ref_s, ref_a in trajs:
            step_errs = []
            t_max = min(500, len(ref_a))
            for t in range(t_max):
                if np.any(np.isnan(ref_s[t])) or np.any(np.isnan(ref_s[t + 1])):
                    break
                flat_to_mj(ref_s[t], m, d)
                mujoco.mj_forward(m, d)
                cur_state = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                flags = get_contact_flags(d)
                inp = _window_input(cur_state, ref_s, flags, t, k)
                with torch.no_grad():
                    tau = model_nn(
                        torch.tensor(inp, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(device)
                    ).cpu().numpy()[0]
                flat_to_mj(ref_s[t], m, d)
                d.ctrl[:] = np.clip(tau, -TAU_MAX_NP, TAU_MAX_NP)
                mujoco.mj_step(m, d)
                actual = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if np.any(np.isnan(actual)):
                    step_errs.append(100.0)
                else:
                    step_errs.append(float(np.mean((actual - ref_s[t + 1]) ** 2)))
            if step_errs:
                errors.append(float(np.mean(step_errs)))
        fam_errors[fam] = float(np.mean(errors)) if errors else 1e6
    return fam_errors


def eval_h500_window(model_nn, families, device, k):
    """Closed-loop tracking with future reference window."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    fam_errors = {}
    for fam, trajs in families.items():
        errors = []
        for ref_s, ref_a in trajs:
            flat_to_mj(ref_s[0], m, d)
            mujoco.mj_forward(m, d)
            traj_err = 0.0
            t_max = min(500, len(ref_a))
            for t in range(t_max):
                cur_state = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if np.any(np.isnan(cur_state)):
                    traj_err += 100.0 * (t_max - t)
                    break
                flags = get_contact_flags(d)
                inp = _window_input(cur_state, ref_s, flags, t, k)
                with torch.no_grad():
                    tau = model_nn(
                        torch.tensor(inp, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(device)
                    ).cpu().numpy()[0]
                d.ctrl[:] = np.clip(tau, -TAU_MAX_NP, TAU_MAX_NP)
                mujoco.mj_step(m, d)
                actual = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if np.any(np.isnan(actual)):
                    traj_err += 100.0 * (t_max - t)
                    break
                traj_err += float(np.mean((actual - ref_s[t + 1]) ** 2))
            errors.append(traj_err / t_max)
        fam_errors[fam] = float(np.mean(errors))
    return fam_errors


def run_condition(name, trajs, families, k):
    print(f"\n--- {name} ---")
    t0 = time.time()
    x, y = make_window_pairs(trajs, k)
    model = PlainMLPDropout(x.shape[1], len(TAU_MAX), drop=0.1).to(DEVICE)
    model, val_loss = train_model(model, x, y, DEVICE, epochs=200)
    model.eval()

    h1 = eval_h1_window(model, families, DEVICE, k)
    h500 = eval_h500_window(model, families, DEVICE, k)
    result = {
        "k": k,
        "pairs": len(x),
        "in_dim": int(x.shape[1]),
        "val_loss": float(val_loss),
        "h1": h1,
        "h1_agg": float(np.mean(list(h1.values()))),
        "h500": h500,
        "h500_agg": float(np.mean(list(h500.values()))),
        "time_s": time.time() - t0,
    }
    print(
        f"  H1={result['h1_agg']:.3f} "
        f"H500={result['h500_agg']:.3f} "
        f"val={result['val_loss']:.4f}"
    )
    return result


def main():
    print("=" * 60)
    print("HUMANOID MULTI-STEP RECIPE FOLLOW-UP")
    print("=" * 60)
    print("Proposal hypothesis 5: future window > single-step")
    print("Data: no_bench blind coverage, Eval: canonical benchmark (20/fam)")
    print(f"Device: {DEVICE}")

    t_total = time.time()
    print("\n[1] Generating no_bench raw trajectories...")
    trajs = gen_raw_trajectories(2000, 500, seed=42)

    print("\n[2] Building canonical benchmark...")
    families = gen_benchmark(20, 500, seed=99)

    print("\n[3] Training/evaluating conditions...")
    results = {
        "k1_accel_drop": run_condition("k1_accel_drop", trajs, families, k=1),
        "k4_accel_drop": run_condition("k4_accel_drop", trajs, families, k=4),
    }

    total_time = time.time() - t_total
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':<18} {'H1_AGG':>10} {'H500_AGG':>12} {'val_loss':>10}")
    for name, r in results.items():
        print(
            f"{name:<18} {r['h1_agg']:>10.3f} "
            f"{r['h500_agg']:>12.3f} {r['val_loss']:>10.4f}"
        )
    print(f"TOTAL: {total_time/60:.1f} min")

    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {OUT}/results.json")


if __name__ == "__main__":
    main()
