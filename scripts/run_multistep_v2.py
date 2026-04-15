#!/usr/bin/env python3
"""Multi-step reference conditioning (Stage 2C test).

Tests whether providing a window of future reference states helps
the model anticipate dynamics and improve per-step accuracy.

Input for window k: [s_t, s_{t+1}, ..., s_{t+k}, contact] → τ_t
- k=1: standard (baseline)
- k=2: see 2 steps ahead
- k=4: see 4 steps ahead
- k=8: see 8 steps ahead

Uses no_bench data (6 patterns, scale=0.15, 2K traj, 200ep).
Budget: 1 GPU, ~80 min total.

Usage:
    CUDA_VISIBLE_DEVICES=2 python -m scripts.run_multistep_v2
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
DT = 0.002
TAU_MAX_NP = np.array(TAU_MAX, dtype=np.float32)
NO_BENCH = ["white", "ou", "brownian", "sine", "bang_bang", "ramp"]

OUT = "outputs/multistep_v2"
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data generation ──────────────────────────────────────────────────

def gen_raw_trajectories(n_traj, traj_len, seed=42):
    """Generate trajectories, returning raw state sequences for windowing."""
    rng = np.random.RandomState(seed)
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    nu = m.nu

    all_trajs = []  # list of (states, actions, flags)
    nan_count = 0

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
        states = [mj_to_flat(d.qpos.copy(), d.qvel.copy())]
        actions = []
        flags_list = []

        for t in range(traj_len):
            flags_list.append(get_contact_flags(d))
            tau = np.clip(torques[t], -TAU_MAX_NP, TAU_MAX_NP)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            sn = mj_to_flat(d.qpos.copy(), d.qvel.copy())
            if np.any(np.isnan(sn)):
                nan_count += 1
                break
            states.append(sn)
            actions.append(tau)

        if len(states) > 10:  # keep if > 10 valid steps
            all_trajs.append((
                np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.float32),
                np.array(flags_list[:len(actions)], dtype=np.float32)
            ))

    print(f"  {len(all_trajs)} valid trajs, nan_count={nan_count}")
    return all_trajs


def make_window_pairs(trajs, k):
    """Extract (s_t, s_{t+1}, ..., s_{t+k}, flags_t) → τ_t pairs."""
    all_X, all_Y = [], []
    for states, actions, flags in trajs:
        T = len(actions)
        for t in range(T - k + 1):
            # [s_t, s_{t+1}, ..., s_{t+k}] = (k+1) * 54D
            window = states[t:t+k+1].flatten()
            x = np.concatenate([window, flags[t]])
            all_X.append(x)
            all_Y.append(actions[t])
    X = np.array(all_X, dtype=np.float32)
    Y = np.array(all_Y, dtype=np.float32)
    print(f"  k={k}: {len(X):,} pairs, in_dim={X.shape[1]}")
    return X, Y


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

def eval_h1_window(model_nn, families, dev, k):
    """H=1 eval with window of k future reference states."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    fam_errors = {}
    for fam, trajs in families.items():
        errors = []
        for ref_s, ref_a in trajs:
            step_errs = []
            T = min(500, len(ref_a))
            for t in range(T - k + 1):
                if any(np.any(np.isnan(ref_s[t+i]))
                       for i in range(k+1)):
                    break
                flat_to_mj(ref_s[t], m, d)
                mujoco.mj_forward(m, d)
                flags = get_contact_flags(d)
                window = ref_s[t:t+k+1].flatten()
                inp = np.concatenate([window, flags])
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


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MULTI-STEP REFERENCE CONDITIONING (Stage 2C)")
    print("=" * 60)

    families = gen_benchmark(10, 500, seed=99)
    results = {}
    t_total = time.time()

    # Generate raw trajectories once
    print("\nGenerating 2000 raw trajectories...")
    t0 = time.time()
    trajs = gen_raw_trajectories(2000, 500, seed=42)
    print(f"Done in {time.time()-t0:.0f}s")

    for k in [1, 2, 4, 8]:
        print(f"\n--- Window k={k} ---")
        t0 = time.time()
        X, Y = make_window_pairs(trajs, k)
        in_dim = (k + 1) * FLAT_DIM + N_BODY_GEOMS
        model = PlainMLP(in_dim, len(TAU_MAX)).to(device)
        model, vl = train_model(model, X, Y, device, epochs=200)

        print(f"  Evaluating H=1...")
        h1 = eval_h1_window(model, families, device, k)
        h1_agg = float(np.mean(list(h1.values())))
        print(f"  H1: {h1} → AGG={h1_agg:.1f}")

        results[f"k={k}"] = {
            "h1": h1, "h1_agg": h1_agg,
            "val_loss": vl, "pairs": len(X),
            "in_dim": in_dim,
            "time": time.time() - t0}
        print(f"  Done in {time.time()-t0:.0f}s")

    total_time = time.time() - t_total
    print(f"\nTOTAL: {total_time:.0f}s ({total_time/60:.1f}min)")
    print("\nSummary:")
    print(f"{'Window':<10} {'H1_AGG':>10} {'val_loss':>10} {'pairs':>10}")
    for name, r in results.items():
        print(f"{name:<10} {r['h1_agg']:>10.1f} {r['val_loss']:>10.5f} "
              f"{r['pairs']:>10,}")

    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT}/results.json")


if __name__ == "__main__":
    main()
