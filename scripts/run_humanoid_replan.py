#!/usr/bin/env python3
"""Short-horizon replanning for humanoid.

Instead of predicting 500 steps in one shot, we replan every K steps
using the oracle to compute the next K reference states from the
current (actual) state. This tests whether short-horizon prediction
can bridge the distribution shift gap.

Compares: K=1 (oracle), K=5, K=10, K=25, K=50, K=100, K=500 (baseline)

The replan oracle is NOT cheating — it represents what a model-based
planner could provide. The question is: what horizon length is needed
for the learned policy to track reliably?

Usage:
    CUDA_VISIBLE_DEVICES=4 python -m scripts.run_humanoid_replan
"""
import json, os, time
import mujoco
import numpy as np
import torch
import torch.nn as nn

from scripts.run_humanoid import (
    HUMANOID_XML, N_BODY_GEOMS, GEOM_OFFSET, TAU_MAX,
    mj_to_flat, flat_to_mj, get_contact_flags,
    gen_data, gen_benchmark,
)
from scripts.models_structured import _build_mlp
from scripts.run_structured import train_model

FLAT_STATE_DIM = 54
TAU_MAX_NP = np.array(TAU_MAX, dtype=np.float32)
OUT = "outputs/humanoid_replan"
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


def oracle_replan(m, d, current_state, target_torques, n_steps):
    """Generate reference trajectory from current state using oracle.

    Given the current actual state and a torque sequence, simulate
    forward to get the reference states the policy should track.
    Returns (ref_states, ref_actions) of length n_steps.
    """
    flat_to_mj(current_state, m, d)
    ref_states = [current_state.copy()]
    ref_actions = []
    for t in range(n_steps):
        tau = np.clip(target_torques[t], -TAU_MAX_NP, TAU_MAX_NP)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        s_next = mj_to_flat(d.qpos.copy(), d.qvel.copy())
        if np.any(np.isnan(s_next)):
            # Pad remaining with last valid state
            for _ in range(n_steps - t):
                ref_states.append(ref_states[-1].copy())
                ref_actions.append(np.zeros_like(tau))
            break
        ref_states.append(s_next)
        ref_actions.append(tau)
    return np.array(ref_states), np.array(ref_actions)


def rollout_with_replan(model_nn, ref_states_orig, ref_actions_orig,
                        replan_horizon, device):
    """Roll out policy with replanning every K steps.

    Every replan_horizon steps, we take the actual current state and
    use the oracle to generate a new local reference trajectory of
    length replan_horizon. The policy tracks this local reference.

    For replan_horizon=500, this is identical to the standard benchmark.
    For replan_horizon=1, this is H=1 (oracle provides every next state).
    """
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)

    T = len(ref_actions_orig)
    K = min(replan_horizon, T)

    # Start from the initial state of the original reference
    flat_to_mj(ref_states_orig[0], m, d)
    mujoco.mj_forward(m, d)
    current_state = mj_to_flat(d.qpos.copy(), d.qvel.copy())

    actual_traj = [current_state.copy()]
    total_error = 0.0
    n_valid = 0

    t = 0
    while t < T:
        # Generate local reference from current actual state
        remaining = T - t
        local_len = min(K, remaining)
        local_torques = ref_actions_orig[t:t+local_len]

        local_ref_s, local_ref_a = oracle_replan(
            m, d, current_state, local_torques, local_len)

        # Track local reference for local_len steps
        flat_to_mj(current_state, m, d)
        mujoco.mj_forward(m, d)

        for j in range(local_len):
            if j + 1 >= len(local_ref_s):
                break
            s_now = mj_to_flat(d.qpos.copy(), d.qvel.copy())
            s_target = local_ref_s[j + 1]
            flags = get_contact_flags(d)
            inp = np.concatenate([s_now, s_target, flags])

            with torch.no_grad():
                tau = model_nn(
                    torch.tensor(inp, dtype=torch.float32)
                    .unsqueeze(0).to(device)
                ).cpu().numpy()[0]

            d.ctrl[:] = np.clip(tau, -TAU_MAX_NP, TAU_MAX_NP)
            mujoco.mj_step(m, d)
            s_actual = mj_to_flat(d.qpos.copy(), d.qvel.copy())

            if np.any(np.isnan(s_actual)):
                # Diverged — fill rest with large error
                for _ in range(T - t - j - 1):
                    actual_traj.append(actual_traj[-1])
                return actual_traj, float('inf')

            actual_traj.append(s_actual.copy())
            # Track error against ORIGINAL reference
            orig_idx = t + j + 1
            if orig_idx < len(ref_states_orig):
                err = np.mean(
                    (s_actual - ref_states_orig[orig_idx])**2)
                total_error += err
                n_valid += 1

        t += local_len
        # Update current state for next replan
        current_state = mj_to_flat(d.qpos.copy(), d.qvel.copy())

    avg_err = total_error / max(n_valid, 1)
    return actual_traj, avg_err


def main():
    print("=" * 60)
    print("HUMANOID SHORT-HORIZON REPLANNING")
    print("=" * 60)

    # Generate data and train
    print("\n[1] Generating training data (2K traj)...")
    t0 = time.time()
    S, SN, A, F = gen_data(2000, 500, seed=42)
    print(f"    {len(S):,} pairs in {time.time()-t0:.0f}s")

    in_dim = 2 * FLAT_STATE_DIM + N_BODY_GEOMS
    X = np.concatenate([S, SN, F], axis=1)
    model = PlainMLP(in_dim, len(TAU_MAX)).to(device)
    print(f"\n[2] Training...")
    train_model(model, X, A, epochs=200, bs=2048, device=device)
    model.eval()

    # Generate benchmark
    print("\n[3] Generating benchmark (20 per family, 500 steps)...")
    families = gen_benchmark(20, 500, seed=99)

    # Test different replan horizons
    horizons = [5, 10, 25, 50, 100, 500]
    results = {}

    for K in horizons:
        print(f"\n[4] Replan horizon K={K}")
        fam_errors = {}
        for fam in ["uniform", "step", "chirp"]:
            errors = []
            for ref_s, ref_a in families[fam]:
                _, avg_err = rollout_with_replan(
                    model, ref_s, ref_a, K, device)
                errors.append(avg_err)
            mean_err = float(np.mean(errors))
            fam_errors[fam] = mean_err
            print(f"    {fam}: {mean_err:.4e}")

        agg = float(np.mean(list(fam_errors.values())))
        results[f"K={K}"] = {
            "horizon": K,
            "agg": agg,
            **fam_errors,
        }
        print(f"    AGG={agg:.4e}")

    # Summary
    print("\n" + "=" * 60)
    print("REPLANNING SUMMARY")
    print("=" * 60)
    print(f"{'Horizon':>8} {'AGG':>12} {'uniform':>12} "
          f"{'step':>12} {'chirp':>12}")
    for k, v in sorted(results.items(), key=lambda x: x[1]["horizon"]):
        print(f"{v['horizon']:>8} {v['agg']:>12.4e} "
              f"{v['uniform']:>12.4e} {v['step']:>12.4e} "
              f"{v['chirp']:>12.4e}")

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT}/results.json")


if __name__ == "__main__":
    main()
