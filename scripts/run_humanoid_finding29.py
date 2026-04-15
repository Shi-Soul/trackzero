#!/usr/bin/env python3
"""Finding 29: Can TRACK-ZERO balance a humanoid upright?

The current benchmark (random chaos trajectories) is not representative
of practical tracking. This experiment tests whether no_bench inverse
dynamics can STABILIZE the humanoid near its upright equilibrium — a
prerequisite for any useful tracking policy.

Protocol:
  1. Train no_bench model (same 2K traj as Finding 24/28)
  2. Reference = humanoid initial upright pose (constant)
  3. Apply model for N steps: model(current, ref_upright) → torques
  4. Measure: height over time, steps before fall (h < 0.4m)
  5. Compare: zero_torque, random_only, no_bench, oracle_matched

Balance is defined as: height >= 0.4m for T steps.
Reference for scale: humanoid starts at h=1.4m, threshold 0.4m.

Usage:
    CUDA_VISIBLE_DEVICES=7 python -m scripts.run_humanoid_finding29
"""
import json, os, time
import mujoco
import numpy as np
import torch
import torch.nn as nn

from scripts.run_humanoid import (
    HUMANOID_XML, N_BODY_GEOMS, GEOM_OFFSET, TAU_MAX,
    mj_to_flat, flat_to_mj, get_contact_flags,
)
from scripts.run_coverage_ablation import gen_data_from_patterns
from scripts.models_structured import _build_mlp
from scripts.run_structured import train_model

FLAT_STATE_DIM = 54
TAU_MAX_NP = np.array(TAU_MAX, dtype=np.float32)
OUT = "outputs/humanoid_finding29"
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HEIGHT_THRESHOLD = 0.4   # fall if torso height < 0.4m
MAX_STEPS = 1000         # maximum steps to test balance
N_TRIALS = 20            # number of trials per config

CONFIGS = {
    "zero_torque": None,       # applies zero torques always
    "random_only": ["white"],
    "no_bench":    ["white", "ou", "brownian", "sine", "bang_bang", "ramp"],
    "oracle_matched": ["step", "chirp"],
}


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


def get_torso_height(d):
    """Return torso (root body) Z position."""
    return float(d.qpos[2])


def get_upright_reference(m, d):
    """Get the initial upright pose as a flat state vector."""
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    return mj_to_flat(d.qpos.copy(), d.qvel.copy())


def run_balance_trial(model_nn, seed, dev):
    """Run one balance trial.

    Returns:
        steps_upright: how many steps the humanoid stayed up
        height_trace: list of heights over time
    """
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    rng = np.random.RandomState(seed)

    # Get upright reference pose
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    ref_upright = mj_to_flat(d.qpos.copy(), d.qvel.copy())

    # Add small initial perturbation to test robustness
    mujoco.mj_resetData(m, d)
    d.qpos[7:] += rng.randn(m.nq - 7) * 0.02  # tiny joint noise
    d.qvel[:] += rng.randn(m.nv) * 0.05        # tiny velocity noise
    mujoco.mj_forward(m, d)

    height_trace = []
    for t in range(MAX_STEPS):
        h = get_torso_height(d)
        height_trace.append(h)

        if h < HEIGHT_THRESHOLD or np.any(np.isnan(d.qpos)):
            return t, height_trace

        cur_s = mj_to_flat(d.qpos.copy(), d.qvel.copy())
        flags = get_contact_flags(d)
        inp = np.concatenate([cur_s, ref_upright, flags]).astype(np.float32)

        if model_nn is None:
            tau = np.zeros(m.nu, dtype=np.float32)
        else:
            with torch.no_grad():
                tau = model_nn(
                    torch.tensor(inp).unsqueeze(0).to(dev)
                ).cpu().numpy()[0]

        d.ctrl[:] = np.clip(tau, -TAU_MAX_NP, TAU_MAX_NP)
        mujoco.mj_step(m, d)

    return MAX_STEPS, height_trace


def train_model_config(patterns):
    S, SN, A, F = gen_data_from_patterns(patterns, 2000, 500, seed=42)
    in_dim = 2 * FLAT_STATE_DIM + N_BODY_GEOMS
    X = np.concatenate([S, SN, F], axis=1)
    model = PlainMLP(in_dim, len(TAU_MAX)).to(device)
    train_model(model, X, A, epochs=200, bs=2048, device=device)
    model.eval()
    return model


def main():
    print("=" * 60)
    print("FINDING 29: Can TRACK-ZERO balance a humanoid upright?")
    print("=" * 60)

    results = {}

    for cfg_name, patterns in CONFIGS.items():
        print(f"\n{'='*50}")
        print(f"Config: {cfg_name}")

        if patterns is None:
            model = None
            print("  Using zero torques (no model)")
        else:
            t0 = time.time()
            model = train_model_config(patterns)
            print(f"  Trained in {time.time() - t0:.0f}s")

        trial_steps = []
        for trial in range(N_TRIALS):
            steps, trace = run_balance_trial(model, seed=trial * 7, dev=device)
            trial_steps.append(steps)
            if (trial + 1) % 5 == 0:
                print(f"  Trials {trial+1}/{N_TRIALS}: "
                      f"mean={np.mean(trial_steps):.0f} steps, "
                      f"max={max(trial_steps)} steps")

        mean_steps = float(np.mean(trial_steps))
        std_steps = float(np.std(trial_steps))
        n_survived = sum(1 for s in trial_steps if s == MAX_STEPS)

        print(f"  RESULT: {mean_steps:.0f} ± {std_steps:.0f} steps upright "
              f"({n_survived}/{N_TRIALS} survived {MAX_STEPS} steps)")

        results[cfg_name] = {
            "mean_steps": mean_steps,
            "std_steps": std_steps,
            "n_survived_full": n_survived,
            "n_trials": N_TRIALS,
            "trial_steps": trial_steps,
        }

    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT}/results.json")

    print("\n" + "=" * 60)
    print("SUMMARY: Steps upright before fall (max=1000)")
    print(f"{'Config':<18s}  {'Mean±Std':>15s}  {'Survived':>8s}")
    for name, r in results.items():
        surv = f"{r['n_survived_full']}/{r['n_trials']}"
        print(f"{name:<18s}  "
              f"{r['mean_steps']:>7.0f}±{r['std_steps']:<7.0f}  {surv:>8s}")


if __name__ == "__main__":
    main()
