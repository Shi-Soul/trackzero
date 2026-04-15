#!/usr/bin/env python3
"""Stage 2B: CEM Model-Predictive Control as planning ceiling.

Compares online trajectory optimization (CEM-MPC) against the learned
residual-PD policy on the standard benchmark. CEM has access to the
full simulator (perfect model) at test time, while the learned policy
runs in a single forward pass.

This establishes the quality ceiling for any model-based approach that
uses the same dynamics knowledge but with online computation.
"""

import json
import sys
import time
import numpy as np
import torch

sys.path.insert(0, ".")
from trackzero.config import Config
from trackzero.sim.simulator import Simulator
from trackzero.data.ood_references import generate_ood_reference_data
from trackzero.data.dataset import TrajectoryDataset

# ── CEM MPC parameters ──────────────────────────────────────────────
HORIZON = 5        # planning horizon (timesteps)
POP_SIZE = 64      # samples per iteration
ELITE_FRAC = 0.15  # top fraction for distribution update
CEM_ITERS = 3      # CEM optimization iterations
SMOOTHING = 0.8    # exponential smoothing for mean update

# ── Benchmark parameters ────────────────────────────────────────────
N_PER_FAMILY = 10   # trajectories per family (CEM is slow)
BENCHMARK_SEED = 12345
FAMILIES = ["multisine", "chirp", "step", "random_walk", "sawtooth", "pulse"]


def cem_mpc_step(sim: Simulator, state: np.ndarray, ref_states: np.ndarray,
                 horizon: int, tau_max: float, vel_weight: float = 0.1):
    """One step of CEM-MPC: find the best action for the current state.

    Args:
        sim: MuJoCo simulator (used as perfect forward model)
        state: current state (nq+nv,)
        ref_states: future reference states (T_remaining, nq+nv)
        horizon: planning horizon
        tau_max: torque limit
        vel_weight: velocity weight in MSE

    Returns:
        best_action: (nu,) optimal first action
    """
    nq = sim._nq
    nu = sim._nu
    H = min(horizon, len(ref_states))

    # Initialize action distribution
    mean = np.zeros((H, nu))
    std = np.ones((H, nu)) * tau_max * 0.5

    n_elite = max(1, int(POP_SIZE * ELITE_FRAC))

    for it in range(CEM_ITERS):
        # Sample action sequences
        actions = np.random.randn(POP_SIZE, H, nu) * std[None] + mean[None]
        actions = np.clip(actions, -tau_max, tau_max)

        # Evaluate each sequence
        costs = np.zeros(POP_SIZE)
        for i in range(POP_SIZE):
            sim.set_state(state)
            cost = 0.0
            for t in range(H):
                s_next = sim.step(actions[i, t])
                if t < len(ref_states):
                    dq = s_next[:nq] - ref_states[t, :nq]
                    dv = s_next[nq:] - ref_states[t, nq:]
                    cost += np.sum(dq**2) + vel_weight * np.sum(dv**2)
            costs[i] = cost

        # Select elites
        elite_idx = np.argsort(costs)[:n_elite]
        elite_actions = actions[elite_idx]

        # Update distribution
        new_mean = elite_actions.mean(axis=0)
        new_std = elite_actions.std(axis=0) + 1e-6
        mean = SMOOTHING * mean + (1 - SMOOTHING) * new_mean
        std = SMOOTHING * std + (1 - SMOOTHING) * new_std

    return mean[0]


def evaluate_cem_trajectory(cfg, ref_states, ref_actions):
    """Evaluate CEM-MPC on a single reference trajectory.

    Args:
        cfg: Config
        ref_states: (T+1, sd) reference states
        ref_actions: (T, nu) reference actions (for oracle comparison only)

    Returns:
        mse: tracking MSE for this trajectory
    """
    sim = Simulator(cfg)
    nq = sim._nq
    tau_max = cfg.pendulum.tau_max
    T = len(ref_actions)

    # Start from reference initial state
    sim.set_state(ref_states[0])
    actual_states = [ref_states[0].copy()]
    total_mse = 0.0

    for t in range(T):
        state = sim.get_state()
        remaining_refs = ref_states[t+1:]  # future reference states
        action = cem_mpc_step(sim, state, remaining_refs, HORIZON, tau_max)
        sim.set_state(state)  # reset before actual step
        next_state = sim.step(action)
        actual_states.append(next_state.copy())

        # Tracking error
        dq = next_state[:nq] - ref_states[t+1, :nq]
        dv = next_state[nq:] - ref_states[t+1, nq:]
        total_mse += np.sum(dq**2) + 0.1 * np.sum(dv**2)

    return total_mse / T


def evaluate_learned_policy(cfg, model_path, ref_states, ref_actions, device="cpu"):
    """Evaluate the residual-PD policy on a trajectory."""
    from trackzero.eval.harness import EvalHarness

    harness = EvalHarness(cfg)

    # Load model
    sd = cfg.pendulum.nq + cfg.pendulum.nv
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    # Build residual PD model
    from scripts.run_arch_comparison import ResidualPD
    nq = cfg.pendulum.nq
    model = ResidualPD(sd, nq, 1024, 6).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Normalize stats
    stats_path = model_path.replace(".pt", "_stats.npz")

    class PolicyWrapper:
        def __init__(self, model, nq, device):
            self.model = model
            self.nq = nq
            self.device = device
            self.s_mean = None
            self.s_std = None

        def __call__(self, state, ref_next):
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=self.device)
                r = torch.tensor(ref_next, dtype=torch.float32, device=self.device)
                if self.s_mean is not None:
                    s = (s - self.s_mean) / self.s_std
                    r = (r - self.s_mean) / self.s_std
                x = torch.cat([s, r]).unsqueeze(0)
                return self.model(x).squeeze(0).cpu().numpy()

    policy = PolicyWrapper(model, nq, device)
    result = harness.evaluate_trajectory(policy, ref_states, ref_actions)
    return result.mse_total


def generate_benchmark():
    """Generate benchmark reference trajectories."""
    cfg = Config()
    families = {}

    # Multisine from test set
    ds = TrajectoryDataset("data/medium/test.h5")
    all_s, all_a = ds.get_all_states(), ds.get_all_actions()
    ds.close()
    families["multisine"] = (all_s[:N_PER_FAMILY], all_a[:N_PER_FAMILY])

    # Other families
    for name in FAMILIES[1:]:
        s, a = generate_ood_reference_data(
            cfg, N_PER_FAMILY, action_type=name, seed=BENCHMARK_SEED
        )
        families[name] = (s, a)

    return cfg, families


def main():
    print("=" * 60)
    print("Stage 2B: CEM-MPC vs Learned Policy Comparison")
    print(f"  Horizon={HORIZON}, Pop={POP_SIZE}, Elite={ELITE_FRAC}")
    print(f"  CEM iters={CEM_ITERS}, N_per_family={N_PER_FAMILY}")
    print("=" * 60)

    cfg, families = generate_benchmark()
    results = {"cem": {}, "meta": {
        "horizon": HORIZON, "pop_size": POP_SIZE,
        "elite_frac": ELITE_FRAC, "cem_iters": CEM_ITERS,
        "n_per_family": N_PER_FAMILY
    }}

    total_cem = []
    t0 = time.time()

    for fname in FAMILIES:
        ref_s, ref_a = families[fname]
        n = min(N_PER_FAMILY, len(ref_s))
        family_mses = []

        print(f"\n--- {fname} ({n} trajectories) ---")
        for i in range(n):
            t1 = time.time()
            mse = evaluate_cem_trajectory(cfg, ref_s[i], ref_a[i])
            dt = time.time() - t1
            family_mses.append(mse)
            print(f"  [{i+1}/{n}] MSE={mse:.6e} ({dt:.1f}s)")

        mean_mse = np.mean(family_mses)
        results["cem"][fname] = float(mean_mse)
        total_cem.extend(family_mses)
        print(f"  {fname} mean: {mean_mse:.6e}")

    agg_cem = float(np.mean(total_cem))
    results["cem"]["AGGREGATE"] = agg_cem
    results["meta"]["total_time_s"] = time.time() - t0

    print(f"\n{'='*60}")
    print(f"CEM-MPC AGGREGATE: {agg_cem:.6e}")
    print(f"Total time: {time.time()-t0:.0f}s")
    print(f"{'='*60}")

    # Save
    import os
    os.makedirs("outputs/cem_mpc", exist_ok=True)
    with open("outputs/cem_mpc/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to outputs/cem_mpc/results.json")


if __name__ == "__main__":
    main()
