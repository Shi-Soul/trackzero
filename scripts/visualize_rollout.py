#!/usr/bin/env python3
"""Visualize closed-loop policy rollouts vs reference trajectory.

For each selected policy the script:
  1. Runs a closed-loop rollout starting from the reference initial state.
  2. Plots all state traces (q1, q2, dq1, dq2) overlaid with the reference.
  3. Optionally produces a GIF animation of the double pendulum.

Usage examples
--------------
# Compare all three Stage-0 policies on trajectory #5:
uv run python scripts/visualize_rollout.py \\
    --dataset data/test.h5 \\
    --trajectory-idx 5 \\
    --policies oracle supervised zero \\
    --checkpoint-1a outputs/stage1a/best_model.pt \\
    --output-dir outputs/rollout_viz

# Same but also save animated GIF:
uv run python scripts/visualize_rollout.py \\
    --dataset data/test.h5 --trajectory-idx 5 \\
    --policies oracle supervised zero \\
    --checkpoint-1a outputs/stage1a/best_model.pt \\
    --animate --output-dir outputs/rollout_viz

# Stage 1B model too:
uv run python scripts/visualize_rollout.py \\
    --dataset data/test.h5 --trajectory-idx 5 \\
    --policies oracle supervised_1a supervised_1b zero \\
    --checkpoint-1a outputs/stage1a/best_model.pt \\
    --checkpoint-1b outputs/stage1b/best_model.pt \\
    --animate --output-dir outputs/rollout_viz
"""

import argparse
from pathlib import Path

import numpy as np

from trackzero.config import load_config
from trackzero.data.dataset import TrajectoryDataset
from trackzero.eval.harness import EvalHarness
from trackzero.oracle.inverse_dynamics import InverseDynamicsOracle
from trackzero.sim.simulator import Simulator
from trackzero.viz.playback import animate_pendulum
from trackzero.viz.plots import plot_trajectory_comparison


def _rollout(policy, ref_states: np.ndarray, cfg) -> np.ndarray:
    """Run one closed-loop rollout; return actual_states (T+1, 4)."""
    sim = Simulator(cfg)
    T = len(ref_states) - 1
    sim.reset(q0=ref_states[0, :2], v0=ref_states[0, 2:])
    actual = np.zeros_like(ref_states)
    actual[0] = sim.get_state()
    for t in range(T):
        action = policy(actual[t], ref_states[t + 1])
        actual[t + 1] = sim.step(action)
    return actual


def main():
    parser = argparse.ArgumentParser(
        description="Visualize closed-loop rollouts vs reference trajectory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default="data/test.h5")
    parser.add_argument("--trajectory-idx", type=int, default=0,
                        help="Which trajectory from the dataset to visualise")
    parser.add_argument("--policies", type=str, nargs="+",
                        default=["oracle", "supervised", "zero"],
                        choices=["oracle", "zero", "supervised_1a", "supervised_1b",
                                 "supervised",   # alias for supervised_1a
                                 ],
                        help="Policies to visualise (can list multiple)")
    parser.add_argument("--oracle-mode", choices=["shooting", "finite_difference"],
                        default="shooting")
    parser.add_argument("--checkpoint-1a", type=str,
                        default="outputs/stage1a/best_model.pt")
    parser.add_argument("--checkpoint-1b", type=str,
                        default="outputs/stage1b/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/rollout_viz")
    parser.add_argument("--animate", action="store_true",
                        help="Also save a GIF animation (slow for long trajectories)")
    parser.add_argument("--animate-skip", type=int, default=2,
                        help="Show every N-th frame in the animation")
    parser.add_argument("--animate-fps", type=int, default=30)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the chosen reference trajectory
    print(f"Loading trajectory #{args.trajectory_idx} from {args.dataset}")
    ds = TrajectoryDataset(args.dataset)
    ref_states, ref_actions = ds[args.trajectory_idx]
    ds.close()
    print(f"  T = {len(ref_actions)} steps  ({len(ref_actions) * cfg.simulation.control_dt:.1f} s)")

    # Build policies
    POLICY_LABELS = {
        "oracle":        f"Oracle ({args.oracle_mode})",
        "zero":          "Zero torque",
        "supervised":    "Supervised 1A",
        "supervised_1a": "Supervised 1A",
        "supervised_1b": "Supervised 1B",
    }

    policies = {}
    for name in args.policies:
        if name == "oracle":
            oracle = InverseDynamicsOracle(cfg)
            policies[name] = (POLICY_LABELS[name], oracle.as_policy(mode=args.oracle_mode))

        elif name == "zero":
            policies[name] = (POLICY_LABELS[name], lambda s, ns: np.zeros(2))

        elif name in ("supervised", "supervised_1a"):
            ckpt = args.checkpoint_1a
            if not Path(ckpt).exists():
                print(f"  WARNING: checkpoint not found: {ckpt} — skipping {name}")
                continue
            from trackzero.policy.mlp import MLPPolicy, load_checkpoint
            model = load_checkpoint(ckpt)
            policies[name] = (POLICY_LABELS[name], MLPPolicy(model, tau_max=cfg.pendulum.tau_max))

        elif name == "supervised_1b":
            ckpt = args.checkpoint_1b
            if not Path(ckpt).exists():
                print(f"  WARNING: checkpoint not found: {ckpt} — skipping {name}")
                continue
            from trackzero.policy.mlp import MLPPolicy, load_checkpoint
            model = load_checkpoint(ckpt)
            policies[name] = (POLICY_LABELS[name], MLPPolicy(model, tau_max=cfg.pendulum.tau_max))

    if not policies:
        print("No valid policies specified. Exiting.")
        return

    # Run rollouts
    policy_rollouts = []
    for name, (label, policy) in policies.items():
        print(f"  Rolling out: {label} ...", end=" ", flush=True)
        actual = _rollout(policy, ref_states, cfg)
        policy_rollouts.append((label, actual))

        # Quick error summary
        q_err = np.mean((ref_states[1:, :2] - actual[1:, :2]) ** 2)
        print(f"MSE_q = {q_err:.4e}")

    # --- State time-series plot ---
    idx = args.trajectory_idx
    trace_path = output_dir / f"traj{idx}_state_traces.png"
    plot_trajectory_comparison(
        ref_states, policy_rollouts,
        control_dt=cfg.simulation.control_dt,
        save_path=trace_path,
    )
    print(f"\nState trace plot saved: {trace_path}")

    # --- GIF animation per policy (+ reference overlay) ---
    if args.animate:
        for name, (label, _) in policies.items():
            _, actual = next(r for r in policy_rollouts if r[0] == label)
            gif_path = output_dir / f"traj{idx}_{name}.gif"
            print(f"  Saving animation: {gif_path}  (this may take ~30 s) ...", end=" ", flush=True)
            animate_pendulum(
                actual,
                control_dt=cfg.simulation.control_dt,
                link_length=cfg.pendulum.link_length,
                ref_states=ref_states,
                save_path=gif_path,
                fps=args.animate_fps,
                skip=args.animate_skip,
            )
            print("done")
        print("Blue = actual policy  |  Red (dashed) = reference")


if __name__ == "__main__":
    main()
