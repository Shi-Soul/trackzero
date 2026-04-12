#!/usr/bin/env python3
"""Stage 1D.1: GPU-accelerated reachability data + ID training.

Generates single-step (s, a, s') pairs by uniformly sampling states
across the full feasible region and stepping the GPU simulator once.
This is ~1000x faster than the CPU version.

Key research question: Does uniform state-space coverage with
single-step data beat random trajectory data for closed-loop tracking?
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from trackzero.config import load_config
from trackzero.sim.gpu_simulator import GPUSimulator


def generate_reachability_gpu(
    cfg,
    n_total: int,
    n_worlds: int = 8192,
    v_range: float = 15.0,
    device: str = "cuda:0",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate single-step transitions on GPU in batches."""
    torch.manual_seed(seed)
    sim = GPUSimulator(cfg, n_worlds=n_worlds, device=device)
    tau_max = cfg.pendulum.tau_max

    all_s0 = []
    all_s1 = []
    all_a = []
    done = 0
    t0 = time.time()

    while done < n_total:
        bs = min(n_worlds, n_total - done)

        # Uniform sampling over full state-action space
        q = torch.rand(bs, 2, device=device) * 2 * np.pi - np.pi
        v = torch.rand(bs, 2, device=device) * 2 * v_range - v_range
        a = torch.rand(bs, 2, device=device) * 2 * tau_max - tau_max

        # If bs < n_worlds, pad (then trim later)
        if bs < n_worlds:
            q_full = torch.zeros(n_worlds, 2, device=device)
            v_full = torch.zeros(n_worlds, 2, device=device)
            a_full = torch.zeros(n_worlds, 2, device=device)
            q_full[:bs] = q
            v_full[:bs] = v
            a_full[:bs] = a
        else:
            q_full, v_full, a_full = q, v, a

        sim.reset_envs(q_full, v_full)
        s0_q = sim._qpos_torch[:bs].clone()
        s0_v = sim._qvel_torch[:bs].clone()

        qpos_after, qvel_after = sim.step_envs(a_full)
        s1_q = qpos_after[:bs]
        s1_v = qvel_after[:bs]

        # Store as (s0, s1) pairs
        s0 = torch.cat([s0_q, s0_v], dim=1).cpu().numpy()
        s1 = torch.cat([s1_q, s1_v], dim=1).cpu().numpy()
        act = a[:bs].cpu().numpy() if bs < n_worlds else a.cpu().numpy()

        all_s0.append(s0)
        all_s1.append(s1)
        all_a.append(act)
        done += bs

        if done % (n_worlds * 50) == 0 or done >= n_total:
            elapsed = time.time() - t0
            rate = done / elapsed
            print(f"  Generated {done:,}/{n_total:,} ({rate:.0f}/s)")

    s0_all = np.concatenate(all_s0)
    s1_all = np.concatenate(all_s1)
    a_all = np.concatenate(all_a)

    # Pack into trajectory format: states (N, 2, 4), actions (N, 1, 2)
    states = np.stack([s0_all, s1_all], axis=1)  # (N, 2, 4)
    actions = a_all[:, np.newaxis, :]  # (N, 1, 2)

    print(f"  Total time: {time.time()-t0:.1f}s")
    return states, actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medium.yaml")
    parser.add_argument("--val-data", default="data/medium/test.h5")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-transitions", type=int, default=5_000_000)
    parser.add_argument("--v-range", type=float, default=15.0)
    parser.add_argument("--n-worlds", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-hidden", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-trajectories", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== Stage 1D.1: Reachability-guided data (GPU) ===")
    print(f"n_transitions={args.n_transitions:,}, v_range=±{args.v_range}")

    # Generate data
    t0 = time.time()
    states, actions = generate_reachability_gpu(
        cfg, args.n_transitions,
        n_worlds=args.n_worlds, v_range=args.v_range,
        device=args.device, seed=args.seed,
    )
    gen_time = time.time() - t0

    # Velocity stats
    v_mag = np.sqrt(states[:, 0, 2]**2 + states[:, 0, 3]**2)
    print(f"Velocity: mean_mag={v_mag.mean():.2f}, max={v_mag.max():.2f}")

    # Load val data and train
    from trackzero.data.dataset import TrajectoryDataset
    from trackzero.policy.mlp import MLPPolicy, load_checkpoint
    from trackzero.policy.train import TrainingConfig, train
    from trackzero.eval.harness import EvalHarness

    print("Loading validation data...")
    val_ds = TrajectoryDataset(args.val_data)
    val_states = val_ds.get_all_states()
    val_actions = val_ds.get_all_actions()
    val_ds.close()

    print(f"=== Training ID model ===")
    tcfg = TrainingConfig(
        hidden_dim=args.hidden_dim, n_hidden=args.n_hidden,
        batch_size=args.batch_size, lr=args.lr,
        epochs=args.epochs, seed=args.seed, output_dir=str(out),
    )
    model, logs = train(
        states, actions, val_states, val_actions,
        cfg=tcfg, tau_max=cfg.pendulum.tau_max, device=args.device,
    )

    # Evaluate
    print("Evaluating...")
    best_model = load_checkpoint(out / "best_model.pt", device=args.device)
    policy = MLPPolicy(best_model, tau_max=cfg.pendulum.tau_max,
                       device=args.device)
    harness = EvalHarness(cfg)
    summary = harness.evaluate_policy(
        policy, val_states, val_actions,
        max_trajectories=args.eval_trajectories,
    )
    summary.to_json(out / "eval_results.json")

    result_mse = summary.mean_mse_total
    print(f"\nReachability result: mean_mse={result_mse:.6e}")
    print(f"Compare: random_matched mean_mse=1.185e-4")

    meta = {
        "method": "reachability_gpu",
        "n_transitions": args.n_transitions,
        "v_range": args.v_range,
        "gen_time_s": gen_time,
        "mean_mse_total": result_mse,
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
