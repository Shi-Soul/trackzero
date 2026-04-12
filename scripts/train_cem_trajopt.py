#!/usr/bin/env python3
"""Stage 1D.2: CEM Trajectory Optimization as Data Generator.

Idea: Use Cross-Entropy Method (CEM) with the known simulator to generate
diverse goal-reaching trajectories. Unlike reachability (1D.1) which failed
because single-step data lacks trajectory coherence, CEM produces full
trajectories with temporal coherence.

Algorithm:
  1. Sample random goal states from the feasible region
  2. For each goal, run CEM to find a torque sequence that reaches it
  3. Successful trajectories become training data for inverse dynamics
  4. Goals are chosen to cover the state space uniformly

This tests whether intelligent trajectory generation (not just random rollouts)
can improve inverse dynamics learning.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from trackzero.config import load_config
from trackzero.data.dataset import TrajectoryDataset
from trackzero.eval.harness import EvalHarness
from trackzero.oracle import InverseDynamicsOracle
from trackzero.policy.mlp import MLPPolicy, InverseDynamicsMLP
from trackzero.sim.simulator import Simulator


def cem_reach_goal(
    sim: Simulator,
    q0: np.ndarray,
    v0: np.ndarray,
    goal_state: np.ndarray,
    tau_max: float,
    horizon: int = 100,
    n_samples: int = 200,
    n_elite: int = 40,
    n_iters: int = 5,
    smoothing: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """CEM to find torque sequence reaching goal_state from (q0, v0).

    Returns: (states, actions, final_distance)
        states: (horizon+1, 4)
        actions: (horizon, 2)
        final_distance: L2 distance to goal at end
    """
    nq = len(q0)
    # Initialize mean/std for torque sequence
    mean = np.zeros((horizon, nq))
    std = np.ones((horizon, nq)) * tau_max * 0.5

    best_actions = None
    best_dist = float("inf")
    best_states = None

    for it in range(n_iters):
        # Sample torque sequences
        samples = np.random.randn(n_samples, horizon, nq) * std + mean
        samples = np.clip(samples, -tau_max, tau_max)

        # Evaluate each sample
        dists = np.full(n_samples, float("inf"))
        all_states = []

        for j in range(n_samples):
            sim.reset(q0=q0, v0=v0)
            traj = [sim.get_state()]
            for t in range(horizon):
                s = sim.step(samples[j, t])
                traj.append(s)
            traj = np.array(traj)
            all_states.append(traj)
            final = traj[-1]
            dists[j] = np.linalg.norm(final - goal_state)

        # Select elite
        elite_idx = np.argsort(dists)[:n_elite]
        elite = samples[elite_idx]

        # Update distribution
        new_mean = elite.mean(axis=0)
        new_std = elite.std(axis=0) + 1e-4
        mean = smoothing * mean + (1 - smoothing) * new_mean
        std = smoothing * std + (1 - smoothing) * new_std

        # Track best
        best_j = elite_idx[0]
        if dists[best_j] < best_dist:
            best_dist = dists[best_j]
            best_actions = samples[best_j].copy()
            best_states = all_states[best_j].copy()

    return best_states, best_actions, best_dist


def sample_feasible_goals(sim, cfg, n_goals, horizon=200):
    """Sample feasible goal states by running random rollouts."""
    tau_max = cfg.pendulum.tau_max
    goals = []
    for _ in range(n_goals * 2):  # oversample
        q0 = np.random.uniform(-np.pi, np.pi, 2)
        v0 = np.random.uniform(-5, 5, 2)
        sim.reset(q0=q0, v0=v0)
        T = np.random.randint(50, horizon)
        for t in range(T):
            u = np.random.uniform(-tau_max, tau_max, 2)
            sim.step(u)
        goals.append(sim.get_state())
        if len(goals) >= n_goals:
            break
    return np.array(goals[:n_goals])


def transitions_from_trajectories_list(traj_list):
    """Extract (s_t, s_{t+1}, u_t) pairs from list of (states, actions)."""
    all_st, all_stp1, all_ut = [], [], []
    for states, actions in traj_list:
        for t in range(len(actions)):
            all_st.append(states[t])
            all_stp1.append(states[t + 1])
            all_ut.append(actions[t])
    return np.array(all_st), np.array(all_stp1), np.array(all_ut)


def train_on_pairs(s_t, s_tp1, u_t, val_s, val_a, cfg_train, device):
    """Train ID model on transition pairs (same as DAgger)."""
    from torch.utils.data import TensorDataset, DataLoader

    state_dim = s_t.shape[-1]
    action_dim = u_t.shape[-1]
    tau_max = 5.0

    model = InverseDynamicsMLP(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=cfg_train["hidden_dim"],
        n_hidden=cfg_train["n_hidden"],
    ).to(device)

    # Validation pairs
    N, Tp1, d = val_s.shape
    T = Tp1 - 1
    v_st = val_s[:, :-1].reshape(-1, d)
    v_stp1 = val_s[:, 1:].reshape(-1, d)
    v_ut = val_a.reshape(-1, val_a.shape[-1])

    u_t_norm = u_t / tau_max
    v_ut_norm = v_ut / tau_max

    X_train = np.concatenate([s_t, s_tp1], axis=-1).astype(np.float32)
    Y_train = u_t_norm.astype(np.float32)
    X_val = np.concatenate([v_st, v_stp1], axis=-1).astype(np.float32)
    Y_val = v_ut_norm.astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(Y_train)
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(Y_val)
    )
    train_dl = DataLoader(
        train_ds, batch_size=cfg_train["batch_size"], shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=cfg_train["batch_size"], shuffle=False,
        num_workers=0, pin_memory=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg_train["epochs"]
    )
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(1, cfg_train["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item()
                n_val_batches += 1

        train_avg = epoch_loss / max(n_batches, 1)
        val_avg = val_loss / max(n_val_batches, 1)
        scheduler.step()

        if val_avg < best_val:
            best_val = val_avg
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  train={train_avg:.6f}  "
                  f"val={val_avg:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return model, best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medium.yaml")
    parser.add_argument("--val-data", default="data/medium/test.h5")
    parser.add_argument("--output-dir", default="outputs/stage1d_cem")
    parser.add_argument("--n-goals", type=int, default=500)
    parser.add_argument("--cem-horizon", type=int, default=100)
    parser.add_argument("--cem-samples", type=int, default=200)
    parser.add_argument("--cem-elite", type=int, default=40)
    parser.add_argument("--cem-iters", type=int, default=5)
    parser.add_argument("--dist-threshold", type=float, default=2.0,
                        help="Max final distance to accept trajectory")
    parser.add_argument("--augment-random", action="store_true",
                        help="Augment CEM data with random rollouts")
    parser.add_argument("--n-random", type=int, default=5000,
                        help="Random rollout trajectories to augment")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-hidden", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    tau_max = cfg.pendulum.tau_max
    sim = Simulator(cfg)

    # Load validation data
    print("Loading validation data...")
    val_ds = TrajectoryDataset(args.val_data)
    val_s, val_a = val_ds.get_all_states(), val_ds.get_all_actions()
    val_ds.close()

    # Sample feasible goals
    print(f"Sampling {args.n_goals} feasible goal states...")
    goals = sample_feasible_goals(sim, cfg, args.n_goals)
    print(f"  Goal state ranges: q={goals[:,:2].min():.2f} to "
          f"{goals[:,:2].max():.2f}, "
          f"v={goals[:,2:].min():.2f} to {goals[:,2:].max():.2f}")

    # Run CEM for each goal
    print(f"\nRunning CEM trajectory optimization ({args.n_goals} goals)...")
    trajectories = []  # list of (states, actions)
    distances = []
    t0 = time.time()

    for i in range(args.n_goals):
        # Random start state
        q0 = np.random.uniform(-np.pi, np.pi, 2)
        v0 = np.random.uniform(-3, 3, 2)

        states, actions, dist = cem_reach_goal(
            sim, q0, v0, goals[i], tau_max,
            horizon=args.cem_horizon,
            n_samples=args.cem_samples,
            n_elite=args.cem_elite,
            n_iters=args.cem_iters,
        )
        distances.append(dist)

        if dist < args.dist_threshold:
            trajectories.append((states, actions))

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            n_ok = len(trajectories)
            print(f"  [{i+1}/{args.n_goals}] {n_ok} successful "
                  f"({n_ok/(i+1)*100:.0f}%), "
                  f"median_dist={np.median(distances):.3f}, "
                  f"time={elapsed:.0f}s")

    elapsed = time.time() - t0
    dists = np.array(distances)
    print(f"\nCEM complete: {len(trajectories)}/{args.n_goals} successful "
          f"({len(trajectories)/args.n_goals*100:.0f}%)")
    print(f"  Distance: mean={dists.mean():.3f}, "
          f"median={np.median(dists):.3f}, "
          f"min={dists.min():.3f}, max={dists.max():.3f}")
    print(f"  Time: {elapsed:.1f}s")

    # Extract training pairs
    cem_st, cem_stp1, cem_ut = transitions_from_trajectories_list(
        trajectories
    )
    print(f"CEM training pairs: {len(cem_st)}")

    # Optionally augment with random rollout data
    if args.augment_random:
        print(f"\nAugmenting with {args.n_random} random rollouts...")
        rand_trajs = []
        for _ in range(args.n_random):
            q0 = np.random.uniform(-np.pi, np.pi, 2)
            v0 = np.random.uniform(-5, 5, 2)
            sim.reset(q0=q0, v0=v0)
            T = 500
            ss = [sim.get_state()]
            aa = []
            for t in range(T):
                u = np.random.uniform(-tau_max, tau_max, 2)
                ss.append(sim.step(u))
                aa.append(u)
            rand_trajs.append((np.array(ss), np.array(aa)))

        r_st, r_stp1, r_ut = transitions_from_trajectories_list(rand_trajs)
        all_st = np.concatenate([cem_st, r_st])
        all_stp1 = np.concatenate([cem_stp1, r_stp1])
        all_ut = np.concatenate([cem_ut, r_ut])
        print(f"Total pairs: {len(all_st)} "
              f"(CEM={len(cem_st)}, random={len(r_st)})")
    else:
        all_st, all_stp1, all_ut = cem_st, cem_stp1, cem_ut

    # Compute oracle labels for CEM data
    print("\nComputing oracle labels...")
    oracle = InverseDynamicsOracle(cfg)
    for i in range(len(all_st)):
        all_ut[i] = oracle.compute_torque(all_st[i], all_stp1[i])
        all_ut[i] = np.clip(all_ut[i], -tau_max, tau_max)
    print("  Done.")

    # Train
    print(f"\nTraining on {len(all_st)} pairs...")
    tcfg = {
        "hidden_dim": args.hidden_dim,
        "n_hidden": args.n_hidden,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
    }
    model, best_val = train_on_pairs(
        all_st, all_stp1, all_ut, val_s, val_a, tcfg, args.device
    )

    # Evaluate
    print("\nEvaluating...")
    policy = MLPPolicy(model, tau_max=tau_max, device=args.device)
    harness = EvalHarness(cfg)
    summary = harness.evaluate_policy(policy, val_s, val_a, max_trajectories=200)
    print(f"  mean_mse={summary.mean_mse_total:.6e}  "
          f"max_mse={summary.max_mse_total:.6e}")
    print(f"  (Random baseline: 1.185e-4)")

    # Save
    torch.save(model.state_dict(), out / "best_model.pt")
    summary.to_json(out / "eval_results.json")

    meta = {
        "method": "cem_trajopt",
        "n_goals": args.n_goals,
        "n_successful": len(trajectories),
        "cem_horizon": args.cem_horizon,
        "cem_pairs": int(len(cem_st)),
        "total_pairs": int(len(all_st)),
        "augment_random": args.augment_random,
        "best_val_loss": float(best_val),
        "mean_mse_total": float(summary.mean_mse_total),
        "max_mse_total": float(summary.max_mse_total),
        "distance_stats": {
            "mean": float(dists.mean()),
            "median": float(np.median(dists)),
            "min": float(dists.min()),
            "max": float(dists.max()),
        },
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Results in {out}")
    ratio = summary.mean_mse_total / 1.185e-4
    print(f"  Ratio vs random: {ratio:.1f}×")


if __name__ == "__main__":
    main()
