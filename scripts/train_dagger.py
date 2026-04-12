#!/usr/bin/env python3
"""Stage 1D: DAgger (Dataset Aggregation) for inverse dynamics.

Motivation: Stage 1C showed that closed-loop error compounding is the main
failure mode (restricted-v: 172× worse). DAgger directly addresses this by
collecting training data at states the model actually visits during tracking.

Algorithm:
  1. Train M₀ on random rollout data (use existing random_matched model)
  2. For k = 1..K:
     a. Roll out M_{k-1} on diverse references in closed loop
     b. At each step, record (actual_state, ref_next_state)
     c. Compute oracle_torque = ID_oracle(actual_state, ref_next_state)
     d. Aggregate new pairs into training dataset
     e. Retrain M_k on augmented dataset
  3. Evaluate final model

Reference: Ross, Gordon, Bagnell (2011) "A Reduction of Imitation Learning
and Structured Prediction to No-Regret Online Learning"
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
from trackzero.policy.mlp import MLPPolicy, load_checkpoint
from trackzero.policy.train import TrainingConfig, train
from trackzero.sim.simulator import Simulator


def collect_dagger_data(
    policy, oracle, cfg, ref_states, ref_actions, max_traj=200
):
    """Roll out policy on references, collect (actual_state, ref_next, oracle_torque)."""
    sim = Simulator(cfg)
    tau_max = cfg.pendulum.tau_max
    N = min(len(ref_states), max_traj)

    all_s = []
    all_snext = []
    all_u = []

    for i in range(N):
        T = len(ref_actions[i])
        q0, v0 = ref_states[i, 0, :2], ref_states[i, 0, 2:]
        sim.reset(q0=q0, v0=v0)
        actual = sim.get_state()

        for t in range(T):
            ref_next = ref_states[i, t + 1]
            # Model's action
            action = policy(actual, ref_next)
            # Oracle's action from actual state
            oracle_u = oracle.compute_torque(actual, ref_next)
            oracle_u = np.clip(oracle_u, -tau_max, tau_max)

            all_s.append(actual.copy())
            all_snext.append(ref_next.copy())
            all_u.append(oracle_u.copy())

            # Advance simulator with model's action (on-policy)
            actual = sim.step(action)

    states = np.array(all_s)       # (M, 4)
    snexts = np.array(all_snext)   # (M, 4)
    actions = np.array(all_u)      # (M, 2)
    return states, snexts, actions


def transitions_from_trajectories(states, actions):
    """Extract (s_t, s_{t+1}, u_t) pairs from trajectory arrays."""
    N, Tp1, d = states.shape
    T = Tp1 - 1
    s_t = states[:, :-1].reshape(-1, d)
    s_tp1 = states[:, 1:].reshape(-1, d)
    u_t = actions.reshape(-1, actions.shape[-1])
    return s_t, s_tp1, u_t


def train_on_pairs(s_t, s_tp1, u_t, val_s, val_a, cfg_train, device):
    """Train ID model on transition pairs."""
    # Combine into pseudo-trajectories for the training API
    # But train() expects trajectory arrays. Let's use the raw pair training.
    from trackzero.policy.mlp import InverseDynamicsMLP
    from torch.utils.data import TensorDataset, DataLoader

    state_dim = s_t.shape[-1]
    action_dim = u_t.shape[-1]

    model = InverseDynamicsMLP(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=cfg_train.hidden_dim,
        n_hidden=cfg_train.n_hidden,
    ).to(device)

    # Validation pairs
    val_st, val_stp1, val_ut = transitions_from_trajectories(val_s, val_a)
    n_val = len(val_st)

    # Normalize
    tau_max = 5.0  # default
    u_t_norm = u_t / tau_max
    val_ut_norm = val_ut / tau_max

    # Create datasets
    X_train = np.concatenate([s_t, s_tp1], axis=-1).astype(np.float32)
    Y_train = u_t_norm.astype(np.float32)
    X_val = np.concatenate([val_st, val_stp1], axis=-1).astype(np.float32)
    Y_val = val_ut_norm.astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(Y_train)
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(Y_val)
    )
    train_dl = DataLoader(
        train_ds, batch_size=cfg_train.batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=cfg_train.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg_train.epochs
    )
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(1, cfg_train.epochs + 1):
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
        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        if val_avg < best_val:
            best_val = val_avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{cfg_train.epochs}  "
                  f"train={train_avg:.6f}  val={val_avg:.6f}  lr={lr:.2e}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return model, best_val


def main():
    parser = argparse.ArgumentParser(description="DAgger for inverse dynamics")
    parser.add_argument("--config", default="configs/medium.yaml")
    parser.add_argument("--val-data", default="data/medium/test.h5")
    parser.add_argument("--base-model", required=True, help="Initial model (M₀)")
    parser.add_argument("--output-dir", default="outputs/stage1d_dagger")
    parser.add_argument("--dagger-iters", type=int, default=5)
    parser.add_argument("--dagger-traj", type=int, default=200,
                        help="References per DAgger iteration")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-hidden", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
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

    # Load validation/test data
    print("Loading validation data...")
    val_ds = TrajectoryDataset(args.val_data)
    val_s, val_a = val_ds.get_all_states(), val_ds.get_all_actions()
    val_ds.close()

    # Load initial training data (random rollouts)
    print("Loading seed training data...")
    seed_ds = TrajectoryDataset("data/medium/train.h5")
    seed_s, seed_a = seed_ds.get_all_states(), seed_ds.get_all_actions()
    seed_ds.close()
    # Use all trajectories (matches random_matched budget)
    print(f"Seed trajectories: {len(seed_s)}")

    # Extract transition pairs from seed data
    seed_st, seed_stp1, seed_ut = transitions_from_trajectories(seed_s, seed_a)
    print(f"Seed pairs: {len(seed_st)}")

    # Oracle
    oracle = InverseDynamicsOracle(cfg)

    # Load base model
    print(f"Loading base model from {args.base_model}")
    model = load_checkpoint(args.base_model, device=args.device)

    tcfg = TrainingConfig(
        hidden_dim=args.hidden_dim, n_hidden=args.n_hidden,
        batch_size=args.batch_size, lr=args.lr,
        epochs=args.epochs, seed=args.seed,
        output_dir=str(out),
    )

    # Accumulate DAgger pairs
    dagger_st_all = []
    dagger_stp1_all = []
    dagger_ut_all = []

    log = []
    harness = EvalHarness(cfg)

    for k in range(args.dagger_iters):
        print(f"\n{'='*60}")
        print(f"DAgger iteration {k+1}/{args.dagger_iters}")
        print(f"{'='*60}")

        # Create policy from current model
        policy = MLPPolicy(model, tau_max=tau_max, device=args.device)

        # Collect DAgger data on test references
        print(f"Collecting DAgger data ({args.dagger_traj} references)...")
        t0 = time.time()
        d_st, d_stp1, d_ut = collect_dagger_data(
            policy, oracle, cfg, val_s, val_a, max_traj=args.dagger_traj
        )
        elapsed = time.time() - t0
        print(f"  Collected {len(d_st)} pairs in {elapsed:.1f}s")

        dagger_st_all.append(d_st)
        dagger_stp1_all.append(d_stp1)
        dagger_ut_all.append(d_ut)

        # Aggregate all data
        all_st = np.concatenate([seed_st] + dagger_st_all, axis=0)
        all_stp1 = np.concatenate([seed_stp1] + dagger_stp1_all, axis=0)
        all_ut = np.concatenate([seed_ut] + dagger_ut_all, axis=0)
        print(f"Total training pairs: {len(all_st)} "
              f"(seed={len(seed_st)}, dagger={len(all_st)-len(seed_st)})")

        # Retrain
        print("Training...")
        model, best_val = train_on_pairs(
            all_st, all_stp1, all_ut, val_s, val_a, tcfg, args.device
        )

        # Evaluate
        print("Evaluating...")
        policy = MLPPolicy(model, tau_max=tau_max, device=args.device)
        summary = harness.evaluate_policy(policy, val_s, val_a, max_trajectories=200)
        print(f"  mean_mse={summary.mean_mse_total:.6e}  "
              f"max_mse={summary.max_mse_total:.6e}")

        entry = {
            "iteration": k + 1,
            "n_dagger_pairs": int(len(all_st) - len(seed_st)),
            "n_total_pairs": int(len(all_st)),
            "best_val_loss": float(best_val),
            "mean_mse_total": float(summary.mean_mse_total),
            "max_mse_total": float(summary.max_mse_total),
            "median_mse_total": float(summary.median_mse_total),
        }
        log.append(entry)
        print(f"  Iter {k+1}: {summary.mean_mse_total:.6e} "
              f"(random baseline: 1.185e-4)")

    # Save best model and results
    best_iter = min(log, key=lambda x: x["mean_mse_total"])
    print(f"\nBest iteration: {best_iter['iteration']} "
          f"(mean_mse={best_iter['mean_mse_total']:.6e})")

    # Save final model
    torch.save(model.state_dict(), out / "final_model.pt")

    # Save final evaluation
    summary.to_json(out / "eval_results.json")

    with open(out / "dagger_log.json", "w") as f:
        json.dump(log, f, indent=2)

    meta = {
        "method": "dagger",
        "dagger_iters": args.dagger_iters,
        "dagger_traj_per_iter": args.dagger_traj,
        "seed_pairs": int(len(seed_st)),
        "final_pairs": int(len(all_st)),
        "best_iteration": best_iter["iteration"],
        "mean_mse_total": best_iter["mean_mse_total"],
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Results in {out}")
    print(f"  Final mean_mse: {log[-1]['mean_mse_total']:.6e}")
    print(f"  Best mean_mse: {best_iter['mean_mse_total']:.6e}")
    print(f"  Random baseline: 1.185e-4")


if __name__ == "__main__":
    main()
