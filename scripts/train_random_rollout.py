#!/usr/bin/env python3
"""Train inverse dynamics from random rollout data (Stage 1B)."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from trackzero.config import load_config
from trackzero.data.dataset import TrajectoryDataset
from trackzero.data.random_rollout import generate_random_rollout_data
from trackzero.eval.harness import EvalHarness
from trackzero.policy.mlp import MLPPolicy, load_checkpoint
from trackzero.policy.train import TrainingConfig, train


def main():
    parser = argparse.ArgumentParser(description="Train from random rollout data (Stage 1B)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--val-data", type=str, default="data/test.h5")
    parser.add_argument("--output-dir", type=str, default="outputs/stage1b")
    parser.add_argument("--n-trajectories", type=int, default=80000)
    parser.add_argument("--action-type", type=str, default="mixed",
                        choices=["uniform", "gaussian", "ou", "bangbang", "multisine", "mixed"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-hidden", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-trajectories", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Generate random rollout data
    print(f"Generating {args.n_trajectories} random rollout trajectories ({args.action_type})...")
    t0 = time.time()
    train_states, train_actions = generate_random_rollout_data(
        cfg, args.n_trajectories, action_type=args.action_type, seed=args.seed,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"  States: {train_states.shape}, Actions: {train_actions.shape}")

    # Load validation data (multisine test set — the evaluation target)
    print(f"Loading validation data: {args.val_data}")
    val_ds = TrajectoryDataset(args.val_data)
    val_states = val_ds.get_all_states()
    val_actions = val_ds.get_all_actions()
    print(f"  {len(val_ds)} trajectories")
    val_ds.close()

    # Train
    train_cfg = TrainingConfig(
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
        batch_size=args.batch_size,
        lr=1e-3,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    model, logs = train(
        train_states, train_actions,
        val_states, val_actions,
        cfg=train_cfg,
        tau_max=cfg.pendulum.tau_max,
        device=args.device,
    )

    # Save training log
    output_dir = Path(args.output_dir)
    log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump([{"epoch": l.epoch, "train_loss": l.train_loss,
                     "val_loss": l.val_loss, "elapsed_s": l.elapsed_s}
                    for l in logs], f, indent=2)

    # Save metadata
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "action_type": args.action_type,
            "n_trajectories": args.n_trajectories,
            "best_val_loss": min(l.val_loss for l in logs),
            "final_val_loss": logs[-1].val_loss,
        }, f, indent=2)

    # Eval
    if args.eval_trajectories > 0:
        print(f"\n=== Eval on {args.eval_trajectories} test trajectories ===")
        best_model = load_checkpoint(output_dir / "best_model.pt")
        policy = MLPPolicy(best_model, tau_max=cfg.pendulum.tau_max)

        val_ds2 = TrajectoryDataset(args.val_data)
        eval_states = val_ds2.get_all_states()
        eval_actions = val_ds2.get_all_actions()

        harness = EvalHarness(cfg)
        summary = harness.evaluate_policy(
            policy, eval_states, eval_actions,
            max_trajectories=args.eval_trajectories,
        )

        print(f"  Mean MSE_q:     {summary.mean_mse_q:.6e}")
        print(f"  Mean MSE_v:     {summary.mean_mse_v:.6e}")
        print(f"  Mean MSE_total: {summary.mean_mse_total:.6e}")
        print(f"  Max MSE_total:  {summary.max_mse_total:.6e}")
        print(f"  Mean max_err_q: {summary.mean_max_error_q:.6e}")

        summary.to_json(output_dir / "eval_results.json")
        val_ds2.close()


if __name__ == "__main__":
    main()
