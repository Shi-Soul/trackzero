#!/usr/bin/env python3
"""Train supervised inverse dynamics baseline (Stage 1A)."""

import argparse
import json
import time
from pathlib import Path

from trackzero.config import load_config
from trackzero.data.dataset import TrajectoryDataset
from trackzero.eval.harness import EvalHarness
from trackzero.policy.mlp import MLPPolicy, load_checkpoint
from trackzero.policy.train import TrainingConfig, train


def main():
    parser = argparse.ArgumentParser(description="Train supervised inverse dynamics MLP")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--train-data", type=str, default="data/train.h5")
    parser.add_argument("--val-data", type=str, default="data/test.h5")
    parser.add_argument("--output-dir", type=str, default="outputs/stage1a")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-hidden", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-trajectories", type=int, default=100,
                        help="Number of trajectories for quick eval after training")
    args = parser.parse_args()

    cfg = load_config(args.config)

    train_cfg = TrainingConfig(
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # Load data
    print(f"Loading training data: {args.train_data}")
    train_ds = TrajectoryDataset(args.train_data)
    train_states = train_ds.get_all_states()
    train_actions = train_ds.get_all_actions()
    print(f"  {len(train_ds)} trajectories, {train_ds.n_steps} steps each")
    train_ds.close()

    print(f"Loading validation data: {args.val_data}")
    val_ds = TrajectoryDataset(args.val_data)
    val_states = val_ds.get_all_states()
    val_actions = val_ds.get_all_actions()
    print(f"  {len(val_ds)} trajectories, {val_ds.n_steps} steps each")
    val_ds.close()

    # Train
    model, logs = train(
        train_states, train_actions,
        val_states, val_actions,
        cfg=train_cfg,
        tau_max=cfg.pendulum.tau_max,
        device=args.device,
    )

    # Save training log
    log_path = Path(args.output_dir) / "training_log.json"
    with open(log_path, "w") as f:
        json.dump([{"epoch": l.epoch, "train_loss": l.train_loss,
                     "val_loss": l.val_loss, "elapsed_s": l.elapsed_s}
                    for l in logs], f, indent=2)
    print(f"Training log saved to {log_path}")

    # Quick eval with trained model
    if args.eval_trajectories > 0:
        print(f"\n=== Quick eval on {args.eval_trajectories} test trajectories ===")
        best_model = load_checkpoint(Path(args.output_dir) / "best_model.pt")
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

        summary.to_json(Path(args.output_dir) / "eval_results.json")
        val_ds2.close()


if __name__ == "__main__":
    main()
