#!/usr/bin/env python3
"""Stage 1D: hindsight relabeling from achieved closed-loop trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trackzero.config import load_config
from trackzero.data.dataset import TrajectoryDataset
from trackzero.data.hindsight import (
    evaluate_teacher_on_requested_refs,
    generate_reference_batch,
    rollout_policy_as_hindsight_data,
)
from trackzero.data.random_rollout import generate_random_rollout_data
from trackzero.eval.harness import EvalHarness
from trackzero.policy.mlp import MLPPolicy, load_checkpoint
from trackzero.policy.train import TrainingConfig, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 1D hindsight-relabeling model")
    parser.add_argument("--config", type=str, default="configs/medium.yaml")
    parser.add_argument("--val-data", type=str, default="data/medium/test.h5")
    parser.add_argument("--base-model", type=str, default="outputs/stage1b_scaled/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/stage1d_hindsight")
    parser.add_argument("--seed-trajectories", type=int, default=10000)
    parser.add_argument("--hindsight-trajectories", type=int, default=10000)
    parser.add_argument("--reference-source", type=str, default="mixed_ood")
    parser.add_argument("--seed-action-type", type=str, default="mixed",
                        choices=["uniform", "gaussian", "ou", "bangbang", "multisine", "mixed"])
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-hidden", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval-trajectories", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading validation data...")
    val_ds = TrajectoryDataset(args.val_data)
    val_states = val_ds.get_all_states()
    val_actions = val_ds.get_all_actions()
    val_ds.close()

    print(f"Generating seed random-rollout data: {args.seed_trajectories} trajectories")
    seed_states, seed_actions = generate_random_rollout_data(
        cfg,
        args.seed_trajectories,
        action_type=args.seed_action_type,
        seed=args.seed,
        use_gpu=True,
        gpu_device=args.device,
    )

    print(f"Loading teacher policy from {args.base_model}")
    teacher_model = load_checkpoint(args.base_model, device=args.device)
    teacher_policy = MLPPolicy(teacher_model, tau_max=cfg.pendulum.tau_max, device=args.device)

    print(f"Generating arbitrary reference batch: {args.hindsight_trajectories} trajectories from {args.reference_source}")
    ref_states, ref_actions = generate_reference_batch(
        cfg,
        args.hindsight_trajectories,
        source=args.reference_source,
        seed=args.seed + 1000,
        gpu_device=args.device,
    )

    print("Evaluating teacher on requested references...")
    teacher_summary = evaluate_teacher_on_requested_refs(
        cfg,
        teacher_policy,
        ref_states,
        ref_actions,
        gpu_device=args.device,
    )
    teacher_summary.to_json(output_dir / "teacher_on_requested_refs.json")

    print("Rolling out teacher and relabeling achieved trajectories...")
    achieved_states, achieved_actions = rollout_policy_as_hindsight_data(
        cfg,
        teacher_policy,
        ref_states,
        ref_actions,
        gpu_device=args.device,
    )

    train_states = np.concatenate([seed_states, achieved_states], axis=0)
    train_actions = np.concatenate([seed_actions, achieved_actions], axis=0)
    print(f"Final hindsight training set: {len(train_states)} trajectories")

    train_cfg = TrainingConfig(
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=str(output_dir),
    )
    train(
        train_states,
        train_actions,
        val_states,
        val_actions,
        cfg=train_cfg,
        tau_max=cfg.pendulum.tau_max,
        device=args.device,
    )

    print("Evaluating final hindsight model on held-out validation references...")
    best_model = load_checkpoint(output_dir / "best_model.pt", device=args.device)
    policy = MLPPolicy(best_model, tau_max=cfg.pendulum.tau_max, device=args.device)
    harness = EvalHarness(cfg)
    summary = harness.evaluate_policy(
        policy,
        val_states,
        val_actions,
        max_trajectories=args.eval_trajectories,
    )
    summary.to_json(output_dir / "eval_results.json")

    achieved_span = {
        "q_min": achieved_states[:, :, :2].min(axis=(0, 1)).tolist(),
        "q_max": achieved_states[:, :, :2].max(axis=(0, 1)).tolist(),
        "v_min": achieved_states[:, :, 2:].min(axis=(0, 1)).tolist(),
        "v_max": achieved_states[:, :, 2:].max(axis=(0, 1)).tolist(),
    }

    metadata = {
        "base_model": args.base_model,
        "reference_source": args.reference_source,
        "seed_trajectories": args.seed_trajectories,
        "hindsight_trajectories": args.hindsight_trajectories,
        "seed_action_type": args.seed_action_type,
        "hidden_dim": args.hidden_dim,
        "n_hidden": args.n_hidden,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "device": args.device,
        "teacher_requested_ref_eval": {
            "mean_mse_total": teacher_summary.mean_mse_total,
            "mean_mse_q": teacher_summary.mean_mse_q,
            "mean_mse_v": teacher_summary.mean_mse_v,
            "max_mse_total": teacher_summary.max_mse_total,
            "mean_max_error_q": teacher_summary.mean_max_error_q,
        },
        "achieved_state_span": achieved_span,
        "final_eval": {
            "mean_mse_total": summary.mean_mse_total,
            "mean_mse_q": summary.mean_mse_q,
            "mean_mse_v": summary.mean_mse_v,
            "max_mse_total": summary.max_mse_total,
            "mean_max_error_q": summary.mean_max_error_q,
        },
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved Stage 1D outputs to {output_dir}")


if __name__ == "__main__":
    main()
