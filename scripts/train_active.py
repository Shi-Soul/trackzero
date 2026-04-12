#!/usr/bin/env python3
"""Stage 1C: active data collection via ensemble disagreement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trackzero.config import load_config
from trackzero.data.active_collection import (
    score_trajectory_disagreement,
    select_top_trajectories,
    train_bootstrap_ensemble,
)
from trackzero.data.dataset import TrajectoryDataset
from trackzero.data.random_rollout import generate_random_rollout_data
from trackzero.eval.harness import EvalHarness
from trackzero.policy.mlp import MLPPolicy, load_checkpoint
from trackzero.policy.train import TrainingConfig, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 1C active-collection model")
    parser.add_argument("--config", type=str, default="configs/medium.yaml")
    parser.add_argument("--val-data", type=str, default="data/medium/test.h5")
    parser.add_argument("--output-dir", type=str, default="outputs/stage1c_active")
    parser.add_argument("--seed-trajectories", type=int, default=2000)
    parser.add_argument("--candidate-trajectories", type=int, default=12000)
    parser.add_argument("--select-trajectories", type=int, default=8000)
    parser.add_argument("--proposal-action-type", type=str, default="mixed",
                        choices=["uniform", "gaussian", "ou", "bangbang", "multisine", "mixed"])
    parser.add_argument("--proposal-action-params-json", type=str, default=None)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--ensemble-hidden-dim", type=int, default=256)
    parser.add_argument("--ensemble-n-hidden", type=int, default=3)
    parser.add_argument("--ensemble-epochs", type=int, default=20)
    parser.add_argument("--final-hidden-dim", type=int, default=512)
    parser.add_argument("--final-n-hidden", type=int, default=4)
    parser.add_argument("--final-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-trajectories", type=int, default=200)
    parser.add_argument("--score-batch-size", type=int, default=65536)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    action_params = json.loads(args.proposal_action_params_json) if args.proposal_action_params_json else None

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    print("Loading validation data...")
    val_ds = TrajectoryDataset(args.val_data)
    val_states = val_ds.get_all_states()
    val_actions = val_ds.get_all_actions()
    val_ds.close()

    print(f"Generating seed dataset: {args.seed_trajectories} trajectories")
    seed_states, seed_actions = generate_random_rollout_data(
        cfg,
        args.seed_trajectories,
        action_type=args.proposal_action_type,
        seed=args.seed,
        action_params=action_params,
    )

    print(f"Training ensemble of {args.ensemble_size} members")
    ensemble_cfg = TrainingConfig(
        hidden_dim=args.ensemble_hidden_dim,
        n_hidden=args.ensemble_n_hidden,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.ensemble_epochs,
        seed=args.seed,
        output_dir=str(output_dir / "ensemble"),
    )
    ensemble_meta = train_bootstrap_ensemble(
        seed_states,
        seed_actions,
        val_states,
        val_actions,
        ensemble_size=args.ensemble_size,
        cfg=ensemble_cfg,
        tau_max=cfg.pendulum.tau_max,
        device=device,
        output_dir=output_dir / "ensemble",
        seed=args.seed,
    )
    ensemble_models = [
        load_checkpoint(m["best_model_path"], device=device)
        for m in ensemble_meta
    ]

    print(f"Generating candidate pool: {args.candidate_trajectories} trajectories")
    candidate_states, candidate_actions = generate_random_rollout_data(
        cfg,
        args.candidate_trajectories,
        action_type=args.proposal_action_type,
        seed=args.seed + 1000,
        action_params=action_params,
    )

    print("Scoring candidate trajectories by ensemble disagreement...")
    disagreement = score_trajectory_disagreement(
        ensemble_models,
        candidate_states,
        device=device,
        batch_size=args.score_batch_size,
    )
    selected_states, selected_actions, selected_idx = select_top_trajectories(
        candidate_states,
        candidate_actions,
        disagreement["trajectory_scores"],
        n_select=args.select_trajectories,
    )

    train_states = np.concatenate([seed_states, selected_states], axis=0)
    train_actions = np.concatenate([seed_actions, selected_actions], axis=0)

    print(f"Final training dataset: {len(train_states)} trajectories")
    final_cfg = TrainingConfig(
        hidden_dim=args.final_hidden_dim,
        n_hidden=args.final_n_hidden,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.final_epochs,
        seed=args.seed,
        output_dir=str(output_dir),
    )
    train(
        train_states,
        train_actions,
        val_states,
        val_actions,
        cfg=final_cfg,
        tau_max=cfg.pendulum.tau_max,
        device=device,
    )

    print("Evaluating final model on held-out validation references...")
    best_model = load_checkpoint(output_dir / "best_model.pt", device=device)
    policy = MLPPolicy(best_model, tau_max=cfg.pendulum.tau_max, device=device)
    harness = EvalHarness(cfg)
    summary = harness.evaluate_policy(
        policy,
        val_states,
        val_actions,
        max_trajectories=args.eval_trajectories,
    )
    summary.to_json(output_dir / "eval_results.json")

    selected_scores = disagreement["trajectory_scores"][selected_idx]
    metadata = {
        "proposal_action_type": args.proposal_action_type,
        "proposal_action_params": action_params,
        "seed_trajectories": args.seed_trajectories,
        "candidate_trajectories": args.candidate_trajectories,
        "select_trajectories": args.select_trajectories,
        "ensemble_size": args.ensemble_size,
        "ensemble_hidden_dim": args.ensemble_hidden_dim,
        "ensemble_n_hidden": args.ensemble_n_hidden,
        "ensemble_epochs": args.ensemble_epochs,
        "final_hidden_dim": args.final_hidden_dim,
        "final_n_hidden": args.final_n_hidden,
        "final_epochs": args.final_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "device": device,
        "ensemble_members": ensemble_meta,
        "disagreement_stats": {
            "mean_score": float(np.mean(disagreement["trajectory_scores"])),
            "median_score": float(np.median(disagreement["trajectory_scores"])),
            "max_score": float(np.max(disagreement["trajectory_scores"])),
            "selected_mean_score": float(np.mean(selected_scores)),
            "selected_min_score": float(np.min(selected_scores)),
            "selected_max_score": float(np.max(selected_scores)),
            "mean_transition_std": disagreement["mean_transition_std"],
            "max_transition_std": disagreement["max_transition_std"],
        },
        "selected_indices_top20": selected_idx[:20].tolist(),
        "best_val_loss": min(m["best_val_loss"] for m in ensemble_meta),
        "final_eval": {
            "mean_mse_total": summary.mean_mse_total,
            "mean_mse_q": summary.mean_mse_q,
            "mean_mse_v": summary.mean_mse_v,
            "max_mse_total": summary.max_mse_total,
        },
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved Stage 1C outputs to {output_dir}")


if __name__ == "__main__":
    main()
