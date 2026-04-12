#!/usr/bin/env python3
"""Stage 1C adversarial hard-example mining experiment.

Pipeline:
1. Generate seed random rollout data
2. Load a pre-trained tracker (Stage 1B or random_matched)
3. Generate large candidate pool
4. Score candidates by tracker's closed-loop error
5. Select hardest trajectories
6. Merge seed + adversarial, train final model
7. Evaluate on ID + report metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trackzero.config import load_config
from trackzero.data.adversarial import (
    score_trajectories_by_tracker_error,
    select_hardest_trajectories,
)
from trackzero.data.dataset import TrajectoryDataset
from trackzero.data.random_rollout import generate_random_rollout_data
from trackzero.eval.harness import EvalHarness
from trackzero.policy.mlp import MLPPolicy, load_checkpoint
from trackzero.policy.train import TrainingConfig, train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adversarial hard-example mining")
    p.add_argument("--config", type=str, default="configs/medium.yaml")
    p.add_argument("--val-data", type=str, default="data/medium/test.h5")
    p.add_argument("--base-model", type=str, required=True,
                   help="Pre-trained tracker to find hard examples against")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--seed-trajectories", type=int, default=2000)
    p.add_argument("--candidate-trajectories", type=int, default=12000)
    p.add_argument("--select-trajectories", type=int, default=8000)
    p.add_argument("--action-type", type=str, default="mixed")
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--n-hidden", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=65536)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--eval-trajectories", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load validation data
    print("Loading validation data...")
    val_ds = TrajectoryDataset(args.val_data)
    val_states = val_ds.get_all_states()
    val_actions = val_ds.get_all_actions()
    val_ds.close()

    # Load pre-trained tracker for scoring
    print(f"Loading base tracker from {args.base_model}")
    base_model = load_checkpoint(args.base_model, device=args.device)
    base_policy = MLPPolicy(
        base_model, tau_max=cfg.pendulum.tau_max, device=args.device
    )

    # Generate seed data
    print(f"Generating {args.seed_trajectories} seed trajectories...")
    seed_s, seed_a = generate_random_rollout_data(
        cfg, args.seed_trajectories,
        action_type=args.action_type, seed=args.seed,
        use_gpu=True, gpu_device=args.device,
    )

    # Generate candidate pool
    print(f"Generating {args.candidate_trajectories} candidate trajectories...")
    cand_s, cand_a = generate_random_rollout_data(
        cfg, args.candidate_trajectories,
        action_type=args.action_type, seed=args.seed + 5000,
        use_gpu=True, gpu_device=args.device,
    )

    # Score candidates by tracker error
    print("Scoring candidates by tracker error (closed-loop rollouts)...")
    scoring = score_trajectories_by_tracker_error(
        cfg, base_policy, cand_s, cand_a
    )
    print(f"  Mean error: {scoring['mean_score']:.6f}")
    print(f"  90th pct:   {scoring['pct90_score']:.6f}")
    print(f"  Max error:  {scoring['max_score']:.6f}")

    # Select hardest
    sel_s, sel_a, sel_idx = select_hardest_trajectories(
        cand_s, cand_a, scoring["trajectory_scores"],
        args.select_trajectories,
    )
    sel_scores = scoring["trajectory_scores"][sel_idx]
    print(f"Selected {len(sel_idx)} hardest trajectories:")
    print(f"  Mean selected error: {sel_scores.mean():.6f}")
    print(f"  Min selected error:  {sel_scores.min():.6f}")

    # Merge seed + adversarial
    train_s = np.concatenate([seed_s, sel_s], axis=0)
    train_a = np.concatenate([seed_a, sel_a], axis=0)
    print(f"Training on {len(train_s)} trajectories "
          f"({args.seed_trajectories} seed + {len(sel_idx)} adversarial)")

    # Train final model
    tcfg = TrainingConfig(
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=str(out),
    )
    train(train_s, train_a, val_states, val_actions,
          cfg=tcfg, tau_max=cfg.pendulum.tau_max, device=args.device)

    # Evaluate
    print("Evaluating final model...")
    final_model = load_checkpoint(out / "best_model.pt", device=args.device)
    final_policy = MLPPolicy(
        final_model, tau_max=cfg.pendulum.tau_max, device=args.device
    )
    harness = EvalHarness(cfg)
    summary = harness.evaluate_policy(
        final_policy, val_states, val_actions,
        max_trajectories=args.eval_trajectories,
    )
    summary.to_json(out / "eval_results.json")

    # Save metadata
    metadata = {
        "method": "adversarial_hard_example_mining",
        "base_model": args.base_model,
        "seed_trajectories": args.seed_trajectories,
        "candidate_trajectories": args.candidate_trajectories,
        "select_trajectories": args.select_trajectories,
        "action_type": args.action_type,
        "scoring_stats": {
            "mean_candidate_error": scoring["mean_score"],
            "median_candidate_error": scoring["median_score"],
            "pct90_candidate_error": scoring["pct90_score"],
            "max_candidate_error": scoring["max_score"],
            "mean_selected_error": float(sel_scores.mean()),
            "min_selected_error": float(sel_scores.min()),
        },
        "final_eval": {
            "mean_mse_total": summary.mean_mse_total,
            "mean_mse_q": summary.mean_mse_q,
            "mean_mse_v": summary.mean_mse_v,
            "max_mse_total": summary.max_mse_total,
        },
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Results in {out}")
    print(f"  ID mean_mse_total: {summary.mean_mse_total:.6e}")
    print(f"  ID max_mse_total:  {summary.max_mse_total:.6e}")


if __name__ == "__main__":
    main()
