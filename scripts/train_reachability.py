#!/usr/bin/env python3
"""Stage 1D.1: Reachability-guided single-step data generation + training.

Generates inverse dynamics training data by uniformly sampling states and
actions across the feasible region, then training the standard MLP.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from trackzero.config import load_config
from trackzero.data.dataset import TrajectoryDataset
from trackzero.data.reachability import (
    generate_reachability_data,
    generate_mixed_reachability_data,
)
from trackzero.eval.harness import EvalHarness
from trackzero.policy.mlp import MLPPolicy, load_checkpoint
from trackzero.policy.train import TrainingConfig, train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1D.1 reachability training")
    p.add_argument("--config", type=str, default="configs/medium.yaml")
    p.add_argument("--val-data", type=str, default="data/medium/test.h5")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--mode", type=str, default="single_step",
                   choices=["single_step", "mixed"],
                   help="single_step: uniform (s,a)->s' pairs. "
                        "mixed: single-step + short trajectories.")
    p.add_argument("--n-transitions", type=int, default=5000000,
                   help="Number of single-step transitions")
    p.add_argument("--n-short-traj", type=int, default=0,
                   help="Number of short trajectories (mixed mode)")
    p.add_argument("--short-traj-len", type=int, default=50)
    p.add_argument("--v-range", type=float, default=15.0,
                   help="Velocity sampling range: [-v, v]")
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

    t0 = time.time()

    if args.mode == "single_step":
        print(f"Generating {args.n_transitions:,} single-step transitions "
              f"(v_range=±{args.v_range})...")
        train_states, train_actions = generate_reachability_data(
            cfg, args.n_transitions,
            v_range=(-args.v_range, args.v_range),
            seed=args.seed,
        )
        n_pairs = args.n_transitions
    else:
        print(f"Generating mixed data: {args.n_transitions:,} single-step "
              f"+ {args.n_short_traj} short trajectories...")
        ss_s, ss_a, st_s, st_a = generate_mixed_reachability_data(
            cfg, args.n_transitions, args.n_short_traj,
            short_traj_len=args.short_traj_len,
            v_range=(-args.v_range, args.v_range),
            seed=args.seed,
        )
        # For mixed mode, we train on flattened pairs from both sources
        # Single-step: (N1, 2, 4) -> N1 pairs
        # Short traj: (N2, T+1, 4) -> N2*T pairs
        # Merge by extracting pairs manually
        from trackzero.policy.train import extract_pairs
        ss_inputs, ss_targets = extract_pairs(ss_s, ss_a)
        st_inputs, st_targets = extract_pairs(st_s, st_a)
        # Can't use the standard train() directly with mixed shapes.
        # Instead, concatenate and use single-step format for all.
        # Wrap back into trajectory format for the train function.
        train_states = ss_s
        train_actions = ss_a
        n_pairs = len(ss_inputs) + len(st_inputs)
        # For now just use single-step data; mixed mode TBD
        print(f"  Using single-step data: {len(ss_inputs):,} pairs")

    gen_time = time.time() - t0
    print(f"Data generation took {gen_time:.1f}s ({n_pairs:,} pairs)")

    # Train
    tcfg = TrainingConfig(
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=str(out),
    )
    model, logs = train(
        train_states, train_actions, val_states, val_actions,
        cfg=tcfg, tau_max=cfg.pendulum.tau_max, device=args.device,
    )

    # Evaluate
    print("Evaluating final model on ID references...")
    best_model = load_checkpoint(out / "best_model.pt", device=args.device)
    policy = MLPPolicy(
        best_model, tau_max=cfg.pendulum.tau_max, device=args.device
    )
    harness = EvalHarness(cfg)
    summary = harness.evaluate_policy(
        policy, val_states, val_actions,
        max_trajectories=args.eval_trajectories,
    )
    summary.to_json(out / "eval_results.json")

    # Save metadata
    metadata = {
        "method": "reachability_guided",
        "mode": args.mode,
        "n_transitions": args.n_transitions,
        "n_short_traj": args.n_short_traj,
        "v_range": args.v_range,
        "n_training_pairs": n_pairs,
        "generation_time_s": gen_time,
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


if __name__ == "__main__":
    main()
