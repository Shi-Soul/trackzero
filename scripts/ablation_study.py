#!/usr/bin/env python3
"""Stage 1E: Ablation study for inverse dynamics learning.

Tests:
1. Model capacity: 128x2, 256x3, 512x4, 1024x5
2. Data size: 10k, 20k, 40k, 80k, 160k random rollout trajectories
3. Action distribution: each type alone vs mixed
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from trackzero.config import load_config
from trackzero.data.dataset import TrajectoryDataset
from trackzero.data.ood_references import generate_ood_reference_data
from trackzero.data.random_rollout import generate_random_rollout_data
from trackzero.eval.harness import EvalHarness
from trackzero.policy.mlp import MLPPolicy, load_checkpoint
from trackzero.policy.train import TrainingConfig, train


def quick_eval(model, tau_max, ref_states, ref_actions, harness, n_traj=100, label="", use_gpu=False, gpu_device="cuda:0", gpu_sim=None):
    """Run quick evaluation and return summary dict."""
    policy = MLPPolicy(model, tau_max=tau_max)
    summary = harness.evaluate_policy(
        policy, ref_states, ref_actions, max_trajectories=n_traj,
        use_gpu=use_gpu, gpu_device=gpu_device, gpu_sim=gpu_sim,
    )
    return {
        "label": label,
        "mean_mse_q": summary.mean_mse_q,
        "mean_mse_v": summary.mean_mse_v,
        "mean_mse_total": summary.mean_mse_total,
        "max_mse_total": summary.max_mse_total,
        "mean_max_error_q": summary.mean_max_error_q,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 1E ablation study")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--val-data", type=str, default="data/test.h5")
    parser.add_argument("--output-dir", type=str, default="outputs/ablation")
    parser.add_argument("--ablation", type=str, required=True,
                        choices=["capacity", "datasize", "action_type"],
                        help="Which ablation to run")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--n-train", type=int, default=None,
                        help="Override number of training trajectories (default varies by ablation)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-trajectories", type=int, default=100)
    parser.add_argument("--gpu-sim", action="store_true",
                        help="Use GPU-parallel simulation for data gen and eval")
    parser.add_argument("--gpu-device", type=str, default="cuda:0",
                        help="CUDA device for GPU simulation (e.g. cuda:0, cuda:1)")
    args = parser.parse_args()

    # Default training device to GPU sim device when using GPU sim
    if args.device is None and args.gpu_sim:
        args.device = args.gpu_device

    cfg = load_config(args.config)
    harness = EvalHarness(cfg)
    output_dir = Path(args.output_dir) / args.ablation
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load multisine validation data (in-distribution)
    print("Loading multisine validation data...")
    val_ds = TrajectoryDataset(args.val_data)
    val_states = val_ds.get_all_states()
    val_actions = val_ds.get_all_actions()
    val_ds.close()

    # Generate OOD data for evaluation
    print("Generating OOD evaluation data (200 mixed_ood)...")
    ood_states, ood_actions = generate_ood_reference_data(
        cfg, 200, action_type="mixed_ood", seed=99,
        use_gpu=args.gpu_sim, gpu_device=args.gpu_device,
    )

    # Create a shared GPU simulator for eval (avoids re-init overhead per call)
    eval_gpu_sim = None
    if args.gpu_sim:
        from trackzero.sim.gpu_simulator import GPUSimulator
        eval_gpu_sim = GPUSimulator(cfg, n_worlds=max(args.eval_trajectories, 200),
                                     device=args.gpu_device)

    all_results = []

    if args.ablation == "capacity":
        n_cap = args.n_train or 80000
        print(f"Generating {n_cap} mixed random rollout data...")
        t0 = time.time()
        train_s, train_a = generate_random_rollout_data(cfg, n_cap, "mixed", seed=0, use_gpu=args.gpu_sim, gpu_device=args.gpu_device)
        print(f"  Done in {time.time()-t0:.1f}s")

        configs = [
            (128, 2, "128x2"),
            (256, 3, "256x3"),
            (512, 4, "512x4"),
            (1024, 5, "1024x5"),
        ]

        for hdim, nhid, name in configs:
            print(f"\n{'='*60}")
            print(f"Training {name} ({hdim}x{nhid})")
            print(f"{'='*60}")

            train_cfg = TrainingConfig(
                hidden_dim=hdim, n_hidden=nhid,
                batch_size=args.batch_size, lr=1e-3,
                epochs=args.epochs, seed=args.seed,
                output_dir=str(output_dir / name),
            )

            model, logs = train(
                train_s, train_a, val_states, val_actions,
                cfg=train_cfg, tau_max=cfg.pendulum.tau_max,
                device=args.device,
            )

            n_params = sum(p.numel() for p in model.parameters())
            best_val = min(l.val_loss for l in logs)

            # Eval on ID
            best_model = load_checkpoint(Path(train_cfg.output_dir) / "best_model.pt")
            id_result = quick_eval(
                best_model, cfg.pendulum.tau_max,
                val_states, val_actions, harness,
                n_traj=args.eval_trajectories, label=f"{name}_id",
                use_gpu=args.gpu_sim, gpu_device=args.gpu_device,
                gpu_sim=eval_gpu_sim,
            )

            # Eval on OOD
            ood_result = quick_eval(
                best_model, cfg.pendulum.tau_max,
                ood_states, ood_actions, harness,
                n_traj=args.eval_trajectories, label=f"{name}_ood",
                use_gpu=args.gpu_sim, gpu_device=args.gpu_device,
                gpu_sim=eval_gpu_sim,
            )

            result = {
                "name": name,
                "hidden_dim": hdim,
                "n_hidden": nhid,
                "n_params": n_params,
                "best_val_loss": best_val,
                "id_eval": id_result,
                "ood_eval": ood_result,
            }
            all_results.append(result)

            print(f"  {name}: {n_params:,} params, val_loss={best_val:.6f}")
            print(f"    ID MSE_total:  {id_result['mean_mse_total']:.6e}")
            print(f"    OOD MSE_total: {ood_result['mean_mse_total']:.6e}")

    elif args.ablation == "datasize":
        # Fix architecture: 512x4
        if args.n_train:
            max_size = args.n_train
            sizes = [max_size // 8, max_size // 4, max_size // 2, max_size]
            sizes = sorted(set(s for s in sizes if s >= 100))
        else:
            sizes = [5000, 10000, 20000, 40000, 80000]

        for n_traj in sizes:
            print(f"\n{'='*60}")
            print(f"Training with {n_traj} trajectories")
            print(f"{'='*60}")

            t0 = time.time()
            train_s, train_a = generate_random_rollout_data(cfg, n_traj, "mixed", seed=0, use_gpu=args.gpu_sim, gpu_device=args.gpu_device)
            print(f"  Generated in {time.time()-t0:.1f}s")

            name = f"n{n_traj//1000}k"
            train_cfg = TrainingConfig(
                hidden_dim=512, n_hidden=4,
                batch_size=args.batch_size, lr=1e-3,
                epochs=args.epochs, seed=args.seed,
                output_dir=str(output_dir / name),
            )

            model, logs = train(
                train_s, train_a, val_states, val_actions,
                cfg=train_cfg, tau_max=cfg.pendulum.tau_max,
                device=args.device,
            )

            best_val = min(l.val_loss for l in logs)
            best_model = load_checkpoint(Path(train_cfg.output_dir) / "best_model.pt")

            id_result = quick_eval(
                best_model, cfg.pendulum.tau_max,
                val_states, val_actions, harness,
                n_traj=args.eval_trajectories, label=f"{name}_id",
                use_gpu=args.gpu_sim, gpu_device=args.gpu_device,
                gpu_sim=eval_gpu_sim,
            )
            ood_result = quick_eval(
                best_model, cfg.pendulum.tau_max,
                ood_states, ood_actions, harness,
                n_traj=args.eval_trajectories, label=f"{name}_ood",
                use_gpu=args.gpu_sim, gpu_device=args.gpu_device,
                gpu_sim=eval_gpu_sim,
            )

            result = {
                "name": name,
                "n_trajectories": n_traj,
                "best_val_loss": best_val,
                "id_eval": id_result,
                "ood_eval": ood_result,
            }
            all_results.append(result)

            print(f"  {name}: val_loss={best_val:.6f}")
            print(f"    ID MSE_total:  {id_result['mean_mse_total']:.6e}")
            print(f"    OOD MSE_total: {ood_result['mean_mse_total']:.6e}")

    elif args.ablation == "action_type":
        # Fix architecture and data size: 512x4
        n_act = args.n_train or 80000
        action_types = ["uniform", "gaussian", "ou", "bangbang", "multisine", "mixed"]

        for atype in action_types:
            print(f"\n{'='*60}")
            print(f"Training with {atype} actions ({n_act} trajectories)")
            print(f"{'='*60}")

            t0 = time.time()
            train_s, train_a = generate_random_rollout_data(cfg, n_act, atype, seed=0, use_gpu=args.gpu_sim, gpu_device=args.gpu_device)
            print(f"  Generated in {time.time()-t0:.1f}s")

            train_cfg = TrainingConfig(
                hidden_dim=512, n_hidden=4,
                batch_size=args.batch_size, lr=1e-3,
                epochs=args.epochs, seed=args.seed,
                output_dir=str(output_dir / atype),
            )

            model, logs = train(
                train_s, train_a, val_states, val_actions,
                cfg=train_cfg, tau_max=cfg.pendulum.tau_max,
                device=args.device,
            )

            best_val = min(l.val_loss for l in logs)
            best_model = load_checkpoint(Path(train_cfg.output_dir) / "best_model.pt")

            id_result = quick_eval(
                best_model, cfg.pendulum.tau_max,
                val_states, val_actions, harness,
                n_traj=args.eval_trajectories, label=f"{atype}_id",
                use_gpu=args.gpu_sim, gpu_device=args.gpu_device,
                gpu_sim=eval_gpu_sim,
            )
            ood_result = quick_eval(
                best_model, cfg.pendulum.tau_max,
                ood_states, ood_actions, harness,
                n_traj=args.eval_trajectories, label=f"{atype}_ood",
                use_gpu=args.gpu_sim, gpu_device=args.gpu_device,
                gpu_sim=eval_gpu_sim,
            )

            result = {
                "name": atype,
                "action_type": atype,
                "best_val_loss": best_val,
                "id_eval": id_result,
                "ood_eval": ood_result,
            }
            all_results.append(result)

            print(f"  {atype}: val_loss={best_val:.6f}")
            print(f"    ID MSE_total:  {id_result['mean_mse_total']:.6e}")
            print(f"    OOD MSE_total: {ood_result['mean_mse_total']:.6e}")

    # Save results
    results_path = output_dir / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"ABLATION SUMMARY: {args.ablation}")
    print(f"{'='*80}")
    print(f"{'Name':<20} {'Val Loss':>12} {'ID MSE_total':>15} {'OOD MSE_total':>15}")
    print("-" * 65)
    for r in all_results:
        print(f"{r['name']:<20} {r['best_val_loss']:>12.6f} "
              f"{r['id_eval']['mean_mse_total']:>15.6e} "
              f"{r['ood_eval']['mean_mse_total']:>15.6e}")

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
