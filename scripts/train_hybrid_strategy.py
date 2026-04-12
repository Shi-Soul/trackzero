#!/usr/bin/env python3
"""Hybrid training strategies to combine maxent OOD robustness with random ID accuracy.

Key hypothesis: maxent degrades only 9.7× from ID→step (vs random 282×).
If we can match random's ID accuracy while preserving maxent's robustness,
we get the best of both worlds.

Strategies:
  concat    - Simply concatenate all maxent + random data (10M pairs)
  curriculum - Pretrain on maxent, finetune on random
  weighted  - Weighted mixture (emphasize random for ID, maxent for coverage)
  finetune  - Train on random, then finetune on combined data
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import h5py
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from trackzero.config import load_config
from trackzero.policy.mlp import InverseDynamicsMLP, MLPPolicy
from trackzero.data.ood_references import OOD_ACTION_GENERATORS, generate_ood_reference_data
from trackzero.data.dataset import TrajectoryDataset
from trackzero.eval.harness import EvalHarness


def load_random_data(path="data/medium/train.h5"):
    with h5py.File(path) as f:
        states = f["states"][:]  # (N, 501, 4)
        actions = f["actions"][:]  # (N, 500, 2)
    s_t = states[:, :-1].reshape(-1, 4)
    s_tp1 = states[:, 1:].reshape(-1, 4)
    a = actions.reshape(-1, 2)
    return s_t, s_tp1, a


def load_maxent_data(state_path="outputs/stage1c_maxent_rl/explorer_states.npy",
                     action_path="outputs/stage1c_maxent_rl/explorer_actions.npy"):
    states = np.load(state_path)  # (N, 501, 4)
    actions = np.load(action_path)  # (N, 500, 2)
    s_t = states[:, :-1].reshape(-1, 4)
    s_tp1 = states[:, 1:].reshape(-1, 4)
    a = actions.reshape(-1, 2)
    return s_t, s_tp1, a


def make_dataset(s_t, s_tp1, a, device, batch_size=65536):
    """Create batched tensors."""
    x = torch.tensor(np.concatenate([s_t, s_tp1], axis=1), dtype=torch.float32, device=device)
    y = torch.tensor(a, dtype=torch.float32, device=device)
    n = len(x)
    idx = torch.randperm(n, device=device)
    x, y = x[idx], y[idx]
    # Split 90/10
    split = int(0.9 * n)
    return (x[:split], y[:split]), (x[split:], y[split:])


def train_one_epoch(model, optimizer, train_x, train_y, batch_size=65536):
    model.train()
    n = len(train_x)
    total_loss = 0
    count = 0
    for i in range(0, n, batch_size):
        xb = train_x[i:i+batch_size]
        yb = train_y[i:i+batch_size]
        pred = model(xb)
        loss = nn.functional.mse_loss(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
        count += len(xb)
    return total_loss / count


@torch.no_grad()
def eval_model(model, val_x, val_y, batch_size=65536):
    model.eval()
    n = len(val_x)
    total_loss = 0
    count = 0
    for i in range(0, n, batch_size):
        xb = val_x[i:i+batch_size]
        yb = val_y[i:i+batch_size]
        pred = model(xb)
        loss = nn.functional.mse_loss(pred, yb)
        total_loss += loss.item() * len(xb)
        count += len(xb)
    return total_loss / count


def eval_ood(model, cfg, device):
    """Run OOD benchmark on 6 settings."""
    harness = EvalHarness(cfg)
    tau_max = cfg.pendulum.tau_max
    policy = MLPPolicy(model, tau_max=tau_max)

    # ID evaluation
    val_ds = TrajectoryDataset("data/medium/test.h5")
    val_s, val_a = val_ds.get_all_states(), val_ds.get_all_actions()
    val_ds.close()

    results = {}
    print("  ID evaluation...")
    id_summary = harness.evaluate_policy(policy, val_s, val_a, max_trajectories=200)
    results["id_multisine"] = float(id_summary.mean_mse_total)
    print(f"    id_multisine: {results['id_multisine']:.4e}")

    # OOD evaluation
    for ood_type in OOD_ACTION_GENERATORS:
        s, a = generate_ood_reference_data(cfg, 200, action_type=ood_type)
        summary = harness.evaluate_policy(policy, s, a, max_trajectories=200)
        mse = float(summary.mean_mse_total)
        results[f"ood_{ood_type}"] = mse
        print(f"    ood_{ood_type}: {mse:.4e}")

    return results


def run_strategy(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Strategy: {args.strategy}")
    print(f"Architecture: {args.hidden_dim}x{args.n_hidden}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    rand_st, rand_stp1, rand_a = load_random_data()
    maxent_st, maxent_stp1, maxent_a = load_maxent_data()
    print(f"  Random: {len(rand_st)} pairs, Maxent: {len(maxent_st)} pairs")

    cfg = load_config()

    if args.strategy == "concat":
        # Simply concatenate all data
        st = np.concatenate([rand_st, maxent_st])
        stp1 = np.concatenate([rand_stp1, maxent_stp1])
        a = np.concatenate([rand_a, maxent_a])
        print(f"  Combined: {len(st)} pairs")
        train_data, val_data = make_dataset(st, stp1, a, device)
        model = train_phase(args, device, train_data, val_data, args.epochs, "concat")

    elif args.strategy == "curriculum":
        # Phase 1: Train on maxent data
        print(f"\n=== Phase 1: Maxent pretraining ({args.pretrain_epochs} epochs) ===")
        train_m, val_m = make_dataset(maxent_st, maxent_stp1, maxent_a, device)
        model = train_phase(args, device, train_m, val_m, args.pretrain_epochs, "phase1_maxent")

        # Phase 2: Finetune on random data
        print(f"\n=== Phase 2: Random finetuning ({args.finetune_epochs} epochs) ===")
        train_r, val_r = make_dataset(rand_st, rand_stp1, rand_a, device)
        model = train_phase(args, device, train_r, val_r, args.finetune_epochs,
                           "phase2_random", model=model, lr=args.finetune_lr)

    elif args.strategy == "reverse_curriculum":
        # Phase 1: Train on random data (get good ID accuracy)
        print(f"\n=== Phase 1: Random pretraining ({args.pretrain_epochs} epochs) ===")
        train_r, val_r = make_dataset(rand_st, rand_stp1, rand_a, device)
        model = train_phase(args, device, train_r, val_r, args.pretrain_epochs, "phase1_random")

        # Phase 2: Finetune on combined data (preserve ID, add OOD)
        st = np.concatenate([rand_st, maxent_st])
        stp1 = np.concatenate([rand_stp1, maxent_stp1])
        a = np.concatenate([rand_a, maxent_a])
        print(f"\n=== Phase 2: Combined finetuning ({args.finetune_epochs} epochs) ===")
        train_c, val_c = make_dataset(st, stp1, a, device)
        model = train_phase(args, device, train_c, val_c, args.finetune_epochs,
                           "phase2_combined", model=model, lr=args.finetune_lr)

    elif args.strategy == "weighted":
        # Weighted mixture: oversample random data
        weight = args.random_weight
        # Subsample maxent or oversample random
        n_random = len(rand_st)
        n_maxent_use = int(n_random / weight)
        idx = np.random.default_rng(42).choice(len(maxent_st), n_maxent_use, replace=False)
        st = np.concatenate([rand_st, maxent_st[idx]])
        stp1 = np.concatenate([rand_stp1, maxent_stp1[idx]])
        a = np.concatenate([rand_a, maxent_a[idx]])
        print(f"  Weighted mix: {n_random} random + {n_maxent_use} maxent (weight={weight})")
        train_data, val_data = make_dataset(st, stp1, a, device)
        model = train_phase(args, device, train_data, val_data, args.epochs, "weighted")

    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    # OOD evaluation
    print("\n=== OOD Evaluation ===")
    ood_results = eval_ood(model, cfg, device)

    # Save results
    results = {
        "strategy": args.strategy,
        "architecture": f"{args.hidden_dim}x{args.n_hidden}",
        "ood_results": ood_results,
        "args": vars(args),
    }
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def train_phase(args, device, train_data, val_data, epochs, phase_name, model=None, lr=None):
    train_x, train_y = train_data
    val_x, val_y = val_data
    print(f"  Training: {len(train_x)} pairs, Validation: {len(val_x)} pairs")

    if model is None:
        model = InverseDynamicsMLP(state_dim=4, action_dim=2,
                                   hidden_dim=args.hidden_dim,
                                   n_hidden=args.n_hidden).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {params:,} params")

    if lr is None:
        lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, optimizer, train_x, train_y)
        val_loss = eval_model(model, val_x, val_y)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - t0
        if epoch <= 3 or epoch % 10 == 0 or epoch == epochs:
            print(f"  [{phase_name}] Epoch {epoch:>3}/{epochs}"
                  f"  train={train_loss:.6f}  val={val_loss:.6f}"
                  f"  best={best_val:.6f}  time={elapsed:.0f}s")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save checkpoint
    ckpt_path = os.path.join(args.output_dir, f"{phase_name}_best.pt")
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    print(f"  Saved: {ckpt_path} (best_val={best_val:.6f})")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid training strategies")
    parser.add_argument("--strategy", required=True,
                        choices=["concat", "curriculum", "reverse_curriculum", "weighted"])
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-hidden", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=100, help="Total epochs (for concat/weighted)")
    parser.add_argument("--pretrain-epochs", type=int, default=50, help="Phase 1 epochs (curriculum)")
    parser.add_argument("--finetune-epochs", type=int, default=50, help="Phase 2 epochs (curriculum)")
    parser.add_argument("--finetune-lr", type=float, default=1e-4, help="Phase 2 LR (curriculum)")
    parser.add_argument("--random-weight", type=float, default=4.0,
                        help="Ratio of random:maxent in weighted strategy")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_strategy(args)
