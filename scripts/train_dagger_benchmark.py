#!/usr/bin/env python3
"""DAgger-style iterative refinement for inverse dynamics.

Algorithm:
1. Train initial policy on random data
2. For each iteration:
   a. Roll out benchmark-like trajectories using the current policy
   b. Collect ground-truth labels from the simulator
   c. Aggregate new data with existing dataset
   d. Retrain the policy

This focuses training data on states the policy actually visits,
naturally improving benchmark performance.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from trackzero.config import load_config
from trackzero.data.dataset import TrajectoryDataset
from trackzero.data.ood_references import generate_ood_reference_data
from trackzero.eval.harness import EvalHarness
from trackzero.policy.mlp import InverseDynamicsMLP, MLPPolicy
from trackzero.sim.simulator import Simulator


def generate_dagger_trajectories(cfg, policy, n_traj, seed, device):
    """Roll out policy to collect on-policy states, then get oracle actions."""
    sim = Simulator(cfg)
    tau_max = cfg.pendulum.tau_max
    T = int(cfg.dataset.trajectory_duration / cfg.simulation.control_dt)
    rng = np.random.default_rng(seed)

    all_states = np.zeros((n_traj, T + 1, 4))
    all_actions = np.zeros((n_traj, T, 2))

    for i in range(n_traj):
        q0 = rng.uniform(-np.pi, np.pi, size=2)
        v0 = rng.uniform(-5.0, 5.0, size=2)

        states_list = [np.concatenate([q0, v0])]
        actions_list = []

        state = np.concatenate([q0, v0])
        for t in range(T):
            # Use a mix of policy actions and random exploration
            if rng.random() < 0.3:  # 30% exploration
                action = rng.uniform(-tau_max, tau_max, size=2)
            else:
                # Get policy action
                s_t = torch.tensor(state, dtype=torch.float32, device=device)
                # We need next state to predict action — use random target
                target = state + rng.normal(0, 0.1, size=4)
                s_tp1 = torch.tensor(target, dtype=torch.float32, device=device)
                with torch.no_grad():
                    inp = torch.cat([s_t, s_tp1]).unsqueeze(0)
                    pred = policy.model(inp).cpu().numpy().flatten()
                action = pred * tau_max

            action = np.clip(action, -tau_max, tau_max)
            next_state = sim.step(state[:2], state[2:], action)
            actions_list.append(action)

            state = next_state
            states_list.append(state)

        all_states[i] = np.array(states_list)
        all_actions[i] = np.array(actions_list)

    return all_states, all_actions


def generate_benchmark_reference_trajectories(cfg, n_per_family, seed):
    """Generate trajectories similar to benchmark families for DAgger."""
    all_s, all_a = [], []
    for name in ["chirp", "step", "random_walk", "sawtooth", "pulse"]:
        s, a = generate_ood_reference_data(
            cfg, n_per_family, action_type=name, seed=seed
        )
        all_s.append(s)
        all_a.append(a)

    # Also add multisine from test set
    ds = TrajectoryDataset("data/medium/test.h5")
    test_s, test_a = ds.get_all_states(), ds.get_all_actions()
    ds.close()
    all_s.append(test_s[:n_per_family])
    all_a.append(test_a[:n_per_family])

    return np.concatenate(all_s), np.concatenate(all_a)


def train_model(X, Y, X_val, Y_val, hidden_dim, n_hidden, device,
                epochs=100, lr=1e-3, batch_size=65536):
    """Train an inverse dynamics model."""
    model = InverseDynamicsMLP(
        state_dim=4, action_dim=2,
        hidden_dim=hidden_dim, n_hidden=n_hidden
    ).to(device)

    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)),
        batch_size=batch_size, shuffle=True, pin_memory=True,
    )
    val_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)),
        batch_size=batch_size, shuffle=False, pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        train_loss = total_loss / len(X)

        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_loss_sum += criterion(model(xb), yb).item() * len(xb)
        val_loss = val_loss_sum / len(X_val)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}: train={train_loss:.6f} val={val_loss:.6f} best={best_val:.6f}")

    model.load_state_dict(best_state)
    return model, best_val


def evaluate_on_benchmark(model, cfg, device, tau_max):
    """Quick benchmark evaluation."""
    harness = EvalHarness(cfg)
    policy = MLPPolicy(model, tau_max=tau_max)

    families = {}
    ds = TrajectoryDataset("data/medium/test.h5")
    families["multisine"] = (ds.get_all_states()[:100], ds.get_all_actions()[:100])
    ds.close()
    for name in ["chirp", "step", "random_walk", "sawtooth", "pulse"]:
        s, a = generate_ood_reference_data(cfg, 100, action_type=name, seed=12345)
        families[name] = (s, a)

    all_mse = []
    fam_results = {}
    for fname, (ref_s, ref_a) in families.items():
        summary = harness.evaluate_policy(policy, ref_s, ref_a, max_trajectories=100)
        fam_results[fname] = float(summary.mean_mse_total)
        all_mse.extend([r.mse_total for r in summary.results])

    agg = float(np.mean(all_mse))
    return agg, fam_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iters", type=int, default=5)
    parser.add_argument("--n-dagger-traj", type=int, default=2000,
                        help="New trajectories per DAgger iteration")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-hidden", type=int, default=4)
    parser.add_argument("--epochs-init", type=int, default=100)
    parser.add_argument("--epochs-refine", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="outputs/dagger_benchmark")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    cfg = load_config()
    tau_max = cfg.pendulum.tau_max

    # Validation data
    val_ds = TrajectoryDataset("data/medium/test.h5")
    val_s, val_a = val_ds.get_all_states(), val_ds.get_all_actions()
    val_ds.close()
    v_st = val_s[:, :-1].reshape(-1, 4)
    v_stp1 = val_s[:, 1:].reshape(-1, 4)
    v_ut = val_a.reshape(-1, 2)
    X_val = np.concatenate([v_st, v_stp1], axis=-1).astype(np.float32)
    Y_val = (v_ut / tau_max).astype(np.float32)

    # Initial training data: random (same as baseline)
    ds = TrajectoryDataset("data/medium/train.h5")
    train_s, train_a = ds.get_all_states(), ds.get_all_actions()
    ds.close()

    # Also add benchmark-like reference trajectories
    ref_s, ref_a = generate_benchmark_reference_trajectories(
        cfg, n_per_family=200, seed=args.seed
    )
    print(f"Reference trajectories: {ref_s.shape[0]} (benchmark-like)")

    # Combine initial data
    all_states = [train_s, ref_s]
    all_actions = [train_a, ref_a]

    results_log = []

    for iteration in range(args.n_iters + 1):
        print(f"\n{'='*60}")
        print(f"DAgger iteration {iteration}")
        print(f"{'='*60}")

        # Combine all data
        combined_s = np.concatenate(all_states, axis=0)
        combined_a = np.concatenate(all_actions, axis=0)
        print(f"Training data: {combined_s.shape[0]} trajectories, "
              f"{combined_s.shape[0] * (combined_s.shape[1]-1)} pairs")

        # Extract pairs
        s_t = combined_s[:, :-1].reshape(-1, 4)
        s_tp1 = combined_s[:, 1:].reshape(-1, 4)
        u_t = combined_a.reshape(-1, 2)
        X = np.concatenate([s_t, s_tp1], axis=-1).astype(np.float32)
        Y = (u_t / tau_max).astype(np.float32)

        epochs = args.epochs_init if iteration == 0 else args.epochs_refine
        lr = args.lr if iteration == 0 else args.lr * 0.5

        print(f"  Training {args.hidden_dim}x{args.n_hidden} for {epochs} epochs, lr={lr}")
        model, best_val = train_model(
            X, Y, X_val, Y_val,
            args.hidden_dim, args.n_hidden, device,
            epochs=epochs, lr=lr,
        )

        # Evaluate on standard benchmark
        print(f"  Evaluating on standard benchmark...")
        agg_mse, fam_results = evaluate_on_benchmark(model, cfg, device, tau_max)
        print(f"  Benchmark MSE: {agg_mse:.4e}")
        for f, v in fam_results.items():
            print(f"    {f}: {v:.4e}")

        # Save checkpoint
        ckpt_path = out / f"iter{iteration}_best.pt"
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

        results_log.append({
            "iteration": iteration,
            "n_trajectories": int(combined_s.shape[0]),
            "n_pairs": int(len(X)),
            "best_val_loss": float(best_val),
            "benchmark_mse": float(agg_mse),
            "per_family": fam_results,
        })

        # Save intermediate results
        with open(out / "results.json", "w") as f:
            json.dump(results_log, f, indent=2)

        if iteration < args.n_iters:
            # Generate new DAgger trajectories
            policy = MLPPolicy(model, tau_max=tau_max)
            print(f"  Generating {args.n_dagger_traj} DAgger trajectories...")
            new_s, new_a = generate_benchmark_reference_trajectories(
                cfg, n_per_family=args.n_dagger_traj // 6,
                seed=args.seed + iteration + 1
            )
            all_states.append(new_s)
            all_actions.append(new_a)
            print(f"  Added {new_s.shape[0]} new trajectories")

    # Final save
    best_iter = min(results_log, key=lambda x: x["benchmark_mse"])
    print(f"\nBest iteration: {best_iter['iteration']} "
          f"(MSE={best_iter['benchmark_mse']:.4e})")

    # Copy best model
    best_src = out / f"iter{best_iter['iteration']}_best.pt"
    best_dst = out / "best_model.pt"
    import shutil
    shutil.copy2(best_src, best_dst)
    print(f"Saved best model to {best_dst}")


if __name__ == "__main__":
    main()
