#!/usr/bin/env python3
"""Stage 1D.3: Planning-Based Distillation.

For each reference trajectory, run CEM online to find optimal tracking torques.
Then distill the optimizer's behavior into a neural network.

This is the "gold standard" — it uses no motion data, just physics + optimization.
The question is whether amortized planning generalizes.

Algorithm:
  1. For each reference trajectory:
     a. Run CEM to find torque sequence minimizing tracking error
     b. Record (state, ref_next_state, optimal_torque) triples
  2. Train MLP on all (state, ref_next_state) → optimal_torque pairs
  3. Evaluate the distilled policy
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
from trackzero.policy.mlp import MLPPolicy, InverseDynamicsMLP
from trackzero.sim.simulator import Simulator


def cem_track_reference(
    sim: Simulator,
    ref_states: np.ndarray,
    tau_max: float,
    lookahead: int = 10,
    n_samples: int = 100,
    n_elite: int = 20,
    n_iters: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """CEM-based online tracker for a single reference trajectory.

    Uses receding-horizon CEM: at each step, optimize over next `lookahead`
    steps, execute first action, shift horizon.

    Returns: (actual_states, actions)
        actual_states: (T+1, 4) — actual trajectory
        actions: (T, 2) — applied torques
    """
    T = len(ref_states) - 1
    nq = 2

    q0, v0 = ref_states[0, :2], ref_states[0, 2:]
    sim.reset(q0=q0, v0=v0)

    actual_states = [sim.get_state()]
    actions = []

    # Warm-start mean with zeros
    mean = np.zeros((lookahead, nq))
    std_init = tau_max * 0.5

    for t in range(T):
        # Horizon for this step
        H = min(lookahead, T - t)
        ref_window = ref_states[t + 1: t + 1 + H]

        # CEM over H-step horizon
        mean_h = mean[:H]
        std_h = np.ones((H, nq)) * std_init

        best_cost = float("inf")
        best_seq = np.zeros((H, nq))

        current_state = sim.get_state()

        for it in range(n_iters):
            samples = (np.random.randn(n_samples, H, nq) * std_h + mean_h)
            samples = np.clip(samples, -tau_max, tau_max)

            costs = np.zeros(n_samples)
            for j in range(n_samples):
                sim.reset(
                    q0=current_state[:2], v0=current_state[2:]
                )
                for h in range(H):
                    s = sim.step(samples[j, h])
                    # Tracking cost
                    costs[j] += np.sum((s - ref_window[h]) ** 2)

            elite_idx = np.argsort(costs)[:n_elite]
            elite = samples[elite_idx]

            mean_h = elite.mean(axis=0)
            std_h = elite.std(axis=0) + 1e-4

            if costs[elite_idx[0]] < best_cost:
                best_cost = costs[elite_idx[0]]
                best_seq = samples[elite_idx[0]].copy()

        # Execute first action
        action = best_seq[0]
        sim.reset(q0=current_state[:2], v0=current_state[2:])
        next_state = sim.step(action)

        actions.append(action)
        actual_states.append(next_state)

        # Shift warm-start
        mean = np.zeros((lookahead, nq))
        if H > 1:
            mean[:H - 1] = best_seq[1:]

    return np.array(actual_states), np.array(actions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medium.yaml")
    parser.add_argument("--val-data", default="data/medium/test.h5")
    parser.add_argument("--train-data", default="data/medium/train.h5")
    parser.add_argument("--output-dir", default="outputs/stage1d_distill")
    parser.add_argument("--n-refs", type=int, default=200,
                        help="Number of references to run CEM on")
    parser.add_argument("--lookahead", type=int, default=10)
    parser.add_argument("--cem-samples", type=int, default=100)
    parser.add_argument("--cem-elite", type=int, default=20)
    parser.add_argument("--cem-iters", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-hidden", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
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
    sim = Simulator(cfg)

    # Load train references to run CEM on
    print("Loading training references...")
    train_ds = TrajectoryDataset(args.train_data)
    train_s = train_ds.get_all_states()
    train_ds.close()

    # Load validation data
    print("Loading validation data...")
    val_ds = TrajectoryDataset(args.val_data)
    val_s, val_a = val_ds.get_all_states(), val_ds.get_all_actions()
    val_ds.close()

    # Select diverse subset of references
    idx = np.random.choice(len(train_s), args.n_refs, replace=False)
    refs = train_s[idx]

    # Run CEM tracker on each reference
    print(f"\nRunning CEM tracker on {args.n_refs} references "
          f"(lookahead={args.lookahead})...")
    all_st, all_stp1, all_ut = [], [], []
    tracking_errors = []
    t0 = time.time()

    for i in range(args.n_refs):
        ref = refs[i]
        actual, actions = cem_track_reference(
            sim, ref, tau_max,
            lookahead=args.lookahead,
            n_samples=args.cem_samples,
            n_elite=args.cem_elite,
            n_iters=args.cem_iters,
        )

        # Compute tracking error
        T = len(actions)
        err = np.mean(np.sum((actual[1:T+1] - ref[1:T+1]) ** 2, axis=-1))
        tracking_errors.append(err)

        # Collect (actual_state, ref_next_state) → optimal_torque
        for t in range(T):
            all_st.append(actual[t])
            all_stp1.append(ref[t + 1])
            all_ut.append(actions[t])

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            me = np.mean(tracking_errors[-20:])
            print(f"  [{i+1}/{args.n_refs}] mean_err={me:.6f} "
                  f"time={elapsed:.0f}s")

    elapsed = time.time() - t0
    errs = np.array(tracking_errors)
    print(f"\nCEM tracking complete in {elapsed:.1f}s")
    print(f"  Tracking error: mean={errs.mean():.6f}, "
          f"median={np.median(errs):.6f}")
    print(f"  Training pairs: {len(all_st)}")

    # Use oracle labels instead of CEM actions (cleaner supervision)
    print("\nRelabeling with oracle...")
    oracle = InverseDynamicsOracle(cfg)
    all_st = np.array(all_st)
    all_stp1 = np.array(all_stp1)
    all_ut_oracle = np.zeros_like(np.array(all_ut))
    for i in range(len(all_st)):
        u = oracle.compute_torque(all_st[i], all_stp1[i])
        all_ut_oracle[i] = np.clip(u, -tau_max, tau_max)
    print("  Done.")

    # Train
    from torch.utils.data import TensorDataset, DataLoader

    model = InverseDynamicsMLP(
        state_dim=4, action_dim=2,
        hidden_dim=args.hidden_dim, n_hidden=args.n_hidden,
    ).to(args.device)

    # Prepare data
    X = np.concatenate([all_st, all_stp1], axis=-1).astype(np.float32)
    Y = (all_ut_oracle / tau_max).astype(np.float32)

    # Validation pairs
    v_st = val_s[:, :-1].reshape(-1, 4)
    v_stp1 = val_s[:, 1:].reshape(-1, 4)
    v_ut = val_a.reshape(-1, 2) / tau_max
    X_val = np.concatenate([v_st, v_stp1], axis=-1).astype(np.float32)
    Y_val = v_ut.astype(np.float32)

    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)),
        batch_size=args.batch_size, shuffle=True, pin_memory=True,
    )
    val_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)),
        batch_size=args.batch_size, shuffle=False, pin_memory=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None

    print(f"\nTraining on {len(X)} CEM-generated pairs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, nb = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(args.device), yb.to(args.device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            nb += 1

        model.eval()
        vl, nvb = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(args.device), yb.to(args.device)
                vl += loss_fn(model(xb), yb).item()
                nvb += 1

        t_avg = epoch_loss / max(nb, 1)
        v_avg = vl / max(nvb, 1)
        scheduler.step()

        if v_avg < best_val:
            best_val = v_avg
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  train={t_avg:.6f}  val={v_avg:.6f}")

    if best_state:
        model.load_state_dict(best_state)
    model.to(args.device)

    # Evaluate
    print("\nEvaluating distilled policy...")
    policy = MLPPolicy(model, tau_max=tau_max, device=args.device)
    harness = EvalHarness(cfg)
    summary = harness.evaluate_policy(
        policy, val_s, val_a, max_trajectories=200
    )
    print(f"  mean_mse={summary.mean_mse_total:.6e}  "
          f"max_mse={summary.max_mse_total:.6e}")
    print(f"  (Random baseline: 1.185e-4)")

    # Save
    torch.save(model.state_dict(), out / "best_model.pt")
    summary.to_json(out / "eval_results.json")

    meta = {
        "method": "planning_distillation",
        "n_refs": args.n_refs,
        "lookahead": args.lookahead,
        "total_pairs": int(len(X)),
        "cem_tracking_error": float(errs.mean()),
        "best_val_loss": float(best_val),
        "mean_mse_total": float(summary.mean_mse_total),
        "max_mse_total": float(summary.max_mse_total),
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    ratio = summary.mean_mse_total / 1.185e-4
    print(f"\nDone! Ratio vs random: {ratio:.1f}×")


if __name__ == "__main__":
    main()
