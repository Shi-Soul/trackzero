#!/usr/bin/env python3
"""Stage 1C: Max-entropy RL exploration data collection.

Train an exploration policy via PPO to maximize state visitation entropy.
Then collect data from this policy and train an inverse dynamics model.

This implements the proposal's "Maximum entropy exploration" (§1C):
  "Formulate data collection as an RL problem where the reward is the
   entropy of the visited state distribution."

The exploration policy IS NOT the tracking policy. It generates diverse
(state, action, next_state) training data for the ID model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from trackzero.config import load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/medium.yaml")
    p.add_argument("--val-data", default="data/medium/test.h5")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:1")
    # RL hyperparams
    p.add_argument("--n-envs", type=int, default=4096)
    p.add_argument("--rl-steps", type=int, default=200,
                   help="Number of PPO iterations")
    p.add_argument("--rollout-len", type=int, default=256,
                   help="Steps per rollout before PPO update")
    p.add_argument("--episode-len", type=int, default=500)
    p.add_argument("--v-range", type=float, default=15.0)
    p.add_argument("--density-bins", type=int, default=20)
    # ID model training
    p.add_argument("--collect-trajectories", type=int, default=10000,
                   help="Trajectories to collect from trained explorer")
    p.add_argument("--collect-traj-len", type=int, default=500)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--n-hidden", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=65536)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-trajectories", type=int, default=200)
    return p.parse_args()


def train_explorer(args, cfg, device, out):
    """Phase 1: Train max-entropy exploration policy via PPO."""
    from trackzero.rl import VecDoublePendulumEnv
    from trackzero.rl.density import HistogramDensity
    from trackzero.rl.ppo import ExplorationPolicy, ValueNet
    from trackzero.rl.train_loop import compute_gae, ppo_update

    print(f"=== Training exploration policy (PPO, {args.rl_steps} iters) ===")

    env = VecDoublePendulumEnv(
        cfg, n_envs=args.n_envs, episode_len=args.episode_len,
        device=device, v_range=(-args.v_range, args.v_range),
    )
    density = HistogramDensity(
        n_bins=args.density_bins, v_range=(-args.v_range, args.v_range),
        device=device,
    )
    policy = ExplorationPolicy(tau_max=cfg.pendulum.tau_max).to(device)
    vf = ValueNet().to(device)
    opt_pi = torch.optim.Adam(policy.parameters(), lr=3e-4)
    opt_v = torch.optim.Adam(vf.parameters(), lr=1e-3)

    obs = env.reset()
    log = []
    t0 = time.time()

    for it in range(1, args.rl_steps + 1):
        # Collect rollout
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = \
            [], [], [], [], [], []

        with torch.no_grad():
            for _ in range(args.rollout_len):
                action, logp = policy.get_action(obs)
                val = vf(obs)
                next_obs, done = env.step(action)
                # Entropy reward
                density.update_batched(next_obs)
                reward = density.reward(next_obs)

                obs_buf.append(obs)
                act_buf.append(action)
                logp_buf.append(logp)
                rew_buf.append(reward)
                done_buf.append(done)
                val_buf.append(val)

                obs = env.partial_reset(done)

            val_buf.append(vf(obs))  # bootstrap value

        obs_t = torch.stack(obs_buf)      # (T, N, 4)
        act_t = torch.stack(act_buf)      # (T, N, 2)
        logp_t = torch.stack(logp_buf)    # (T, N)
        rew_t = torch.stack(rew_buf)      # (T, N)
        done_t = torch.stack(done_buf)    # (T, N)
        val_t = torch.stack(val_buf)      # (T+1, N)

        # GAE per env, then flatten
        T, N = rew_t.shape
        advs, rets = compute_gae(rew_t, val_t, done_t)
        obs_flat = obs_t.reshape(-1, 4)
        act_flat = act_t.reshape(-1, 2)
        logp_flat = logp_t.reshape(-1)
        adv_flat = advs.reshape(-1)
        ret_flat = rets.reshape(-1)

        pi_loss, v_loss, ent = ppo_update(
            policy, vf, opt_pi, opt_v,
            obs_flat, act_flat, logp_flat, adv_flat, ret_flat,
        )

        cov = density.coverage_stats()
        elapsed = time.time() - t0
        total_steps = it * args.rollout_len * args.n_envs
        entry = {
            "iter": it, "pi_loss": pi_loss, "v_loss": v_loss,
            "entropy": ent, "mean_reward": rew_t.mean().item(),
            "coverage_pct": cov["coverage_pct"],
            "occupied_bins": cov["occupied_bins"],
            "hist_entropy": cov["entropy"],
            "total_steps": total_steps, "elapsed_s": elapsed,
        }
        log.append(entry)

        if it % 10 == 0 or it == 1:
            print(f"  [{it:4d}/{args.rl_steps}] "
                  f"cov={cov['coverage_pct']:.1f}% "
                  f"({cov['occupied_bins']}/{cov['total_bins']}) "
                  f"H={cov['entropy']:.2f} "
                  f"r={rew_t.mean():.4f} "
                  f"steps={total_steps/1e6:.1f}M "
                  f"[{elapsed:.0f}s]")

    # Save explorer
    torch.save(policy.state_dict(), out / "explorer_policy.pt")
    with open(out / "explorer_log.json", "w") as f:
        json.dump(log, f, indent=2)

    return policy, density, log


def collect_data(args, cfg, policy, device, out):
    """Phase 2: Collect trajectories from trained exploration policy."""
    from trackzero.sim.gpu_simulator import GPUSimulator

    print(f"\n=== Collecting {args.collect_trajectories} trajectories ===")
    N = args.collect_trajectories
    T = args.collect_traj_len
    tau_max = cfg.pendulum.tau_max

    gsim = GPUSimulator(cfg, n_worlds=min(N, 4096), device=device)
    rng = np.random.default_rng(args.seed)
    chunk = gsim.n_worlds

    all_states = []
    all_actions = []
    done = 0

    while done < N:
        bs = min(chunk, N - done)
        q0 = torch.empty(bs, 2, device=device).uniform_(-3.14159, 3.14159)
        v0 = torch.empty(bs, 2, device=device).uniform_(
            -args.v_range, args.v_range)
        # Pad to n_worlds
        nw = gsim.n_worlds
        q0_pad = torch.zeros(nw, 2, device=device)
        v0_pad = torch.zeros(nw, 2, device=device)
        q0_pad[:bs] = q0
        v0_pad[:bs] = v0
        gsim.reset_envs(q0_pad, v0_pad)

        states_list = [torch.cat([q0_pad[:bs].clone(),
                                   v0_pad[:bs].clone()], dim=-1).cpu()]
        actions_list = []

        obs = torch.cat([q0_pad, v0_pad], dim=-1)
        with torch.no_grad():
            for t in range(T):
                act, _ = policy.get_action(obs)
                qpos, qvel = gsim.step_envs(act)
                obs = torch.cat([qpos, qvel], dim=-1)
                states_list.append(obs[:bs].cpu())
                actions_list.append(act[:bs].cpu())

        traj_states = torch.stack(states_list, dim=1).numpy()  # (bs, T+1, 4)
        traj_actions = torch.stack(actions_list, dim=1).numpy()  # (bs, T, 2)
        all_states.append(traj_states)
        all_actions.append(traj_actions)
        done += bs
        print(f"  Collected {done}/{N}")

    states = np.concatenate(all_states, axis=0)[:N]
    actions = np.concatenate(all_actions, axis=0)[:N]
    print(f"  Data shape: states={states.shape}, actions={actions.shape}")

    np.save(out / "explorer_states.npy", states)
    np.save(out / "explorer_actions.npy", actions)
    return states, actions


def train_id_model(args, cfg, train_states, train_actions, device, out):
    """Phase 3: Train inverse dynamics model on explorer data."""
    from trackzero.data.dataset import TrajectoryDataset
    from trackzero.eval.harness import EvalHarness
    from trackzero.policy.mlp import MLPPolicy, load_checkpoint
    from trackzero.policy.train import TrainingConfig, train

    print(f"\n=== Training ID model on explorer data ===")
    val_ds = TrajectoryDataset(args.val_data)
    val_states = val_ds.get_all_states()
    val_actions = val_ds.get_all_actions()
    val_ds.close()

    tcfg = TrainingConfig(
        hidden_dim=args.hidden_dim, n_hidden=args.n_hidden,
        batch_size=args.batch_size, lr=args.lr,
        epochs=args.epochs, seed=args.seed,
        output_dir=str(out),
    )
    train(train_states, train_actions, val_states, val_actions,
          cfg=tcfg, tau_max=cfg.pendulum.tau_max, device=device)

    # Evaluate
    print("Evaluating...")
    model = load_checkpoint(out / "best_model.pt", device=device)
    policy = MLPPolicy(model, tau_max=cfg.pendulum.tau_max, device=device)
    harness = EvalHarness(cfg)
    summary = harness.evaluate_policy(
        policy, val_states, val_actions,
        max_trajectories=args.eval_trajectories)
    summary.to_json(out / "eval_results.json")
    return summary


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = args.device
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Phase 1: Train explorer
    policy, density, log = train_explorer(args, cfg, device, out)

    # Phase 2: Collect data
    states, actions = collect_data(args, cfg, policy, device, out)

    # Phase 3: Train ID model
    summary = train_id_model(args, cfg, states, actions, device, out)

    # Save metadata
    cov = density.coverage_stats()
    meta = {
        "method": "maxent_rl_exploration",
        "rl_steps": args.rl_steps,
        "n_envs": args.n_envs,
        "rollout_len": args.rollout_len,
        "total_rl_steps": args.rl_steps * args.rollout_len * args.n_envs,
        "collect_trajectories": args.collect_trajectories,
        "final_coverage_pct": cov["coverage_pct"],
        "final_hist_entropy": cov["entropy"],
        "final_eval": {
            "mean_mse_total": summary.mean_mse_total,
            "mean_mse_q": summary.mean_mse_q,
            "max_mse_total": summary.max_mse_total,
        },
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! ID mean_mse_total: {summary.mean_mse_total:.6e}")


if __name__ == "__main__":
    main()
