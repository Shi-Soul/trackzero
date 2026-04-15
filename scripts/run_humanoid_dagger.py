#!/usr/bin/env python3
"""DAgger for humanoid: iterative data collection to fix compounding error.

Standard DAgger loop:
  1. Train initial policy on random rollout data
  2. Roll out policy in closed loop (it diverges from training distribution)
  3. Query oracle (inverse dynamics) at visited states → new labels
  4. Aggregate new data with old data, retrain
  5. Repeat until convergence

The key insight: the policy visits states the random data never covers
(e.g., mid-fall states). By collecting oracle labels there, we close
the distribution gap that causes compounding error.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m scripts.run_humanoid_dagger
"""
import json, os, time
import mujoco
import numpy as np
import torch
import torch.nn as nn

from scripts.run_humanoid import (
    HUMANOID_XML, N_BODY_GEOMS, GEOM_OFFSET,
    TAU_MAX, LIMB_DOFS,
    mj_to_flat, flat_to_mj, get_contact_flags,
    gen_data, gen_benchmark,
)

FLAT_STATE_DIM = 54  # pos3 + rotvec3 + joints21 + vel27
from scripts.models_structured import _build_mlp
from scripts.run_structured import train_model

OUT = "outputs/humanoid_dagger"
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TAU_MAX_NP = np.array(TAU_MAX, dtype=np.float32)


class PlainMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=1024, layers=4):
        super().__init__()
        self.net = _build_mlp(in_dim, out_dim, hidden, layers)
        self.register_buffer("norm_m", torch.zeros(in_dim))
        self.register_buffer("norm_s", torch.ones(in_dim))

    def set_norm(self, m, s):
        self.norm_m = torch.tensor(m, dtype=torch.float32).to(
            next(self.parameters()).device)
        self.norm_s = torch.tensor(s, dtype=torch.float32).to(
            next(self.parameters()).device)

    def forward(self, x):
        return self.net((x - self.norm_m) / (self.norm_s + 1e-8))


def oracle_label(m, d, state, target_state):
    """Query inverse dynamics oracle: given current state and desired
    next state, what torque achieves it?

    We use MuJoCo's mj_inverse: set state + desired acceleration →
    get required forces. Since target_state encodes the desired
    next state, we compute the implied acceleration and use mj_inverse.
    """
    flat_to_mj(state, m, d)
    mujoco.mj_forward(m, d)

    # Compute desired acceleration from target state
    dt = m.opt.timestep
    # target_state = [pos3, rotvec3, joints21, vel27]
    # current vel = state[27:54], target vel = target_state[27:54]
    target_vel = target_state[27:54] if len(target_state) >= 54 else None
    current_vel = state[27:54] if len(state) >= 54 else None
    if target_vel is not None and current_vel is not None:
        desired_qacc = (target_vel - current_vel) / dt
        d.qacc[:] = desired_qacc
        mujoco.mj_inverse(m, d)
        tau = d.qfrc_inverse[6:].copy()  # skip freejoint DOFs
        return np.clip(tau, -TAU_MAX_NP, TAU_MAX_NP)
    return None


def collect_dagger_data(model_nn, families, device, n_traj=60,
                        max_steps=100, beta=0.5, seed=0):
    """Roll out policy, collect (state, target, oracle_action) tuples.

    With probability beta, use oracle action (exploration);
    with probability 1-beta, use policy action (on-policy states).
    Always label with oracle action.
    """
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    rng = np.random.RandomState(seed)

    new_data = []
    n_nan = 0

    all_trajs = []
    for fam, trajs in families.items():
        all_trajs.extend(trajs)

    for i in range(min(n_traj, len(all_trajs))):
        ref_s, ref_a = all_trajs[i % len(all_trajs)]
        T = min(max_steps, len(ref_a))
        flat_to_mj(ref_s[0], m, d)

        for t in range(T):
            s = mj_to_flat(d.qpos.copy(), d.qvel.copy())
            if np.any(np.isnan(s)) or np.any(np.abs(s) > 50):
                n_nan += 1
                break

            target = ref_s[t + 1]
            flags = get_contact_flags(d)

            # Get oracle label
            tau_oracle = oracle_label(m, d, s, target)
            if tau_oracle is None:
                break

            # Get policy prediction
            inp = np.concatenate([s, target, flags])
            with torch.no_grad():
                tau_policy = model_nn(
                    torch.tensor(inp, dtype=torch.float32)
                    .unsqueeze(0).to(device)
                ).cpu().numpy()[0]

            # Mix: use oracle with prob beta, policy with 1-beta
            if rng.random() < beta:
                tau_exec = tau_oracle
            else:
                tau_exec = np.clip(tau_policy, -TAU_MAX_NP, TAU_MAX_NP)

            # Record with oracle label
            new_data.append((s, target, tau_oracle, flags))

            # Execute
            d.ctrl[:] = tau_exec
            mujoco.mj_step(m, d)

    print(f"    DAgger collected {len(new_data)} pairs "
          f"({n_nan} NaN resets)")
    return new_data


def benchmark_humanoid(model_nn, families, device, horizon=500):
    """Benchmark with resettable horizon."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    results = {}

    for fam, trajs in families.items():
        mses = []
        for ref_s, ref_a in trajs:
            T = min(horizon, len(ref_a))
            flat_to_mj(ref_s[0], m, d)
            errs = []
            for t in range(T):
                if np.any(np.isnan(d.qpos)):
                    errs.append(100.0)
                    break
                s = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if np.any(np.isnan(s)):
                    errs.append(100.0)
                    break
                flags = get_contact_flags(d)
                inp = np.concatenate([s, ref_s[t+1], flags])
                with torch.no_grad():
                    tau = model_nn(
                        torch.tensor(inp, dtype=torch.float32)
                        .unsqueeze(0).to(device)
                    ).cpu().numpy()[0]
                d.ctrl[:] = np.clip(tau, -TAU_MAX_NP, TAU_MAX_NP)
                mujoco.mj_step(m, d)
                actual = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if np.any(np.isnan(actual)):
                    errs.append(100.0)
                    break
                errs.append(float(np.mean((actual - ref_s[t+1])**2)))
            mses.append(np.mean(errs) if errs else 100.0)
        results[fam] = float(np.mean(mses))
    results["AGGREGATE"] = float(
        np.mean([results[k] for k in ["uniform", "step", "chirp"]]))
    return results


def main():
    print("=" * 60)
    print("HUMANOID DAGGER EXPERIMENT")
    print("=" * 60)

    # ── Phase 0: Generate initial data ──
    print("\n[Phase 0] Generating initial data...")
    t0 = time.time()
    all_S, all_SN, all_A, all_F = gen_data(2000, 500, seed=42)
    print(f"    {len(all_S)} pairs in {time.time()-t0:.0f}s")

    contact_dim = N_BODY_GEOMS
    in_dim = 2 * FLAT_STATE_DIM + contact_dim

    # Generate benchmark
    print("    Generating benchmark...")
    families = gen_benchmark(20, 500, seed=99)

    # ── DAgger loop ──
    N_ROUNDS = 5
    dagger_results = {}
    beta_schedule = [1.0, 0.7, 0.5, 0.3, 0.1]  # decay oracle mixing

    for round_i in range(N_ROUNDS):
        beta = beta_schedule[min(round_i, len(beta_schedule)-1)]
        print(f"\n{'='*60}")
        print(f"DAgger Round {round_i} (beta={beta:.1f}, "
              f"dataset size={len(all_S)})")
        print(f"{'='*60}")

        # Build dataset
        X = np.concatenate([all_S, all_SN, all_F], axis=1)
        Y = all_A

        # Train
        model = PlainMLP(in_dim, len(TAU_MAX), hidden=1024,
                         layers=4).to(device)
        t0 = time.time()
        train_model(model, X, Y, epochs=200, bs=2048,
                    device=device)
        train_time = time.time() - t0
        print(f"    Training: {train_time:.0f}s")

        # Benchmark at different horizons
        model.eval()
        for h in [50, 200, 500]:
            res = benchmark_humanoid(model, families, device, horizon=h)
            print(f"    Benchmark H={h}: AGG={res['AGGREGATE']:.4e}")

        res_full = benchmark_humanoid(model, families, device, horizon=500)
        dagger_results[f"round_{round_i}"] = {
            "AGG": res_full["AGGREGATE"],
            "uniform": res_full["uniform"],
            "step": res_full["step"],
            "chirp": res_full["chirp"],
            "dataset_size": len(all_S),
            "train_time": train_time,
            "beta": beta,
        }

        # Save model
        torch.save(model.state_dict(),
                    os.path.join(OUT, f"model_round{round_i}.pt"))

        # Collect DAgger data (skip on last round)
        if round_i < N_ROUNDS - 1:
            print(f"\n    Collecting DAgger data (beta={beta:.1f})...")
            new_data = collect_dagger_data(
                model, families, device,
                n_traj=60, max_steps=100,
                beta=beta, seed=round_i*100)
            if new_data:
                new_S = np.array([d[0] for d in new_data])
                new_SN = np.array([d[1] for d in new_data])
                new_A = np.array([d[2] for d in new_data])
                new_F = np.array([d[3] for d in new_data])
                all_S = np.concatenate([all_S, new_S])
                all_SN = np.concatenate([all_SN, new_SN])
                all_A = np.concatenate([all_A, new_A])
                all_F = np.concatenate([all_F, new_F])
                print(f"    Dataset grew to {len(all_S)} pairs")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("DAGGER SUMMARY")
    print("=" * 60)
    for k, v in dagger_results.items():
        print(f"  {k}: AGG={v['AGG']:.4e}  "
              f"data={v['dataset_size']}  beta={v['beta']}")

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(dagger_results, f, indent=2)
    print(f"\nSaved to {OUT}/results.json")


if __name__ == "__main__":
    main()
