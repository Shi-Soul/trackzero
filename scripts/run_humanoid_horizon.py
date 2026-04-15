#!/usr/bin/env python3
"""Diagnose humanoid failure: compounding error vs per-step error.

Trains raw_mlp_contact (best humanoid config) once, then benchmarks
at horizons 10, 25, 50, 100, 200, 500. If short horizons succeed
but long horizons diverge, the per-step accuracy is adequate and
the failure is purely compounding error.

Also saves the trained model and computes per-step MSE distribution
to quantify single-step accuracy.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.run_humanoid_horizon
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

OUT = "outputs/humanoid_horizon"
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── PlainMLP (same as run_humanoid) ──────────────────────────────────
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


# ── Benchmark with variable horizon ─────────────────────────────────
def benchmark_horizon(model_nn, families, device, horizon):
    """Run closed-loop tracking for `horizon` steps, resetting to
    reference state every `horizon` steps."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    results = {}

    for fam, trajs in families.items():
        mses = []
        for ref_s, ref_a in trajs:
            T = len(ref_a)
            errs = []
            seg_start = 0
            while seg_start < T:
                # Reset to reference state at segment boundary
                flat_to_mj(ref_s[seg_start], m, d)
                seg_end = min(seg_start + horizon, T)
                for t in range(seg_start, seg_end):
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
                    d.ctrl[:] = np.clip(tau, -TAU_MAX, TAU_MAX)
                    mujoco.mj_step(m, d)
                    actual = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                    if np.any(np.isnan(actual)):
                        errs.append(100.0)
                        break
                    errs.append(float(np.mean(
                        (actual - ref_s[t+1])**2)))
                seg_start = seg_end
            mses.append(np.mean(errs) if errs else 100.0)
        results[fam] = float(np.mean(mses))
    results["AGGREGATE"] = float(
        np.mean([results[k] for k in ["uniform", "step", "chirp"]]))
    return results


# ── Per-step accuracy analysis ───────────────────────────────────────
def per_step_accuracy(model_nn, device, n_traj=200, traj_len=50):
    """Measure single-step prediction accuracy (open-loop)."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    rng = np.random.RandomState(777)

    step_errors = []
    for _ in range(n_traj):
        mujoco.mj_resetData(m, d)
        d.qpos[:] += rng.randn(m.nq) * 0.01
        mujoco.mj_forward(m, d)

        for _ in range(traj_len):
            s = mj_to_flat(d.qpos.copy(), d.qvel.copy())
            if np.any(np.isnan(s)) or np.any(np.abs(s) > 50):
                break
            # Random torque to generate ground truth next state
            tau_scale = np.array(TAU_MAX) * 0.15
            tau = rng.uniform(-tau_scale, tau_scale)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            actual = mj_to_flat(d.qpos.copy(), d.qvel.copy())
            if np.any(np.isnan(actual)) or np.any(np.abs(actual) > 50):
                break
            # Predict torque from (s, actual) — what should the net
            # predict to go from s to actual?
            flags = get_contact_flags(d)
            inp = np.concatenate([s, actual, flags])
            with torch.no_grad():
                pred_tau = model_nn(
                    torch.tensor(inp, dtype=torch.float32)
                    .unsqueeze(0).to(device)
                ).cpu().numpy()[0]
            # Apply predicted torque from state s
            flat_to_mj(s, m, d)
            d.ctrl[:] = np.clip(pred_tau, -TAU_MAX, TAU_MAX)
            mujoco.mj_step(m, d)
            pred_actual = mj_to_flat(d.qpos.copy(), d.qvel.copy())
            if np.any(np.isnan(pred_actual)):
                step_errors.append(100.0)
                break
            err = float(np.mean((pred_actual - actual)**2))
            step_errors.append(err)
            # Reset to the ground truth actual for next step
            flat_to_mj(actual, m, d)

    return np.array(step_errors)


def main():
    print("=" * 60)
    print("HUMANOID HORIZON ANALYSIS")
    print("=" * 60)

    # ── Generate data ──
    print("\n[1] Generating training data...")
    t0 = time.time()
    S, SN, A, F = gen_data(2000, 500, seed=42)
    print(f"    {len(S)} pairs, {time.time()-t0:.0f}s")

    # ── Prepare dataset ──
    contact_dim = N_BODY_GEOMS
    in_dim = 2 * FLAT_STATE_DIM + contact_dim

    X = np.concatenate([S, SN, F], axis=1)
    Y = A

    # ── Train ──
    print("\n[2] Training raw_mlp_contact...")
    model = PlainMLP(in_dim, len(TAU_MAX), hidden=1024, layers=4).to(device)
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")

    t0 = time.time()
    train_model(model, X, Y, epochs=200, bs=2048, device=device)
    train_time = time.time() - t0
    print(f"    Training done in {train_time:.0f}s")

    # Save model
    torch.save(model.state_dict(), os.path.join(OUT, "model.pt"))
    print(f"    Model saved to {OUT}/model.pt")

    # ── Generate benchmark ──
    print("\n[3] Generating benchmark trajectories...")
    families = gen_benchmark(20, 500, seed=99)

    # ── Horizon sweep ──
    horizons = [1, 5, 10, 25, 50, 100, 200, 500]
    print("\n[4] Horizon sweep benchmark...")
    model.eval()
    horizon_results = {}
    for h in horizons:
        t0 = time.time()
        res = benchmark_horizon(model, families, device, h)
        dt = time.time() - t0
        horizon_results[str(h)] = res
        print(f"    H={h:4d}: AGG={res['AGGREGATE']:.4e}  "
              f"uni={res['uniform']:.4e}  "
              f"step={res['step']:.4e}  "
              f"chirp={res['chirp']:.4e}  "
              f"({dt:.0f}s)")

    # ── Per-step accuracy ──
    print("\n[5] Per-step accuracy analysis...")
    step_errs = per_step_accuracy(model, device)
    step_stats = {
        "mean": float(np.mean(step_errs)),
        "median": float(np.median(step_errs)),
        "p95": float(np.percentile(step_errs, 95)),
        "p99": float(np.percentile(step_errs, 99)),
        "max": float(np.max(step_errs)),
        "n_samples": len(step_errs),
        "frac_above_1": float(np.mean(step_errs > 1.0)),
        "frac_above_10": float(np.mean(step_errs > 10.0)),
    }
    print(f"    Mean step error: {step_stats['mean']:.6f}")
    print(f"    Median: {step_stats['median']:.6f}")
    print(f"    P95: {step_stats['p95']:.6f}")
    print(f"    P99: {step_stats['p99']:.6f}")
    print(f"    Max: {step_stats['max']:.6f}")
    print(f"    Frac > 1.0: {step_stats['frac_above_1']:.4f}")
    print(f"    Frac > 10.0: {step_stats['frac_above_10']:.4f}")

    # ── Save ──
    results = {
        "horizon_sweep": horizon_results,
        "step_accuracy": step_stats,
        "train_time": train_time,
    }
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT}/results.json")


if __name__ == "__main__":
    main()
