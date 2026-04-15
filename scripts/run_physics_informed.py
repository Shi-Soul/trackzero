#!/usr/bin/env python3
"""Finding 28: Physics-informed architecture ablation for humanoid.

Tests whether leveraging physics structure in the network architecture
can improve inverse dynamics learning at 21 DOF.

Conditions:
1. baseline:     [s_t, s_{t+1}, contact] → τ  (current approach)
2. accel_feat:   [s_t, s_{t+1}, contact, q̈] → τ  (explicit acceleration)
3. delta_input:  [s_t, Δs, contact] → τ  (delta representation)
4. residual_pd:  τ = PD(s_t, s_{t+1}) + MLP([s_t, s_{t+1}, contact])
5. joint_weight: baseline + per-joint weighted loss (1/τ_max²)

All use 2K diverse trajectories, 1024×4 MLP, 200 epochs.
Budget: 1 GPU, ≤ 2h total.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m scripts.run_physics_informed
"""
import json, os, time
import numpy as np
import torch
import torch.nn as nn
import mujoco

from scripts.run_humanoid import (
    HUMANOID_XML, N_BODY_GEOMS, TAU_MAX,
    mj_to_flat, flat_to_mj, get_contact_flags,
    gen_benchmark,
)
from scripts.run_humanoid_entropy import gen_diverse_random
from scripts.models_structured import _build_mlp
from scripts.run_structured import train_model

FLAT_DIM = 54
VEL_START = 27  # flat[27:54] = velocities
N_JOINTS = 21   # actuated DOF
DT = 0.002      # simulation timestep
TAU_MAX_NP = np.array(TAU_MAX, dtype=np.float32)

OUT = "outputs/physics_informed"
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Models ───────────────────────────────────────────────────────────

class PlainMLP(nn.Module):
    """Standard MLP (baseline)."""
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


class ResidualPD(nn.Module):
    """τ = K_p(q_ref - q) + K_d(v_ref - v) + MLP(input).

    PD controller handles proportional tracking, MLP learns residual
    (gravity compensation, Coriolis, nonlinear inertia effects).
    K_p and K_d are learnable per-joint gains.
    """
    def __init__(self, in_dim, out_dim, hidden=1024, layers=4):
        super().__init__()
        self.net = _build_mlp(in_dim, out_dim, hidden, layers)
        self.register_buffer("norm_m", torch.zeros(in_dim))
        self.register_buffer("norm_s", torch.ones(in_dim))
        # Learnable PD gains (initialized to reasonable values)
        self.kp = nn.Parameter(torch.ones(out_dim) * 50.0)
        self.kd = nn.Parameter(torch.ones(out_dim) * 5.0)

    def set_norm(self, m, s):
        self.norm_m = torch.tensor(m, dtype=torch.float32).to(
            next(self.parameters()).device)
        self.norm_s = torch.tensor(s, dtype=torch.float32).to(
            next(self.parameters()).device)

    def forward(self, x):
        # x = [s_t(54), s_{t+1}(54), contact(15)]
        # Joint positions: s_t[6:27] (21D), s_{t+1}[6:27] (21D)
        # Joint velocities: s_t[27+6:54] (21D), s_{t+1}[27+6:54] (21D)
        q_cur = x[:, 6:27]      # current joint angles
        q_ref = x[:, 54+6:81]   # reference joint angles
        v_cur = x[:, 33:54]     # current joint velocities
        v_ref = x[:, 54+33:108] # reference joint velocities

        pd = self.kp * (q_ref - q_cur) + self.kd * (v_ref - v_cur)
        mlp_out = self.net(
            (x - self.norm_m) / (self.norm_s + 1e-8))
        return pd + mlp_out


# ── Data preparation ─────────────────────────────────────────────────

def prepare_inputs(S, SN, F, mode="baseline"):
    """Prepare model inputs depending on representation mode."""
    if mode == "baseline" or mode == "residual_pd" or mode == "joint_weight":
        # Standard: [s_t, s_{t+1}, contact]
        return np.concatenate([S, SN, F], axis=1)

    elif mode == "accel_feat":
        # Add acceleration: [s_t, s_{t+1}, contact, q̈]
        v_cur = S[:, VEL_START:]   # (N, 27)
        v_next = SN[:, VEL_START:] # (N, 27)
        accel = (v_next - v_cur) / DT  # (N, 27)
        return np.concatenate([S, SN, F, accel], axis=1)

    elif mode == "delta_input":
        # Delta: [s_t, Δs, contact]
        delta = SN - S  # (N, 54)
        return np.concatenate([S, delta, F], axis=1)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_input_dim(mode):
    """Input dimension for each representation mode."""
    base = 2 * FLAT_DIM + N_BODY_GEOMS  # 54+54+15 = 123
    if mode in ("baseline", "residual_pd", "joint_weight", "delta_input"):
        return base  # 123
    elif mode == "accel_feat":
        return base + 27  # 123 + 27 = 150
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ── Custom training for joint-weighted loss ──────────────────────────

def train_joint_weighted(model, X, Y, dev, epochs=200, bs=4096, lr=3e-4):
    """Train with per-joint weighted MSE loss."""
    n = len(X)
    n_val = max(int(n * 0.1), 1000)

    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y)
    Xv, Yv = Xt[:n_val].to(dev), Yt[:n_val].to(dev)
    Xtr, Ytr = Xt[n_val:].to(dev), Yt[n_val:].to(dev)

    xm = X[n_val:].mean(0)
    xs = X[n_val:].std(0) + 1e-8
    model.set_norm(xm, xs)
    model = model.to(dev)

    # Per-joint weights: 1/τ_max² (normalize low-torque joints matter more)
    w = 1.0 / (TAU_MAX_NP ** 2 + 1e-6)
    w = w / w.sum() * len(w)  # normalize so mean weight = 1
    weights = torch.tensor(w, dtype=torch.float32).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val, best_sd = float("inf"), None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), device=dev)
        el, nb = 0.0, 0
        for i in range(0, len(Xtr), bs):
            idx = perm[i : i + bs]
            pred = model(Xtr[idx])
            diff = (pred - Ytr[idx]) ** 2
            loss = (diff * weights).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            el += loss.item()
            nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            pred_v = model(Xv)
            diff_v = (pred_v - Yv) ** 2
            vl = (diff_v * weights).mean().item()
        if vl < best_val:
            best_val = vl
            best_sd = {k: v.cpu().clone()
                       for k, v in model.state_dict().items()}
        if ep % 20 == 0 or ep == 1:
            print(f"    Ep {ep:4d}: t={el/nb:.5f} v={vl:.5f} "
                  f"best={best_val:.5f}")

    model.load_state_dict(best_sd)
    model.eval()
    return model, best_val


# ── Evaluation (mode-aware) ───────────────────────────────────────────

def eval_h1_mode(model_nn, families, device, mode="baseline"):
    """Evaluate H=1 benchmark with mode-specific input construction."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    fam_errors = {}
    for fam, trajs in families.items():
        errors = []
        for ref_s, ref_a in trajs:
            step_errs = []
            for t in range(min(500, len(ref_a))):
                if np.any(np.isnan(ref_s[t])) or np.any(
                        np.isnan(ref_s[t+1])):
                    break
                flat_to_mj(ref_s[t], m, d)
                mujoco.mj_forward(m, d)
                flags = get_contact_flags(d)

                # Build input based on mode
                if mode in ("baseline", "residual_pd", "joint_weight"):
                    inp = np.concatenate([ref_s[t], ref_s[t+1], flags])
                elif mode == "accel_feat":
                    v_cur = ref_s[t][VEL_START:]
                    v_next = ref_s[t+1][VEL_START:]
                    accel = (v_next - v_cur) / DT
                    inp = np.concatenate(
                        [ref_s[t], ref_s[t+1], flags, accel])
                elif mode == "delta_input":
                    delta = ref_s[t+1] - ref_s[t]
                    inp = np.concatenate([ref_s[t], delta, flags])

                with torch.no_grad():
                    tau = model_nn(
                        torch.tensor(inp, dtype=torch.float32)
                        .unsqueeze(0).to(device)
                    ).cpu().numpy()[0]
                flat_to_mj(ref_s[t], m, d)
                d.ctrl[:] = np.clip(tau, -TAU_MAX_NP, TAU_MAX_NP)
                mujoco.mj_step(m, d)
                actual = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if np.any(np.isnan(actual)):
                    step_errs.append(100.0)
                else:
                    step_errs.append(float(np.mean(
                        (actual - ref_s[t+1])**2)))
            if step_errs:
                errors.append(float(np.mean(step_errs)))
        fam_errors[fam] = float(np.mean(errors)) if errors else 1e6
    return fam_errors


def eval_h500_mode(model_nn, families, device, mode="baseline"):
    """Evaluate H=500 full trajectory with mode-specific input."""
    m = mujoco.MjModel.from_xml_string(HUMANOID_XML)
    d = mujoco.MjData(m)
    fam_errors = {}
    for fam, trajs in families.items():
        errors = []
        for ref_s, ref_a in trajs:
            flat_to_mj(ref_s[0], m, d)
            mujoco.mj_forward(m, d)
            traj_err = 0.0
            T = min(500, len(ref_a))
            for t in range(T):
                s_now = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if np.any(np.isnan(s_now)):
                    traj_err += 100.0 * (T - t)
                    break
                flags = get_contact_flags(d)

                if mode in ("baseline", "residual_pd", "joint_weight"):
                    inp = np.concatenate([s_now, ref_s[t+1], flags])
                elif mode == "accel_feat":
                    v_cur = s_now[VEL_START:]
                    v_next = ref_s[t+1][VEL_START:]
                    accel = (v_next - v_cur) / DT
                    inp = np.concatenate(
                        [s_now, ref_s[t+1], flags, accel])
                elif mode == "delta_input":
                    delta = ref_s[t+1] - s_now
                    inp = np.concatenate([s_now, delta, flags])

                with torch.no_grad():
                    tau = model_nn(
                        torch.tensor(inp, dtype=torch.float32)
                        .unsqueeze(0).to(device)
                    ).cpu().numpy()[0]
                d.ctrl[:] = np.clip(tau, -TAU_MAX_NP, TAU_MAX_NP)
                mujoco.mj_step(m, d)
                actual = mj_to_flat(d.qpos.copy(), d.qvel.copy())
                if np.any(np.isnan(actual)):
                    traj_err += 100.0 * (T - t)
                    break
                traj_err += float(np.mean(
                    (actual - ref_s[t+1])**2))
            errors.append(traj_err / T)
        fam_errors[fam] = float(np.mean(errors)) if errors else 1e6
    return fam_errors


# ── Main experiment ──────────────────────────────────────────────────

def run():
    t0 = time.time()
    print("=== Finding 28: Physics-Informed Architecture Ablation ===")
    print(f"Device: {device}")

    # Generate diverse training data (same for all conditions)
    print("\n--- Generating diverse training data ---")
    S, SN, A, F = gen_diverse_random(2000, 500, seed=42)
    Y = A.copy()
    print(f"Data: {len(S):,} pairs")

    # Generate benchmark
    print("\n--- Generating benchmark ---")
    families = gen_benchmark(n_per_fam=10, traj_len=500, seed=99)
    print(f"Benchmark families: {list(families.keys())}")

    configs = [
        ("baseline",     "baseline"),
        ("accel_feat",   "accel_feat"),
        ("delta_input",  "delta_input"),
        ("residual_pd",  "residual_pd"),
        ("joint_weight", "joint_weight"),
    ]

    results = {}
    for name, mode in configs:
        print(f"\n{'='*60}")
        print(f"  Config: {name}")
        print(f"{'='*60}")
        ct = time.time()

        # Prepare inputs
        X = prepare_inputs(S, SN, F, mode)
        in_dim = get_input_dim(mode)
        assert X.shape[1] == in_dim, \
            f"Input dim mismatch: {X.shape[1]} vs {in_dim}"
        print(f"  Input dim: {in_dim}")

        # Create model
        if mode == "residual_pd":
            model = ResidualPD(in_dim, N_JOINTS, hidden=1024, layers=4)
        else:
            model = PlainMLP(in_dim, N_JOINTS, hidden=1024, layers=4)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params:,}")

        # Train
        if mode == "joint_weight":
            model, val = train_joint_weighted(model, X, Y, device)
        else:
            model, val = train_model(model, X, Y, device)

        # Eval H=1
        print("  Evaluating H=1...")
        h1 = eval_h1_mode(model, families, device, mode)
        h1_agg = float(np.mean(list(h1.values())))
        print(f"  H1: {h1} → AGG={h1_agg:.1f}")

        # Eval H=500
        print("  Evaluating H=500...")
        h500 = eval_h500_mode(model, families, device, mode)
        h500_agg = float(np.mean(list(h500.values())))
        print(f"  H500: {h500} → AGG={h500_agg:.2e}")

        elapsed = time.time() - ct
        results[name] = {
            "h1": h1, "h1_agg": h1_agg,
            "h500": h500, "h500_agg": h500_agg,
            "val_loss": val, "params": n_params,
            "input_dim": in_dim, "elapsed_s": elapsed,
        }
        print(f"  Done in {elapsed:.0f}s")

        # Save incrementally
        with open(f"{OUT}/results.json", "w") as f:
            json.dump(results, f, indent=2)

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"TOTAL: {total:.0f}s ({total/60:.1f}min)")
    print(f"\nSummary:")
    print(f"{'Config':<15} {'H1_AGG':>8} {'H500_AGG':>12} {'val_loss':>10}")
    for name, r in results.items():
        print(f"{name:<15} {r['h1_agg']:>8.1f} {r['h500_agg']:>12.2e} "
              f"{r['val_loss']:>10.5f}")


if __name__ == "__main__":
    run()
