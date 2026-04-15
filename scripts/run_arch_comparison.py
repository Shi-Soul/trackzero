#!/usr/bin/env python3
"""Unified architecture comparison on standard benchmark.

Fair head-to-head of all Stage 2 architectures using identical:
  - Training data: 2K trajectories (subsampled from train.h5, seed=42)
  - Architecture: 1024×6, cosine LR, WD=1e-4
  - Training: 100 epochs, bs=8192
  - Benchmark: 50 trajectories per family, 6 families

Architectures:
  1. raw_mlp:     π(s, s_ref) → τ
  2. residual_pd: π(s, s_ref) → (Kp, Kd, τ_ff), τ = Kp*Δq + Kd*Δv + τ_ff
  3. error_input: π(Δq, Δv, s_ref) → τ  (error-conditioned)
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from trackzero.config import load_config
from trackzero.data.dataset import TrajectoryDataset
from trackzero.data.ood_references import generate_ood_reference_data
from trackzero.eval.harness import EvalHarness

BENCH_SEED, N_PER = 12345, 50


# ── Architecture 1: Raw MLP (baseline) ─────────────────────────
class RawMLP(nn.Module):
    def __init__(self, sd, nu, hd=1024, nh=6):
        super().__init__()
        layers = [nn.Linear(2 * sd, hd), nn.ReLU()]
        for _ in range(nh - 1):
            layers += [nn.Linear(hd, hd), nn.ReLU()]
        layers.append(nn.Linear(hd, nu))
        self.net = nn.Sequential(*layers)
        self.register_buffer("mu", torch.zeros(2 * sd))
        self.register_buffer("sigma", torch.ones(2 * sd))

    def set_norm(self, m, s):
        self.mu.copy_(torch.from_numpy(m).float())
        self.sigma.copy_(torch.from_numpy(s).float())

    def forward(self, x):
        return self.net((x - self.mu) / self.sigma)


# ── Architecture 2: Residual PD ────────────────────────────────
class ResidualPD(nn.Module):
    def __init__(self, sd, nu, hd=1024, nh=6):
        super().__init__()
        self.nq = sd // 2
        layers = [nn.Linear(2 * sd, hd), nn.ReLU()]
        for _ in range(nh - 1):
            layers += [nn.Linear(hd, hd), nn.ReLU()]
        layers.append(nn.Linear(hd, 3 * self.nq))  # Kp, Kd, tau_ff
        self.net = nn.Sequential(*layers)
        self.register_buffer("mu", torch.zeros(2 * sd))
        self.register_buffer("sigma", torch.ones(2 * sd))
        self.sd = sd

    def set_norm(self, m, s):
        self.mu.copy_(torch.from_numpy(m).float())
        self.sigma.copy_(torch.from_numpy(s).float())

    def forward(self, x):
        nq = self.nq
        q_t, v_t = x[:, :nq], x[:, nq:self.sd]
        q_ref, v_ref = x[:, self.sd:self.sd + nq], x[:, self.sd + nq:]
        out = self.net((x - self.mu) / self.sigma)
        Kp = nn.functional.softplus(out[:, :nq])
        Kd = nn.functional.softplus(out[:, nq:2 * nq])
        tau_ff = out[:, 2 * nq:]
        return Kp * (q_ref - q_t) + Kd * (v_ref - v_t) + tau_ff


# ── Architecture 3: Error-conditioned ──────────────────────────
class ErrorMLP(nn.Module):
    """Input = [Δq, Δv, q_ref, v_ref] instead of [q, v, q_ref, v_ref]."""
    def __init__(self, sd, nu, hd=1024, nh=6):
        super().__init__()
        layers = [nn.Linear(2 * sd, hd), nn.ReLU()]
        for _ in range(nh - 1):
            layers += [nn.Linear(hd, hd), nn.ReLU()]
        layers.append(nn.Linear(hd, nu))
        self.net = nn.Sequential(*layers)
        self.register_buffer("mu", torch.zeros(2 * sd))
        self.register_buffer("sigma", torch.ones(2 * sd))
        self.sd = sd

    def set_norm(self, m, s):
        self.mu.copy_(torch.from_numpy(m).float())
        self.sigma.copy_(torch.from_numpy(s).float())

    def forward(self, x):
        return self.net((x - self.mu) / self.sigma)


def make_error_pairs(s_t, s_tp1):
    """Convert [s_t, s_ref] to [Δq, Δv, q_ref, v_ref]."""
    nq = s_t.shape[1] // 2
    dq = s_tp1[:, :nq] - s_t[:, :nq]
    dv = s_tp1[:, nq:] - s_t[:, nq:]
    return np.concatenate([dq, dv, s_tp1], axis=-1)


# ── Shared infrastructure ──────────────────────────────────────
class PolicyWrapper:
    def __init__(self, model, tau_max, device, transform=None):
        self.model = model.eval()
        self.tau_max = tau_max
        self.device = device
        self.transform = transform

    def __call__(self, state, ref):
        x = np.concatenate([state, ref]).astype(np.float32)
        if self.transform:
            x = self.transform(x)
        with torch.no_grad():
            xt = torch.from_numpy(x).unsqueeze(0).to(self.device)
            a = self.model(xt).cpu().numpy().squeeze()
        return np.clip(a, -self.tau_max, self.tau_max)


def error_transform(x):
    """Transform [s_t, s_ref] to [Δq, Δv, q_ref, v_ref]."""
    nq = len(x) // 4
    s_t, s_ref = x[:2 * nq], x[2 * nq:]
    dq = s_ref[:nq] - s_t[:nq]
    dv = s_ref[nq:] - s_t[nq:]
    return np.concatenate([dq, dv, s_ref]).astype(np.float32)


def run_benchmark(model, cfg, device, transform=None):
    harness = EvalHarness(cfg)
    tau_max = cfg.pendulum.tau_max
    policy = PolicyWrapper(model, tau_max, device, transform)

    ds = TrajectoryDataset("data/medium/test.h5")
    ms, ma = ds.get_all_states(), ds.get_all_actions()
    ds.close()
    families = {"multisine": (ms[:N_PER], ma[:N_PER])}
    for nm in ["chirp", "step", "random_walk", "sawtooth", "pulse"]:
        s, a = generate_ood_reference_data(cfg, N_PER, action_type=nm,
                                            seed=BENCH_SEED)
        families[nm] = (s, a)

    results = {}
    for fname, (ref_s, ref_a) in families.items():
        mses = []
        for i in range(len(ref_s)):
            r = harness.evaluate_trajectory(policy, ref_s[i], ref_a[i], i)
            mses.append(r.mse_total)
        results[fname] = float(np.mean(mses))
        print(f"    {fname:15s}: {results[fname]:.4e}")

    agg = float(np.mean(list(results.values())))
    results["_aggregate"] = agg
    print(f"    {'AGGREGATE':15s}: {agg:.4e}")
    return results


def train_model(model, X, Y, Xv, Yv, device, epochs=100):
    Xt = torch.from_numpy(X).to(device)
    Yt = torch.from_numpy(Y).to(device)
    Xvt = torch.from_numpy(Xv).to(device)
    Yvt = torch.from_numpy(Yv).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                         eta_min=1e-6)
    bs, best_vl, best_sd = 8192, float("inf"), None
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        el, nb = 0.0, max(1, len(Xt) // bs)
        for b in range(nb):
            idx = perm[b * bs:(b + 1) * bs]
            loss = nn.functional.mse_loss(model(Xt[idx]), Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item()
        model.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(model(Xvt), Yvt).item()
        if vl < best_vl:
            best_vl = vl
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        sched.step()
        if ep % 20 == 0 or ep == 1:
            print(f"    Ep {ep:3d}: train={el/nb:.6f} val={vl:.6f} "
                  f"best={best_vl:.6f} [{time.time()-t0:.0f}s]")

    model.load_state_dict(best_sd)
    model.eval()
    return best_vl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    args = p.parse_args()
    device = torch.device(args.device)
    cfg = load_config()

    # Load and subsample data (same for all architectures)
    ds = TrajectoryDataset("data/medium/train.h5")
    all_s, all_a = ds.get_all_states(), ds.get_all_actions()
    ds.close()
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(all_s), size=2000, replace=False)
    states, actions = all_s[idx], all_a[idx]

    # Standard pairs
    s_t = states[:, :-1].reshape(-1, 4).astype(np.float32)
    s_tp1 = states[:, 1:].reshape(-1, 4).astype(np.float32)
    u_t = actions.reshape(-1, 2).astype(np.float32)
    X_raw = np.concatenate([s_t, s_tp1], axis=-1)
    X_err = make_error_pairs(s_t, s_tp1)

    # Val
    ds_val = TrajectoryDataset("data/medium/test.h5")
    vs, va = ds_val.get_all_states()[:200], ds_val.get_all_actions()[:200]
    ds_val.close()
    vs_t = vs[:, :-1].reshape(-1, 4).astype(np.float32)
    vs_tp1 = vs[:, 1:].reshape(-1, 4).astype(np.float32)
    vu_t = va.reshape(-1, 2).astype(np.float32)
    Xv_raw = np.concatenate([vs_t, vs_tp1], axis=-1)
    Xv_err = make_error_pairs(vs_t, vs_tp1)

    print(f"Data: {X_raw.shape[0]:,} train pairs, {Xv_raw.shape[0]:,} val")

    all_results = {}
    architectures = [
        ("raw_mlp", RawMLP, X_raw, Xv_raw, None),
        ("residual_pd", ResidualPD, X_raw, Xv_raw, None),
        ("error_input", ErrorMLP, X_err, Xv_err, error_transform),
    ]

    for name, cls, X, Xv, transform in architectures:
        print(f"\n{'='*50}")
        print(f"Architecture: {name}")
        print(f"{'='*50}")
        model = cls(sd=4, nu=2).to(device)
        npar = sum(p.numel() for p in model.parameters())
        print(f"  Params: {npar:,}")

        xm, xs = X.mean(0), X.std(0) + 1e-8
        model.set_norm(xm, xs)

        best_vl = train_model(model, X, u_t, Xv, vu_t, device, args.epochs)
        print(f"  Best val: {best_vl:.6f}")
        print(f"  Benchmark:")
        bench = run_benchmark(model, cfg, device, transform)
        all_results[name] = {
            "val_loss": best_vl, "benchmark": bench, "n_params": npar
        }

    # Summary table
    print(f"\n{'='*60}")
    print("ARCHITECTURE COMPARISON (2K data, 50 trajs/family)")
    print(f"{'='*60}")
    print(f"{'Arch':15s} {'AGG':>10s} {'step':>10s} {'rnd_walk':>10s} {'val':>10s}")
    print("-" * 60)
    for name, r in all_results.items():
        b = r["benchmark"]
        print(f"{name:15s} {b['_aggregate']:10.4e} "
              f"{b.get('step', 0):10.4e} "
              f"{b.get('random_walk', 0):10.4e} "
              f"{r['val_loss']:10.6f}")

    outdir = Path("outputs/arch_comparison")
    outdir.mkdir(exist_ok=True, parents=True)
    (outdir / "results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved to {outdir}")


if __name__ == "__main__":
    main()
