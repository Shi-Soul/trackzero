#!/usr/bin/env python3
"""PD-gain prediction experiment.

Hypothesis: Instead of predicting raw torques τ = π(s, s_ref), predict
state-dependent PD gains: (Kp, Kd) = π(s, s_ref), then compute
  τ = Kp * (q_ref - q) + Kd * (v_ref - v)

This encodes the physical structure of feedback control into the
architecture, potentially improving generalization to unseen reference
families (especially step/random_walk where large tracking errors occur).

Comparison:
  1. Raw MLP:     π(s, s_ref) → τ ∈ R^2
  2. PD-gain MLP: π(s, s_ref) → (Kp, Kd) ∈ R^(2×2), τ = Kp*(q_ref-q) + Kd*(v_ref-v)
  3. Residual-PD:  π(s, s_ref) → (Kp, Kd, τ_ff) ∈ R^(2×3), τ = Kp*Δq + Kd*Δv + τ_ff

All use same 1024×6 + cosine+WD + 2K data for fair comparison.
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

BENCH_SEED, N_PER = 12345, 100


class PDGainMLP(nn.Module):
    """Predicts PD gains instead of raw torques."""

    def __init__(self, state_dim, action_dim, mode="pd",
                 hidden_dim=1024, n_hidden=6):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.nq = state_dim // 2
        self.mode = mode  # "pd" or "residual_pd"

        input_dim = 2 * state_dim  # [s_t, s_ref]
        if mode == "pd":
            out_dim = 2 * self.nq  # Kp(nq) + Kd(nq)
        else:  # residual_pd
            out_dim = 3 * self.nq  # Kp(nq) + Kd(nq) + tau_ff(nq)

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))

    def set_normalization(self, mean, std):
        self.input_mean.copy_(torch.from_numpy(mean).float())
        self.input_std.copy_(torch.from_numpy(std).float())

    def forward(self, x):
        """x: (B, 2*state_dim) = [s_t, s_ref].
        Returns torques (B, nq)."""
        nq = self.nq
        s_t = x[:, :self.state_dim]
        s_ref = x[:, self.state_dim:]
        q_t, v_t = s_t[:, :nq], s_t[:, nq:]
        q_ref, v_ref = s_ref[:, :nq], s_ref[:, nq:]

        x_norm = (x - self.input_mean) / self.input_std
        out = self.net(x_norm)

        # Use softplus to keep gains positive
        Kp = nn.functional.softplus(out[:, :nq])
        Kd = nn.functional.softplus(out[:, nq:2*nq])

        delta_q = q_ref - q_t
        delta_v = v_ref - v_t
        tau = Kp * delta_q + Kd * delta_v

        if self.mode == "residual_pd":
            tau_ff = out[:, 2*nq:]
            tau = tau + tau_ff

        return tau


class PDGainPolicy:
    """Wraps PDGainMLP for evaluation harness."""

    def __init__(self, model, tau_max, device):
        self.model = model.eval()
        self.tau_max = tau_max
        self.device = device

    def __call__(self, current_state, ref_next_state):
        x = np.concatenate([current_state, ref_next_state]).astype(np.float32)
        with torch.no_grad():
            xt = torch.from_numpy(x).unsqueeze(0).to(self.device)
            a = self.model(xt).cpu().numpy().squeeze()
        return np.clip(a, -self.tau_max, self.tau_max)


def benchmark(model, cfg, device):
    harness = EvalHarness(cfg)
    tau_max = cfg.pendulum.tau_max
    policy = PDGainPolicy(model, tau_max, device)

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
        results[fname] = {"mean_mse": float(np.mean(mses))}

    agg = float(np.mean([v["mean_mse"] for v in results.values()]))
    results["_aggregate"] = {"mean_mse": agg}
    return results


def train_and_eval(mode, device, cfg, seed=42, epochs=100):
    print(f"\n{'='*50}")
    print(f"Mode: {mode}")
    print(f"{'='*50}")

    t0 = time.time()
    ds = TrajectoryDataset("data/medium/train.h5")
    all_s, all_a = ds.get_all_states(), ds.get_all_actions()
    ds.close()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_s), size=2000, replace=False)
    states, actions = all_s[idx], all_a[idx]

    # Standard pairs: [s_t, s_{t+1}] -> u_t
    s_t = states[:, :-1].reshape(-1, 4).astype(np.float32)
    s_tp1 = states[:, 1:].reshape(-1, 4).astype(np.float32)
    u_t = actions.reshape(-1, 2).astype(np.float32)
    X = np.concatenate([s_t, s_tp1], axis=-1)
    Y = u_t
    print(f"  Data: {X.shape[0]:,} pairs, {X.shape[1]}D → {Y.shape[1]}D")

    sd = states.shape[-1]
    model = PDGainMLP(state_dim=sd, action_dim=2, mode=mode,
                      hidden_dim=1024, n_hidden=6).to(device)
    npar = sum(p.numel() for p in model.parameters())
    print(f"  Model: {npar:,} params ({mode})")

    xm, xs = X.mean(0), X.std(0) + 1e-8
    model.set_normalization(xm, xs)

    Xt = torch.from_numpy(X).to(device)
    Yt = torch.from_numpy(Y).to(device)

    # Val
    ds_val = TrajectoryDataset("data/medium/test.h5")
    val_s, val_a = ds_val.get_all_states()[:200], ds_val.get_all_actions()[:200]
    ds_val.close()
    vs_t = val_s[:, :-1].reshape(-1, 4).astype(np.float32)
    vs_tp1 = val_s[:, 1:].reshape(-1, 4).astype(np.float32)
    vu_t = val_a.reshape(-1, 2).astype(np.float32)
    Xv = torch.from_numpy(np.concatenate([vs_t, vs_tp1], axis=-1)).to(device)
    Yv = torch.from_numpy(vu_t).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                         eta_min=1e-6)
    bs = 8192
    best_vl = float("inf")
    best_sd = None

    print(f"  Training {epochs} epochs...")
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        el, nb = 0.0, max(1, len(Xt) // bs)
        for b in range(nb):
            bx = Xt[perm[b*bs:(b+1)*bs]]
            by = Yt[perm[b*bs:(b+1)*bs]]
            pred = model(bx)
            loss = torch.nn.functional.mse_loss(pred, by)
            opt.zero_grad()
            loss.backward()
            opt.step()
            el += loss.item()
        model.eval()
        with torch.no_grad():
            vl = torch.nn.functional.mse_loss(model(Xv), Yv).item()
        if vl < best_vl:
            best_vl = vl
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        sched.step()
        if ep % 10 == 0 or ep == 1:
            print(f"    Ep {ep:3d}: train={el/nb:.6f} val={vl:.6f} "
                  f"best={best_vl:.6f} [{time.time()-t0:.0f}s]")

    model.load_state_dict(best_sd)
    model.eval()

    print(f"  Benchmarking...")
    results = benchmark(model, cfg, device)
    for k, v in results.items():
        print(f"    {k:15s}: {v['mean_mse']:.4e}")

    return {"mode": mode, "best_val": best_vl, "benchmark": results,
            "n_params": npar, "train_time": time.time() - t0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    args = p.parse_args()
    device = torch.device(args.device)
    cfg = load_config()

    all_results = {}
    for mode in ["pd", "residual_pd"]:
        r = train_and_eval(mode, device, cfg, args.seed, args.epochs)
        all_results[mode] = r

    print(f"\n{'='*60}")
    print("SUMMARY: PD-gain prediction")
    print(f"{'='*60}")
    for name, r in all_results.items():
        agg = r["benchmark"]["_aggregate"]["mean_mse"]
        print(f"  {name}: AGG={agg:.4e} (val={r['best_val']:.6f})")

    outdir = Path("outputs/pd_gain_ablation")
    outdir.mkdir(exist_ok=True, parents=True)
    (outdir / "results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved to {outdir}")


if __name__ == "__main__":
    main()
