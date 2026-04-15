#!/usr/bin/env python3
"""Multi-step reference conditioning experiment.

Hypothesis: Conditioning the policy on a window of K future reference
states (instead of just the next one) lets it anticipate upcoming
dynamics and plan ahead, improving tracking especially for step
references where discontinuities are visible in the future window.

Architecture comparison:
  K=1: [state(4), ref_{t+1}(4)]         = 8D   (standard)
  K=2: [state(4), ref_{t+1}(4), ref_{t+2}(4)] = 12D
  K=4: [state(4), ref_{t+1..t+4}(4×4)]  = 20D
  K=8: [state(4), ref_{t+1..t+8}(4×8)]  = 36D

All models use 1024×6, cosine+WD, 10K random rollout data.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from trackzero.config import load_config
from trackzero.data.random_rollout import generate_random_rollout_data
from trackzero.data.dataset import TrajectoryDataset
from trackzero.data.ood_references import generate_ood_reference_data
from trackzero.eval.harness import EvalHarness

BENCH_SEED, N_PER = 12345, 20


class MultiStepMLP(nn.Module):
    """MLP with K-step reference conditioning."""

    def __init__(self, state_dim, action_dim, K, hidden_dim=1024, n_hidden=6):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.K = K
        input_dim = state_dim + K * state_dim  # current + K future refs

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))

    def set_normalization(self, mean, std):
        self.input_mean.copy_(torch.from_numpy(mean).float())
        self.input_std.copy_(torch.from_numpy(std).float())

    def forward(self, x):
        x = (x - self.input_mean) / self.input_std
        return self.net(x)


def build_multistep_pairs(states, actions, K):
    """Build (state, ref_{t+1}..ref_{t+K}) -> action pairs.

    states: (N, T+1, state_dim)
    actions: (N, T, action_dim)
    Returns X: (N*(T-K+1), state_dim*(1+K)), Y: (N*(T-K+1), action_dim)
    """
    N, Tp1, sd = states.shape
    T = Tp1 - 1
    nu = actions.shape[-1]
    usable_T = T - K + 1  # can't use last K-1 steps (no future ref)
    if usable_T <= 0:
        raise ValueError(f"K={K} too large for T={T}")

    X_list, Y_list = [], []
    for t in range(usable_T):
        curr = states[:, t, :]  # (N, sd)
        refs = [states[:, t + 1 + k, :] for k in range(K)]  # K × (N, sd)
        x = np.concatenate([curr] + refs, axis=-1)  # (N, sd*(1+K))
        X_list.append(x)
        Y_list.append(actions[:, t, :])

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    Y = np.concatenate(Y_list, axis=0).astype(np.float32)
    return X, Y


class MultiStepPolicy:
    """Wraps MultiStepMLP for closed-loop evaluation."""

    def __init__(self, model, ref_states, tau_max, device):
        """ref_states: (T+1, state_dim) full reference trajectory."""
        self.model = model.eval()
        self.ref_states = ref_states
        self.tau_max = tau_max
        self.device = device
        self.K = model.K
        self.sd = model.state_dim
        self.t = 0

    def reset(self, ref_states):
        self.ref_states = ref_states
        self.t = 0

    def __call__(self, current_state, ref_next_state):
        """Called at each timestep during eval."""
        # Build multi-step input using stored reference trajectory
        refs = []
        T = len(self.ref_states) - 1
        for k in range(self.K):
            idx = min(self.t + 1 + k, T)
            refs.append(self.ref_states[idx])

        x = np.concatenate([current_state] + refs).astype(np.float32)
        with torch.no_grad():
            xt = torch.from_numpy(x).unsqueeze(0).to(self.device)
            a = self.model(xt).cpu().numpy().squeeze()
        self.t += 1
        return np.clip(a, -self.tau_max, self.tau_max)


def benchmark_multistep(model, cfg, device, K):
    """Run 6-family benchmark with multi-step policy."""
    harness = EvalHarness(cfg)
    tau_max = cfg.pendulum.tau_max

    ds = TrajectoryDataset("data/medium/test.h5")
    ms, ma = ds.get_all_states(), ds.get_all_actions()
    ds.close()
    families = {"multisine": (ms[:N_PER], ma[:N_PER])}
    for nm in ["chirp", "step", "random_walk", "sawtooth", "pulse"]:
        s, a = generate_ood_reference_data(cfg, N_PER, action_type=nm,
                                            seed=BENCH_SEED)
        families[nm] = (s, a)

    results = {}
    policy = MultiStepPolicy(model, None, tau_max, device)
    for fname, (ref_s, ref_a) in families.items():
        mses = []
        for i in range(len(ref_s)):
            policy.reset(ref_s[i])
            r = harness.evaluate_trajectory(policy, ref_s[i], ref_a[i], i)
            mses.append(r.mse_total)
        results[fname] = {"mean_mse": float(np.mean(mses))}

    agg = float(np.mean([v["mean_mse"] for v in results.values()]))
    results["_aggregate"] = {"mean_mse": agg}
    return results


def train_and_eval(K, device, cfg, seed=42, epochs=200):
    """Train and evaluate a K-step model."""
    print(f"\n{'='*50}")
    print(f"K={K}: {4 + K*4}D input")
    print(f"{'='*50}")

    t0 = time.time()
    # Use cached data — subsample to 2K trajectories for speed
    # (fair comparison across K values; 2K×500 = 1M pairs is sufficient)
    ds = TrajectoryDataset("data/medium/train.h5")
    all_s, all_a = ds.get_all_states(), ds.get_all_actions()
    ds.close()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_s), size=2000, replace=False)
    states, actions = all_s[idx], all_a[idx]
    print(f"  Loaded+subsampled data: {states.shape}")

    X, Y = build_multistep_pairs(states, actions, K)
    print(f"  Pairs: {X.shape[0]:,} × {X.shape[1]}D → {Y.shape[1]}D")

    sd = states.shape[-1]
    nu = actions.shape[-1]
    model = MultiStepMLP(state_dim=sd, action_dim=nu, K=K,
                          hidden_dim=1024, n_hidden=6).to(device)
    npar = sum(p.numel() for p in model.parameters())
    print(f"  Model: {npar:,} params")

    # Normalization
    xm, xs = X.mean(0), X.std(0) + 1e-8
    model.set_normalization(xm, xs)

    Xt = torch.from_numpy(X).to(device)
    Yt = torch.from_numpy(Y).to(device)

    # Val data — use test set (already cached), subsample
    ds_val = TrajectoryDataset("data/medium/test.h5")
    val_s, val_a = ds_val.get_all_states()[:200], ds_val.get_all_actions()[:200]
    ds_val.close()
    Xv, Yv = build_multistep_pairs(val_s, val_a, K)
    Xv = torch.from_numpy(Xv).to(device)
    Yv = torch.from_numpy(Yv).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                         eta_min=1e-6)
    bs = 8192
    best_vl = float("inf")
    best_sd = None

    print(f"  Training {epochs} epochs...")
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        el, nb = 0.0, max(1, len(Xt) // bs)
        for b in range(nb):
            idx = perm[b * bs:(b + 1) * bs]
            loss = torch.nn.functional.mse_loss(model(Xt[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            el += loss.item()
        model.eval()
        with torch.no_grad():
            vt, vn = 0.0, 0
            for vb in range(0, len(Xv), bs):
                vl = torch.nn.functional.mse_loss(
                    model(Xv[vb:vb + bs]), Yv[vb:vb + bs])
                n = min(bs, len(Xv) - vb)
                vt += vl.item() * n
                vn += n
            vt /= vn
        if vt < best_vl:
            best_vl = vt
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        sched.step()
        if ep % 10 == 0 or ep == 1:
            print(f"    Ep {ep:3d}: train={el/nb:.6f} val={vt:.6f} "
                  f"best={best_vl:.6f} [{time.time()-t0:.0f}s]")

    # Load best
    model.load_state_dict(best_sd)
    model.eval()

    # Benchmark
    print(f"  Benchmarking (best val={best_vl:.6f})...")
    results = benchmark_multistep(model, cfg, device, K)
    for k, v in results.items():
        print(f"    {k:15s}: {v['mean_mse']:.4e}")

    return {"K": K, "best_val": best_vl, "benchmark": results,
            "n_params": npar, "train_time": time.time() - t0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    args = p.parse_args()
    device = torch.device(args.device)
    cfg = load_config()

    all_results = {}
    for K in [1, 2, 4, 8]:
        r = train_and_eval(K, device, cfg, args.seed, args.epochs)
        all_results[f"K={K}"] = r

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Multi-step reference conditioning")
    print(f"{'='*60}")
    for name, r in all_results.items():
        agg = r["benchmark"]["_aggregate"]["mean_mse"]
        print(f"  {name}: AGG={agg:.4e} (val={r['best_val']:.6f})")

    outdir = Path("outputs/multistep_ablation")
    outdir.mkdir(exist_ok=True, parents=True)
    (outdir / "results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved to {outdir}")


if __name__ == "__main__":
    main()
