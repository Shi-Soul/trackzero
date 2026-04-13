#!/usr/bin/env python3
"""Stage 1 extension: Physics-informed residual architecture.

Hypothesis: If we feed the analytical inverse-dynamics estimate as an
additional input to the MLP, the network only needs to learn the
residual error, which should be easier and generalize better.

Architecture: MLP input = [state(4), ref_next(4), oracle_torque(2)] = 10D
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from trackzero.config import load_config
from trackzero.data.random_rollout import generate_random_rollout_data
from trackzero.data.dataset import TrajectoryDataset
from trackzero.data.ood_references import generate_ood_reference_data
from trackzero.oracle import InverseDynamicsOracle
from trackzero.eval.harness import EvalHarness
from trackzero.policy.mlp import InverseDynamicsMLP

BENCH_SEED, N_PER = 12345, 100

def oracle_features(oracle, s_t, s_tp1, tau_max):
    """Compute analytical ID torque for each transition."""
    N = len(s_t)
    feats = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        feats[i] = np.clip(oracle.compute_torque(s_t[i], s_tp1[i]),
                           -tau_max, tau_max)
    return feats

def benchmark(model, cfg, device, tau_max, oracle):
    """Run 6-family benchmark, return dict."""
    harness = EvalHarness(cfg)
    ds = TrajectoryDataset("data/medium/test.h5")
    ms, ma = ds.get_all_states(), ds.get_all_actions()
    ds.close()
    families = {"multisine": (ms[:N_PER], ma[:N_PER])}
    for nm in ["chirp","step","random_walk","sawtooth","pulse"]:
        s, a = generate_ood_reference_data(cfg, N_PER, action_type=nm, seed=BENCH_SEED)
        families[nm] = (s, a)

    # Wrap model as a residual policy
    class ResidualPolicy:
        def __init__(self, mdl, orc, tm, dev):
            self.mdl = mdl.eval(); self.orc = orc; self.tm = tm; self.dev = dev
        def __call__(self, cs, rns):
            orc_t = np.clip(self.orc.compute_torque(cs, rns), -self.tm, self.tm)
            x = np.concatenate([cs, rns, orc_t]).astype(np.float32)
            with torch.no_grad():
                xt = torch.from_numpy(x).unsqueeze(0).to(self.dev)
                a = self.mdl(xt).squeeze(0).cpu().numpy()
            return np.clip(a, -self.tm, self.tm)

    policy = ResidualPolicy(model, oracle, tau_max, device)
    results = {}; all_mse = []
    for fn, (rs, ra) in families.items():
        sm = harness.evaluate_policy(policy, rs, ra, max_trajectories=N_PER)
        results[fn] = {"mean_mse": float(sm.mean_mse_total)}
        all_mse.extend([r.mse_total for r in sm.results])
        print(f"  {fn:<15s}: {sm.mean_mse_total:.4e}")
    agg = float(np.mean(all_mse))
    results["_aggregate"] = {"mean_mse": agg}
    print(f"  {'_aggregate':<15s}: {agg:.4e}")
    return results

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--device", default="cuda:0")
    pa.add_argument("--n-traj", type=int, default=10000)
    pa.add_argument("--epochs", type=int, default=200)
    pa.add_argument("--seed", type=int, default=42)
    args = pa.parse_args()

    cfg = load_config()
    device = torch.device(args.device)
    tau_max = cfg.pendulum.tau_max
    oracle = InverseDynamicsOracle(cfg)
    out = Path("outputs/residual_10k"); out.mkdir(parents=True, exist_ok=True)

    # Generate data
    print("Generating 10K mixed trajectories...")
    t0 = time.time()
    train_s, train_a = generate_random_rollout_data(
        cfg, args.n_traj, action_type="mixed", seed=args.seed,
        use_gpu=True, gpu_device=args.device)
    print(f"  Done in {time.time()-t0:.0f}s")

    # Prepare pairs with oracle features
    s_t = train_s[:, :-1].reshape(-1, 4)
    s_tp1 = train_s[:, 1:].reshape(-1, 4)
    u_t = train_a.reshape(-1, 2)

    print("Computing oracle features...")
    orc_feat = oracle_features(oracle, s_t, s_tp1, tau_max)
    X = np.concatenate([s_t, s_tp1, orc_feat], axis=-1).astype(np.float32)
    Y = u_t.astype(np.float32)
    print(f"  Input dim: {X.shape[1]}, N={len(X)}")

    # Validation
    vds = TrajectoryDataset("data/medium/test.h5")
    vs, va = vds.get_all_states(), vds.get_all_actions()
    vds.close()
    vs_t = vs[:, :-1].reshape(-1, 4)
    vs_tp1 = vs[:, 1:].reshape(-1, 4)
    vu_t = va.reshape(-1, 2)
    orc_val = oracle_features(oracle, vs_t, vs_tp1, tau_max)
    Xv = np.concatenate([vs_t, vs_tp1, orc_val], axis=-1).astype(np.float32)
    Yv = vu_t.astype(np.float32)

    # Model: 10D input (8 state + 2 oracle torque)
    model = InverseDynamicsMLP(state_dim=5, action_dim=2,
                                hidden_dim=1024, n_hidden=6).to(device)
    npar = sum(p.numel() for p in model.parameters())
    print(f"Model: {npar:,} params (input_dim=10)")

    Xt = torch.from_numpy(X).to(device)
    Yt = torch.from_numpy(Y).to(device)
    Xvt = torch.from_numpy(Xv).to(device)
    Yvt = torch.from_numpy(Yv).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    bs = 4096; best_vl = float("inf")

    print(f"\nTraining {args.epochs} epochs...")
    t0 = time.time()
    for ep in range(1, args.epochs+1):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        el = 0.0; nb = len(Xt)//bs
        for b in range(nb):
            idx = perm[b*bs:(b+1)*bs]
            loss = torch.nn.functional.mse_loss(model(Xt[idx]), Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item()
        model.eval()
        with torch.no_grad():
            vt, vn = 0.0, 0
            for vb in range(0, len(Xvt), bs):
                xv, yv = Xvt[vb:vb+bs], Yvt[vb:vb+bs]
                vt += torch.nn.functional.mse_loss(model(xv), yv).item()*len(xv)
                vn += len(xv)
            vl = vt/vn
        if vl < best_vl:
            best_vl = vl
            torch.save({"model_state_dict": model.state_dict()}, out/"best_model.pt")
        sched.step()
        if ep % 50 == 0 or ep == 1:
            print(f"  Ep {ep:3d}: train={el/nb:.6f} val={vl:.6f} best={best_vl:.6f} [{time.time()-t0:.0f}s]")

    # Load best and benchmark
    ckpt = torch.load(out/"best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    print(f"\nBenchmark (best val={best_vl:.6f}):")
    bench = benchmark(model, cfg, device, tau_max, oracle)
    results = {"method": "residual_oracle_input", "best_val": float(best_vl),
               "train_time": time.time()-t0, "benchmark": bench}
    with open(out/"results.json", "w") as f: json.dump(results, f, indent=2)
    print(f"\nDone! Saved to {out}")

if __name__ == "__main__":
    main()
