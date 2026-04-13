#!/usr/bin/env python3
"""Stage 2C: Noise-augmented training for robust tracking.

Trains with corrupted references so the policy learns to handle
infeasible inputs gracefully instead of diverging.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from trackzero.config import load_config
from trackzero.data.random_rollout import generate_random_rollout_data
from trackzero.data.dataset import TrajectoryDataset
from trackzero.data.ood_references import generate_ood_reference_data
from trackzero.eval.harness import EvalHarness
from trackzero.policy.mlp import InverseDynamicsMLP, MLPPolicy

BENCH_SEED, N_PER = 12345, 100

def benchmark(model, cfg, device, tau_max):
    harness = EvalHarness(cfg)
    ds = TrajectoryDataset("data/medium/test.h5")
    ms, ma = ds.get_all_states(), ds.get_all_actions()
    ds.close()
    fams = {"multisine": (ms[:N_PER], ma[:N_PER])}
    for nm in ["chirp","step","random_walk","sawtooth","pulse"]:
        s, a = generate_ood_reference_data(cfg, N_PER, action_type=nm, seed=BENCH_SEED)
        fams[nm] = (s, a)
    policy = MLPPolicy(model, tau_max=tau_max, device=str(device))
    res = {}; all_m = []
    for fn, (rs, ra) in fams.items():
        sm = harness.evaluate_policy(policy, rs, ra, max_trajectories=N_PER)
        res[fn] = {"mean_mse": float(sm.mean_mse_total)}
        all_m.extend([r.mse_total for r in sm.results])
        print(f"  {fn:<15s}: {sm.mean_mse_total:.4e}")
    agg = float(np.mean(all_m))
    res["_aggregate"] = {"mean_mse": agg}
    print(f"  {'_aggregate':<15s}: {agg:.4e}")
    return res

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--device", default="cuda:0")
    pa.add_argument("--noise-std", type=float, default=0.05)
    pa.add_argument("--noise-frac", type=float, default=0.5,
                    help="Fraction of training pairs to corrupt")
    pa.add_argument("--n-traj", type=int, default=10000)
    pa.add_argument("--epochs", type=int, default=200)
    pa.add_argument("--seed", type=int, default=42)
    args = pa.parse_args()

    cfg = load_config()
    device = torch.device(args.device)
    tau_max = cfg.pendulum.tau_max
    tag = f"noise{args.noise_std}_frac{args.noise_frac}"
    out = Path(f"outputs/stage2c_{tag}"); out.mkdir(parents=True, exist_ok=True)

    # Generate data
    print(f"Generating {args.n_traj//1000}K trajectories...")
    t0 = time.time()
    train_s, train_a = generate_random_rollout_data(
        cfg, args.n_traj, action_type="mixed", seed=args.seed,
        use_gpu=True, gpu_device=args.device)
    print(f"  Done in {time.time()-t0:.0f}s")

    s_t = train_s[:, :-1].reshape(-1, 4)
    s_tp1 = train_s[:, 1:].reshape(-1, 4)
    u_t = train_a.reshape(-1, 2)

    # Corrupt a fraction of ref_next_states
    rng = np.random.default_rng(args.seed + 1)
    N = len(s_t)
    n_corrupt = int(N * args.noise_frac)
    corrupt_idx = rng.choice(N, n_corrupt, replace=False)
    s_tp1_aug = s_tp1.copy()
    s_tp1_aug[corrupt_idx, :2] += rng.normal(0, args.noise_std, (n_corrupt, 2))
    s_tp1_aug[corrupt_idx, 2:] += rng.normal(0, args.noise_std*5, (n_corrupt, 2))
    # For corrupted pairs, target is the CLEAN action (teaches graceful handling)
    X = np.concatenate([s_t, s_tp1_aug], axis=-1).astype(np.float32)
    Y = u_t.astype(np.float32)
    print(f"  Corrupted {n_corrupt}/{N} pairs ({args.noise_frac*100:.0f}%)")

    # Validation
    vds = TrajectoryDataset("data/medium/test.h5")
    vs, va = vds.get_all_states(), vds.get_all_actions()
    vds.close()
    Xv = np.concatenate([vs[:,:-1].reshape(-1,4), vs[:,1:].reshape(-1,4)],
                        axis=-1).astype(np.float32)
    Yv = va.reshape(-1, 2).astype(np.float32)

    model = InverseDynamicsMLP(4, 2, 1024, 6).to(device)
    Xt = torch.from_numpy(X).to(device)
    Yt = torch.from_numpy(Y).to(device)
    Xvt = torch.from_numpy(Xv).to(device)
    Yvt = torch.from_numpy(Yv).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    bs = 4096; best_vl = float("inf")

    print(f"\nTraining {args.epochs} epochs, cosine+wd...")
    t0 = time.time()
    for ep in range(1, args.epochs+1):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        el, nb = 0.0, len(Xt)//bs
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
            print(f"  Ep {ep:3d}: train={el/nb:.6f} val={vl:.6f} [{time.time()-t0:.0f}s]")

    ckpt = torch.load(out/"best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    print(f"\nBenchmark (noise-augmented, σ={args.noise_std}):")
    bench = benchmark(model, cfg, device, tau_max)

    # Also evaluate on noisy references
    print(f"\nNoisy-ref evaluation:")
    harness = EvalHarness(cfg)
    policy = MLPPolicy(model, tau_max=tau_max, device=str(device))
    noisy_res = {}
    rng2 = np.random.default_rng(99)
    for nl in [0.0, 0.05, 0.1, 0.2]:
        ds2 = TrajectoryDataset("data/medium/test.h5")
        ts, ta = ds2.get_all_states()[:50], ds2.get_all_actions()[:50]
        ds2.close()
        if nl > 0:
            ts2 = ts.copy()
            ts2[:, 1:, :2] += rng2.normal(0, nl, ts2[:, 1:, :2].shape)
            ts2[:, 1:, 2:] += rng2.normal(0, nl*5, ts2[:, 1:, 2:].shape)
        else:
            ts2 = ts
        sm = harness.evaluate_policy(policy, ts2, ta, max_trajectories=50)
        noisy_res[f"noise_{nl}"] = float(sm.mean_mse_total)
        print(f"  σ={nl:.2f}: {sm.mean_mse_total:.4e}")

    results = {"method": "noise_augmented", "noise_std": args.noise_std,
               "noise_frac": args.noise_frac, "best_val": float(best_vl),
               "benchmark": bench, "noisy_eval": noisy_res}
    with open(out/"results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone! Saved to {out}")

if __name__ == "__main__":
    main()
