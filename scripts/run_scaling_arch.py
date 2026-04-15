#!/usr/bin/env python3
"""Residual PD with full 10K training data on 2-link.

Tests whether architecture improvement (residual PD) and data scaling
(2K→10K) compound. If so, this represents our best possible 2-link result.
"""

import json, time, sys, os
import numpy as np, torch, torch.nn as nn

sys.path.insert(0, ".")
from trackzero.config import Config
from trackzero.data.dataset import TrajectoryDataset
from trackzero.data.ood_references import generate_ood_reference_data
from trackzero.eval.harness import EvalHarness

FAMILIES = ["multisine", "chirp", "step", "random_walk", "sawtooth", "pulse"]
N_PER = 50
SEED = 12345


class ResidualPD(nn.Module):
    def __init__(self, sd, nq, hd=1024, nh=6):
        super().__init__()
        self.nq, self.sd = nq, sd
        layers = [nn.Linear(2*sd, hd), nn.ReLU()]
        for _ in range(nh-1):
            layers += [nn.Linear(hd, hd), nn.ReLU()]
        layers.append(nn.Linear(hd, 3*nq))
        self.net = nn.Sequential(*layers)
        self.register_buffer("mu", torch.zeros(2*sd))
        self.register_buffer("sigma", torch.ones(2*sd))

    def set_norm(self, m, s):
        self.mu.copy_(torch.from_numpy(m).float())
        self.sigma.copy_(torch.from_numpy(s).float())

    def forward(self, x):
        nq, sd = self.nq, self.sd
        q_t, v_t = x[:, :nq], x[:, nq:sd]
        q_ref, v_ref = x[:, sd:sd+nq], x[:, sd+nq:]
        out = self.net((x - self.mu) / self.sigma)
        Kp = nn.functional.softplus(out[:, :nq])
        Kd = nn.functional.softplus(out[:, nq:2*nq])
        tau_ff = out[:, 2*nq:]
        return Kp * (q_ref - q_t) + Kd * (v_ref - v_t) + tau_ff


class RawMLP(nn.Module):
    def __init__(self, sd, nq, hd=1024, nh=6):
        super().__init__()
        layers = [nn.Linear(2*sd, hd), nn.ReLU()]
        for _ in range(nh-1):
            layers += [nn.Linear(hd, hd), nn.ReLU()]
        layers.append(nn.Linear(hd, nq))
        self.net = nn.Sequential(*layers)
        self.register_buffer("mu", torch.zeros(2*sd))
        self.register_buffer("sigma", torch.ones(2*sd))

    def set_norm(self, m, s):
        self.mu.copy_(torch.from_numpy(m).float())
        self.sigma.copy_(torch.from_numpy(s).float())

    def forward(self, x):
        return self.net((x - self.mu) / self.sigma)


def main():
    device = torch.device("cuda:0")
    cfg = Config()
    nq = 2  # double pendulum
    sd = 4  # nq + nv
    harness = EvalHarness(cfg)

    print("=" * 60)
    print("Residual PD + Data Scaling (2K vs 10K)")
    print("=" * 60)

    # Load full training data
    ds = TrajectoryDataset("data/medium/train.h5")
    all_states = ds.get_all_states()  # (10000, 501, 4)
    all_actions = ds.get_all_actions()  # (10000, 500, 2)
    ds.close()

    # Load test data for val
    ds_test = TrajectoryDataset("data/medium/test.h5")
    test_states = ds_test.get_all_states()[:200]
    test_actions = ds_test.get_all_actions()[:200]
    ds_test.close()

    # Prepare val pairs
    vs_t = test_states[:, :-1].reshape(-1, sd).astype(np.float32)
    vs_tp1 = test_states[:, 1:].reshape(-1, sd).astype(np.float32)
    vu_t = test_actions.reshape(-1, nq).astype(np.float32)
    Xv = np.concatenate([vs_t, vs_tp1], axis=-1)

    # Generate benchmark
    print("Generating benchmark references...")
    bench = {}
    ds2 = TrajectoryDataset("data/medium/test.h5")
    ts, ta = ds2.get_all_states(), ds2.get_all_actions()
    ds2.close()
    bench["multisine"] = (ts[:N_PER], ta[:N_PER])
    for name in FAMILIES[1:]:
        s, a = generate_ood_reference_data(cfg, N_PER, action_type=name, seed=SEED)
        bench[name] = (s, a)

    # Data sizes to test
    data_sizes = [2000, 10000]
    archs = [("raw_mlp", RawMLP), ("residual_pd", ResidualPD)]
    results = {}

    for n_data in data_sizes:
        for arch_name, ArchCls in archs:
            key = f"{arch_name}_{n_data}"
            print(f"\n{'='*50}")
            print(f"{arch_name} with {n_data} trajectories")
            print(f"{'='*50}")

            # Prepare training data
            s_t = all_states[:n_data, :-1].reshape(-1, sd).astype(np.float32)
            s_tp1 = all_states[:n_data, 1:].reshape(-1, sd).astype(np.float32)
            u_t = all_actions[:n_data].reshape(-1, nq).astype(np.float32)
            X = np.concatenate([s_t, s_tp1], axis=-1)
            print(f"  Data: {X.shape[0]:,} pairs")

            model = ArchCls(sd, nq, 1024, 6).to(device)
            npar = sum(p.numel() for p in model.parameters())
            print(f"  Params: {npar:,}")

            # Normalize
            xm, xs = X.mean(0), X.std(0) + 1e-8
            model.set_norm(xm, xs)

            # Train
            Xt = torch.from_numpy(X).to(device)
            Yt = torch.from_numpy(u_t).to(device)
            Xvt = torch.from_numpy(Xv).to(device)
            Yvt = torch.from_numpy(vu_t).to(device)

            opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=1e-6)
            bs = 8192
            best_vl, best_sd_dict = float("inf"), None
            t0 = time.time()

            for ep in range(1, 201):
                model.train()
                perm = torch.randperm(len(Xt), device=device)
                el, nb = 0.0, max(1, len(Xt)//bs)
                for b in range(nb):
                    idx = perm[b*bs:(b+1)*bs]
                    loss = nn.functional.mse_loss(model(Xt[idx]), Yt[idx])
                    opt.zero_grad(); loss.backward(); opt.step()
                    el += loss.item()
                model.eval()
                with torch.no_grad():
                    vl = nn.functional.mse_loss(model(Xvt), Yvt).item()
                if vl < best_vl:
                    best_vl = vl
                    best_sd_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                sched.step()
                if ep % 20 == 0 or ep == 1:
                    print(f"    Ep {ep:3d}: t={el/nb:.5f} v={vl:.5f} best={best_vl:.5f} [{time.time()-t0:.0f}s]")

            model.load_state_dict(best_sd_dict)
            model.eval()

            # Benchmark
            print("  Benchmarking...")
            family_results = {}
            all_mses = []

            def make_policy(model):
                _mu = torch.from_numpy(xm).float().to(device)
                _std = torch.from_numpy(xs).float().to(device)
                def policy(s, r):
                    with torch.no_grad():
                        s_t = torch.tensor(s, dtype=torch.float32, device=device)
                        r_t = torch.tensor(r, dtype=torch.float32, device=device)
                        x = torch.cat([s_t, r_t]).unsqueeze(0)
                        return model(x).squeeze(0).cpu().numpy()
                return policy

            policy = make_policy(model)
            for fname in FAMILIES:
                ref_s, ref_a = bench[fname]
                n = min(N_PER, len(ref_s))
                fmses = []
                for i in range(n):
                    res = harness.evaluate_trajectory(policy, ref_s[i], ref_a[i])
                    fmses.append(res.mse_total)
                mean_mse = float(np.mean(fmses))
                family_results[fname] = mean_mse
                all_mses.extend(fmses)
                print(f"    {fname:12s}: {mean_mse:.4e}")

            agg = float(np.mean(all_mses))
            family_results["AGGREGATE"] = agg
            family_results["val_loss"] = best_vl
            results[key] = family_results
            print(f"    AGGREGATE   : {agg:.4e}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Architecture × Data Scaling")
    print(f"{'='*60}")
    for key, res in results.items():
        print(f"  {key:25s}: AGG={res['AGGREGATE']:.4e}")

    os.makedirs("outputs/scaling_arch", exist_ok=True)
    with open("outputs/scaling_arch/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to outputs/scaling_arch/results.json")


if __name__ == "__main__":
    main()
