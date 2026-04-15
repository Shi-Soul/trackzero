#!/usr/bin/env python3
"""Residual PD + Multi-step conditioning (K future refs).

Stage 2 found: (1) multi-step hurts raw MLP (closed-loop gap),
(2) residual PD gives 9.7× improvement. Hypothesis: combining both
might rescue multi-step — the PD structure handles feedback correction
while multi-step helps the feedforward τ_ff plan ahead.
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


class ResidualPDMultistep(nn.Module):
    """Residual PD with K future reference states for feedforward."""
    def __init__(self, sd, nq, K, hd=1024, nh=6):
        super().__init__()
        self.nq, self.sd, self.K = nq, sd, K
        # Input: [s_t(sd), s_ref_1(sd), ..., s_ref_K(sd)] = (1+K)*sd
        in_dim = (1 + K) * sd
        layers = [nn.Linear(in_dim, hd), nn.ReLU()]
        for _ in range(nh-1):
            layers += [nn.Linear(hd, hd), nn.ReLU()]
        layers.append(nn.Linear(hd, 3*nq))
        self.net = nn.Sequential(*layers)
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sigma", torch.ones(in_dim))

    def set_norm(self, m, s):
        self.mu.copy_(torch.from_numpy(m).float())
        self.sigma.copy_(torch.from_numpy(s).float())

    def forward(self, x):
        nq, sd = self.nq, self.sd
        # Current state is first sd dims, first ref is next sd dims
        q_t = x[:, :nq]
        v_t = x[:, nq:sd]
        q_ref = x[:, sd:sd+nq]
        v_ref = x[:, sd+nq:2*sd]
        out = self.net((x - self.mu) / self.sigma)
        Kp = nn.functional.softplus(out[:, :nq])
        Kd = nn.functional.softplus(out[:, nq:2*nq])
        tau_ff = out[:, 2*nq:]
        return Kp * (q_ref - q_t) + Kd * (v_ref - v_t) + tau_ff


def build_multistep_pairs(states, actions, K):
    """Build (s_t, s_{t+1}, ..., s_{t+K}, u_t) training pairs."""
    N, T_plus_1, sd = states.shape
    T = T_plus_1 - 1
    nq = sd // 2
    valid_T = T - K + 1

    s_current = []
    s_refs = []  # list of K reference arrays
    u_targets = []

    for i in range(N):
        for t in range(valid_T):
            s_current.append(states[i, t])
            refs = []
            for k in range(1, K+1):
                refs.append(states[i, t+k])
            s_refs.append(np.concatenate(refs))
            u_targets.append(actions[i, t])

    X_current = np.array(s_current, dtype=np.float32)
    X_refs = np.array(s_refs, dtype=np.float32)
    X = np.concatenate([X_current, X_refs], axis=-1)
    Y = np.array(u_targets, dtype=np.float32)
    return X, Y


def main():
    device = torch.device("cuda:0")
    cfg = Config()
    nq = 2
    sd = 4
    harness = EvalHarness(cfg)

    print("=" * 60)
    print("Residual PD + Multi-step Conditioning")
    print("=" * 60)

    # Load training data (2K for budget compliance)
    ds = TrajectoryDataset("data/medium/train.h5")
    all_states = ds.get_all_states()[:2000]
    all_actions = ds.get_all_actions()[:2000]
    ds.close()

    # Load val data
    ds_test = TrajectoryDataset("data/medium/test.h5")
    val_states = ds_test.get_all_states()[:200]
    val_actions = ds_test.get_all_actions()[:200]
    ds_test.close()

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

    K_values = [1, 2, 4]
    results = {}

    for K in K_values:
        key = f"residual_pd_K{K}"
        print(f"\n{'='*50}")
        print(f"Residual PD with K={K} future refs")
        print(f"{'='*50}")

        # Build training pairs
        X, Y = build_multistep_pairs(all_states, all_actions, K)
        Xv, Yv = build_multistep_pairs(val_states, val_actions, K)
        print(f"  Data: {X.shape[0]:,} train, {Xv.shape[0]:,} val")
        print(f"  Input dim: {X.shape[1]}")

        model = ResidualPDMultistep(sd, nq, K, 1024, 6).to(device)
        npar = sum(p.numel() for p in model.parameters())
        print(f"  Params: {npar:,}")

        # Normalize
        xm, xs = X.mean(0), X.std(0) + 1e-8
        model.set_norm(xm, xs)

        # Train
        Xt = torch.from_numpy(X).to(device)
        Yt = torch.from_numpy(Y).to(device)
        Xvt = torch.from_numpy(Xv).to(device)
        Yvt = torch.from_numpy(Yv).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-6)
        bs = 8192
        best_vl, best_sd_dict = float("inf"), None
        t0 = time.time()

        for ep in range(1, 101):
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

        # Benchmark — need a policy wrapper that provides K future refs
        print("  Benchmarking...")

        def make_policy(model, K, xm, xs):
            _mu = torch.from_numpy(xm).float().to(device)
            _std = torch.from_numpy(xs).float().to(device)
            _K = K
            def policy(s, r):
                # At eval time, we only have s_t and s_ref_{t+1}
                # For K>1, we don't have future refs available in the harness
                # Repeat the next ref K times as approximation
                with torch.no_grad():
                    s_t = torch.tensor(s, dtype=torch.float32, device=device)
                    r_t = torch.tensor(r, dtype=torch.float32, device=device)
                    refs = r_t.repeat(_K)
                    x = torch.cat([s_t, refs]).unsqueeze(0)
                    return model(x).squeeze(0).cpu().numpy()
            return policy

        # Also make a proper multi-step policy using full reference trajectory
        family_results = {}
        all_mses = []

        for fname in FAMILIES:
            ref_s, ref_a = bench[fname]
            n = min(N_PER, len(ref_s))
            fmses = []
            for i in range(n):
                # Custom closed-loop evaluation with multi-step refs
                T_traj = len(ref_a[i])
                from trackzero.sim.simulator import Simulator
                sim = Simulator(cfg)
                sim.set_state(ref_s[i, 0])
                total_err = 0.0
                for t in range(T_traj):
                    state = sim.get_state()
                    # Build multi-step input with true future refs
                    refs = []
                    for k in range(1, K+1):
                        idx = min(t+k, T_traj)
                        refs.append(ref_s[i, idx])
                    refs = np.concatenate(refs).astype(np.float32)
                    s_arr = state.astype(np.float32)
                    x_in = np.concatenate([s_arr, refs])
                    with torch.no_grad():
                        x_t = torch.from_numpy(x_in).unsqueeze(0).to(device)
                        action = model(x_t).squeeze(0).cpu().numpy()
                    next_state = sim.step(action)
                    dq = next_state[:nq] - ref_s[i, t+1, :nq]
                    dv = next_state[nq:] - ref_s[i, t+1, nq:]
                    total_err += np.sum(dq**2) + 0.1 * np.sum(dv**2)
                mse = total_err / T_traj
                fmses.append(mse)

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
    print("SUMMARY: Residual PD + Multi-step")
    print(f"{'='*60}")
    for key, res in results.items():
        print(f"  {key:25s}: AGG={res['AGGREGATE']:.4e}")

    os.makedirs("outputs/pd_multistep", exist_ok=True)
    with open("outputs/pd_multistep/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to outputs/pd_multistep/results.json")


if __name__ == "__main__":
    main()
