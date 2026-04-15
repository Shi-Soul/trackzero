#!/usr/bin/env python3
"""Full-matrix PD for 5-link chain: testing whether cross-joint coupling
is why diagonal residual PD fails at higher DOF.

Architectures:
  1. raw_mlp: baseline
  2. diag_pd: Kp_i*(Δq_i) + Kd_i*(Δv_i) + τ_ff_i (diagonal, existing)
  3. full_pd: K_p @ Δq + K_d @ Δv + τ_ff (full nq×nq matrices)
  4. block_pd: Lower-triangular gains (causal coupling in chain)

The hypothesis: diagonal PD fails at 5 DOF because cross-joint coupling
through M(q) makes diagonal gains insufficient. Full-matrix PD should
capture this coupling while still providing the structural prior.
"""

import argparse, json, time, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, mujoco

sys.path.insert(0, ".")
from trackzero.config import load_config
from trackzero.sim.pendulum_model import build_chain_xml


class ChainSim:
    def __init__(self, cfg, n_links):
        xml = build_chain_xml(n_links, cfg.pendulum, cfg.simulation)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.nq = self.model.nq
        self.nu = self.model.nu
        self._sub = cfg.simulation.substeps
        self._tau = cfg.pendulum.tau_max

    def reset(self, q0, v0):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = q0
        self.data.qvel[:] = v0
        mujoco.mj_forward(self.model, self.data)
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, -self._tau, self._tau)
        for _ in range(self._sub):
            mujoco.mj_step(self.model, self.data)
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])


class RawMLP(nn.Module):
    def __init__(self, sd, nu, hd=1024, nh=6):
        super().__init__()
        layers = [nn.Linear(2*sd, hd), nn.ReLU()]
        for _ in range(nh-1):
            layers += [nn.Linear(hd, hd), nn.ReLU()]
        layers.append(nn.Linear(hd, nu))
        self.net = nn.Sequential(*layers)
        self.register_buffer("mu", torch.zeros(2*sd))
        self.register_buffer("sigma", torch.ones(2*sd))

    def set_norm(self, m, s):
        self.mu.copy_(torch.from_numpy(m).float())
        self.sigma.copy_(torch.from_numpy(s).float())

    def forward(self, x):
        return self.net((x - self.mu) / self.sigma)


class DiagPD(nn.Module):
    """Diagonal PD: Kp_i*Δq_i + Kd_i*Δv_i + τ_ff_i"""
    def __init__(self, sd, nu, hd=1024, nh=6):
        super().__init__()
        self.nq = sd // 2
        self.sd = sd
        layers = [nn.Linear(2*sd, hd), nn.ReLU()]
        for _ in range(nh-1):
            layers += [nn.Linear(hd, hd), nn.ReLU()]
        layers.append(nn.Linear(hd, 3*self.nq))
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


class FullMatrixPD(nn.Module):
    """Full-matrix PD: K_p @ Δq + K_d @ Δv + τ_ff
    K_p, K_d are nq×nq matrices (state-dependent, via network output).
    """
    def __init__(self, sd, nu, hd=1024, nh=6):
        super().__init__()
        self.nq = sd // 2
        self.sd = sd
        # Output: nq*nq (Kp) + nq*nq (Kd) + nq (tau_ff)
        out_dim = 2 * self.nq * self.nq + self.nq
        layers = [nn.Linear(2*sd, hd), nn.ReLU()]
        for _ in range(nh-1):
            layers += [nn.Linear(hd, hd), nn.ReLU()]
        layers.append(nn.Linear(hd, out_dim))
        self.net = nn.Sequential(*layers)
        self.register_buffer("mu", torch.zeros(2*sd))
        self.register_buffer("sigma", torch.ones(2*sd))

    def set_norm(self, m, s):
        self.mu.copy_(torch.from_numpy(m).float())
        self.sigma.copy_(torch.from_numpy(s).float())

    def forward(self, x):
        nq, sd = self.nq, self.sd
        B = x.shape[0]
        q_t, v_t = x[:, :nq], x[:, nq:sd]
        q_ref, v_ref = x[:, sd:sd+nq], x[:, sd+nq:]
        out = self.net((x - self.mu) / self.sigma)

        # Reshape into gain matrices
        Kp_flat = out[:, :nq*nq].reshape(B, nq, nq)
        Kd_flat = out[:, nq*nq:2*nq*nq].reshape(B, nq, nq)
        tau_ff = out[:, 2*nq*nq:]

        # Make gains positive-semi-definite via K = L @ L^T + eps*I
        # Simple approach: use softplus on diagonal, no constraint on off-diag
        dq = (q_ref - q_t).unsqueeze(-1)  # (B, nq, 1)
        dv = (v_ref - v_t).unsqueeze(-1)  # (B, nq, 1)

        tau_p = torch.bmm(Kp_flat, dq).squeeze(-1)  # (B, nq)
        tau_d = torch.bmm(Kd_flat, dv).squeeze(-1)
        return tau_p + tau_d + tau_ff


def gen_data(sim, n_traj, T, tau_max, seed):
    rng = np.random.default_rng(seed)
    nq, nu = sim.nq, sim.nu
    sd = nq * 2
    all_s = np.zeros((n_traj, T+1, sd))
    all_a = np.zeros((n_traj, T, nu))
    for i in range(n_traj):
        q0 = rng.uniform(-np.pi, np.pi, nq)
        v0 = rng.uniform(-3, 3, nq)
        st = ["uniform", "ou", "gaussian"][i % 3]
        if st == "uniform":
            a = rng.uniform(-tau_max, tau_max, (T, nu))
        elif st == "ou":
            a = np.zeros((T, nu))
            xou = rng.uniform(-tau_max*0.5, tau_max*0.5, nu)
            for t in range(T):
                xou = xou - 0.15*xou*0.01 + 0.5*tau_max*np.sqrt(0.01)*rng.standard_normal(nu)
                a[t] = np.clip(xou, -tau_max, tau_max)
        else:
            a = np.clip(rng.normal(0, tau_max/3, (T, nu)), -tau_max, tau_max)
        sim.reset(q0, v0)
        all_s[i, 0] = np.concatenate([sim.data.qpos.copy(), sim.data.qvel.copy()])
        for t in range(T):
            all_s[i, t+1] = sim.step(a[t])
        all_a[i] = a
    return all_s, all_a


def gen_ood(sim, n_traj, T, tau_max, seed, ood_type="step"):
    rng = np.random.default_rng(seed)
    nq, nu = sim.nq, sim.nu
    sd = nq * 2
    all_s = np.zeros((n_traj, T+1, sd))
    all_a = np.zeros((n_traj, T, nu))
    for i in range(n_traj):
        q0 = rng.uniform(-np.pi, np.pi, nq)
        v0 = rng.uniform(-3, 3, nq)
        if ood_type == "step":
            n_steps = int(rng.integers(3, 8))
            a = np.zeros((T, nu))
            for j in range(nu):
                sw = sorted(rng.integers(0, T, n_steps-1).tolist()) + [T]
                sw = [0] + sw
                lvls = rng.uniform(-tau_max, tau_max, n_steps)
                for k in range(n_steps):
                    a[sw[k]:sw[k+1], j] = lvls[k]
        elif ood_type == "chirp":
            t_arr = np.arange(T) * 0.01
            a = np.zeros((T, nu))
            for j in range(nu):
                f0 = rng.uniform(0.1, 0.5)
                f1 = rng.uniform(1.0, 5.0)
                amp = rng.uniform(0.3*tau_max, tau_max)
                ph = 2*np.pi*(f0*t_arr + 0.5*(f1-f0)*t_arr**2/max(T*0.01, 1e-9))
                a[:, j] = np.clip(amp*np.sin(ph), -tau_max, tau_max)
        else:
            a = rng.uniform(-tau_max, tau_max, (T, nu))
        sim.reset(q0, v0)
        all_s[i, 0] = np.concatenate([sim.data.qpos.copy(), sim.data.qvel.copy()])
        for t in range(T):
            all_s[i, t+1] = sim.step(a[t])
        all_a[i] = a
    return all_s, all_a


def eval_model(sim, model, ref_s, ref_a, tau_max, device):
    N, nq = len(ref_s), sim.nq
    mses = []
    for i in range(N):
        T = len(ref_a[i])
        sim.reset(ref_s[i, 0, :nq], ref_s[i, 0, nq:])
        actual = [np.concatenate([sim.data.qpos.copy(), sim.data.qvel.copy()])]
        for t in range(T):
            x = np.concatenate([actual[-1], ref_s[i, t+1]]).astype(np.float32)
            with torch.no_grad():
                a = model(torch.from_numpy(x).unsqueeze(0).to(device)).cpu().numpy().squeeze()
            actual.append(sim.step(np.clip(a, -tau_max, tau_max)))
        actual = np.array(actual)
        q_err = np.mean((ref_s[i, 1:, :nq] - actual[1:, :nq])**2)
        v_err = np.mean((ref_s[i, 1:, nq:] - actual[1:, nq:])**2)
        mses.append(q_err + 0.1 * v_err)
    return float(np.mean(mses))


def train(model, X, Y, Xv, Yv, device, epochs=100):
    Xt = torch.from_numpy(X).to(device)
    Yt = torch.from_numpy(Y).to(device)
    Xvt = torch.from_numpy(Xv).to(device)
    Yvt = torch.from_numpy(Yv).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    bs, best_vl, best_sd = 8192, float("inf"), None
    t0 = time.time()
    for ep in range(1, epochs+1):
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
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        sched.step()
        if ep % 20 == 0 or ep == 1:
            print(f"    Ep {ep:3d}: t={el/nb:.5f} v={vl:.5f} best={best_vl:.5f} [{time.time()-t0:.0f}s]")
    model.load_state_dict(best_sd)
    model.eval()
    return best_vl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-links", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--n-traj", type=int, default=2000)
    args = p.parse_args()
    device = torch.device(args.device)
    cfg = load_config()
    tau_max = cfg.pendulum.tau_max
    sim = ChainSim(cfg, args.n_links)
    nq, nu = sim.nq, sim.nu
    sd = nq * 2
    T = int(cfg.dataset.trajectory_duration / cfg.simulation.control_dt)

    print(f"{'='*60}")
    print(f"Full-matrix PD ablation on {args.n_links}-link chain")
    print(f"nq={nq}, sd={sd}, nu={nu}")
    print(f"{'='*60}")

    # Data
    print("Generating training data...")
    t0 = time.time()
    train_s, train_a = gen_data(sim, args.n_traj, T, tau_max, args.seed)
    print(f"  {args.n_traj} trajs in {time.time()-t0:.0f}s")
    s_t = train_s[:, :-1].reshape(-1, sd).astype(np.float32)
    s_tp1 = train_s[:, 1:].reshape(-1, sd).astype(np.float32)
    u_t = train_a.reshape(-1, nu).astype(np.float32)
    X = np.concatenate([s_t, s_tp1], axis=-1)

    print("Generating val data...")
    val_s, val_a = gen_data(sim, 200, T, tau_max, args.seed + 100)
    vs_t = val_s[:, :-1].reshape(-1, sd).astype(np.float32)
    vs_tp1 = val_s[:, 1:].reshape(-1, sd).astype(np.float32)
    vu_t = val_a.reshape(-1, nu).astype(np.float32)
    Xv = np.concatenate([vs_t, vs_tp1], axis=-1)

    print("Generating benchmark refs...")
    bench = {}
    for ood in ["uniform", "step", "chirp"]:
        bench[ood] = gen_ood(sim, 20, T, tau_max, 12345, ood)
    print(f"Data: {X.shape[0]:,} train, {Xv.shape[0]:,} val\n")

    archs = [
        ("raw_mlp", RawMLP),
        ("diag_pd", DiagPD),
        ("full_pd", FullMatrixPD),
    ]
    all_results = {}

    for arch_name, cls in archs:
        print(f"\n{'='*50}")
        print(f"Architecture: {arch_name}")
        print(f"{'='*50}")
        model = cls(sd=sd, nu=nu, hd=1024, nh=6).to(device)
        npar = sum(p.numel() for p in model.parameters())
        print(f"  Params: {npar:,}")
        xm, xs = X.mean(0), X.std(0) + 1e-8
        model.set_norm(xm, xs)
        vl = train(model, X, u_t, Xv, vu_t, device, args.epochs)
        print(f"  Best val: {vl:.6f}")

        print("  Benchmark:")
        res = {}
        all_b = []
        for ood_name, (rs, ra) in bench.items():
            m = eval_model(sim, model, rs, ra, tau_max, device)
            res[ood_name] = m
            all_b.append(m)
            print(f"    {ood_name:10s}: {m:.4e}")
        agg = float(np.mean(all_b))
        res["AGGREGATE"] = agg
        res["val_loss"] = vl
        all_results[arch_name] = res

    # Summary
    print(f"\n{'='*60}")
    print(f"{args.n_links}-LINK CHAIN PD MATRIX COMPARISON")
    print(f"{'='*60}")
    for name, res in all_results.items():
        print(f"  {name:15s}: AGG={res['AGGREGATE']:.4e} "
              f"(step={res.get('step',0):.4e} chirp={res.get('chirp',0):.4e})")

    out_dir = Path(f"outputs/chain{args.n_links}_pd_matrix")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_dir}/results.json")


if __name__ == "__main__":
    main()
