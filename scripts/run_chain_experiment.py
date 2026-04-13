#!/usr/bin/env python3
"""Stage 3A: N-link chain experiment.

Tests whether TRACK-ZERO recipe (random rollout + cosine+WD) scales
to higher-DOF systems. Uses the parametric chain builder.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch, mujoco
from trackzero.config import load_config, Config
from trackzero.sim.pendulum_model import build_chain_xml
from trackzero.policy.mlp import InverseDynamicsMLP


class ChainSim:
    """Lightweight N-link chain simulator."""
    def __init__(self, cfg: Config, n_links: int):
        xml = build_chain_xml(n_links, cfg.pendulum, cfg.simulation)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nu
        self._sub = cfg.simulation.substeps
        self._tau = cfg.pendulum.tau_max

    def reset(self, q0, v0):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = q0
        self.data.qvel[:] = v0
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def state(self):
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, -self._tau, self._tau)
        for _ in range(self._sub):
            mujoco.mj_step(self.model, self.data)
        return self.state()

    def rollout(self, actions, q0, v0):
        self.reset(q0, v0)
        T = len(actions)
        states = np.zeros((T+1, self.nq+self.nv))
        states[0] = self.state()
        for t in range(T):
            states[t+1] = self.step(actions[t])
        return states


def gen_data(sim, n_traj, T, tau_max, seed):
    """Generate random rollout data for N-link chain."""
    rng = np.random.default_rng(seed)
    nq, nu = sim.nq, sim.nu
    sd = nq * 2
    all_s = np.zeros((n_traj, T+1, sd))
    all_a = np.zeros((n_traj, T, nu))
    subtypes = ["uniform", "ou", "gaussian"]
    for i in range(n_traj):
        q0 = rng.uniform(-np.pi, np.pi, nq)
        v0 = rng.uniform(-3, 3, nq)
        st = subtypes[i % len(subtypes)]
        if st == "uniform":
            a = rng.uniform(-tau_max, tau_max, (T, nu))
        elif st == "ou":
            a = np.zeros((T, nu))
            x = rng.uniform(-tau_max*0.5, tau_max*0.5, nu)
            for t in range(T):
                x = x - 0.15*x*0.01 + 0.5*tau_max*np.sqrt(0.01)*rng.standard_normal(nu)
                a[t] = np.clip(x, -tau_max, tau_max)
        else:
            a = np.clip(rng.normal(0, tau_max/3, (T, nu)), -tau_max, tau_max)
        states = sim.rollout(a, q0, v0)
        all_s[i] = states
        all_a[i] = a
    return all_s, all_a


def gen_ood(sim, n_traj, T, tau_max, seed, ood_type="step"):
    """Generate OOD references for chain."""
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
                sw = np.sort(rng.integers(0, T, n_steps-1)).tolist()
                sw = [0] + sw + [T]
                lvls = rng.uniform(-tau_max, tau_max, n_steps)
                for k in range(n_steps):
                    a[sw[k]:sw[k+1], j] = lvls[k]
        elif ood_type == "chirp":
            t_arr = np.arange(T) * 0.01
            a = np.zeros((T, nu))
            for j in range(nu):
                f0, f1 = rng.uniform(0.1, 0.5), rng.uniform(1.0, 5.0)
                amp = rng.uniform(0.3*tau_max, tau_max)
                ph = 2*np.pi*(f0*t_arr + 0.5*(f1-f0)*t_arr**2/max(T*0.01, 1e-9))
                a[:, j] = np.clip(amp*np.sin(ph), -tau_max, tau_max)
        else:
            a = rng.uniform(-tau_max, tau_max, (T, nu))
        states = sim.rollout(a, q0, v0)
        all_s[i] = states
        all_a[i] = a
    return all_s, all_a


def eval_chain(sim, model, ref_s, ref_a, tau_max, device):
    """Evaluate tracking policy on chain references."""
    N = len(ref_s)
    nq = sim.nq
    mse_list = []
    for i in range(N):
        T = len(ref_a[i])
        sim.reset(ref_s[i, 0, :nq], ref_s[i, 0, nq:])
        actual = [sim.state()]
        for t in range(T):
            cs = actual[-1]
            rns = ref_s[i, t+1]
            x = np.concatenate([cs, rns]).astype(np.float32)
            with torch.no_grad():
                xt = torch.from_numpy(x).unsqueeze(0).to(device)
                a = model(xt).squeeze(0).cpu().numpy()
            a = np.clip(a, -tau_max, tau_max)
            actual.append(sim.step(a))
        actual = np.array(actual)
        # MSE on positions + 0.1 * velocities
        q_err = np.mean((ref_s[i, 1:, :nq] - actual[1:, :nq])**2)
        v_err = np.mean((ref_s[i, 1:, nq:] - actual[1:, nq:])**2)
        mse_list.append(q_err + 0.1 * v_err)
    return float(np.mean(mse_list))


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--n-links", type=int, required=True)
    pa.add_argument("--n-traj", type=int, default=10000)
    pa.add_argument("--epochs", type=int, default=200)
    pa.add_argument("--device", default="cuda:0")
    pa.add_argument("--seed", type=int, default=42)
    args = pa.parse_args()

    cfg = load_config()
    tau_max = cfg.pendulum.tau_max
    device = torch.device(args.device)
    sim = ChainSim(cfg, args.n_links)
    nq, nu = sim.nq, sim.nu
    sd = nq * 2
    T = int(cfg.dataset.trajectory_duration / cfg.simulation.control_dt)
    out = Path(f"outputs/chain_{args.n_links}link_{args.n_traj//1000}k")
    out.mkdir(parents=True, exist_ok=True)

    print(f"{'='*50}")
    print(f"STAGE 3: {args.n_links}-link chain (nq={nq}, nu={nu})")
    print(f"{'='*50}")

    # Generate training data
    print(f"\nGenerating {args.n_traj//1000}K trajectories...")
    t0 = time.time()
    train_s, train_a = gen_data(sim, args.n_traj, T, tau_max, args.seed)
    print(f"  Done in {time.time()-t0:.0f}s, state_dim={sd}")

    # Prepare pairs
    s_t = train_s[:, :-1].reshape(-1, sd)
    s_tp1 = train_s[:, 1:].reshape(-1, sd)
    u_t = train_a.reshape(-1, nu)
    X = np.concatenate([s_t, s_tp1], axis=-1).astype(np.float32)
    Y = u_t.astype(np.float32)

    # Scale hidden dim with DOF
    hdim = min(1024, 256 * nq)
    nlayer = 6
    model = InverseDynamicsMLP(state_dim=sd, action_dim=nu,
                                hidden_dim=hdim, n_hidden=nlayer).to(device)
    npar = sum(p.numel() for p in model.parameters())
    print(f"  Model: {hdim}x{nlayer}, {npar:,} params, input={sd*2}")

    Xt = torch.from_numpy(X).to(device)
    Yt = torch.from_numpy(Y).to(device)

    # Val data = last 500 trajectories from a different seed
    val_s, val_a = gen_data(sim, 500, T, tau_max, args.seed + 100)
    vs_t = val_s[:, :-1].reshape(-1, sd)
    vs_tp1 = val_s[:, 1:].reshape(-1, sd)
    vu_t = val_a.reshape(-1, nu)
    Xv = torch.from_numpy(np.concatenate([vs_t, vs_tp1], axis=-1).astype(np.float32)).to(device)
    Yv = torch.from_numpy(vu_t.astype(np.float32)).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    bs = 4096; best_vl = float("inf")

    print(f"\nTraining {args.epochs} epochs...")
    t0 = time.time()
    for ep in range(1, args.epochs+1):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        el, nb = 0.0, max(1, len(Xt)//bs)
        for b in range(nb):
            idx = perm[b*bs:(b+1)*bs]
            loss = torch.nn.functional.mse_loss(model(Xt[idx]), Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item()
        model.eval()
        with torch.no_grad():
            vt, vn = 0.0, 0
            for vb in range(0, len(Xv), bs):
                xv, yv = Xv[vb:vb+bs], Yv[vb:vb+bs]
                vt += torch.nn.functional.mse_loss(model(xv), yv).item()*len(xv)
                vn += len(xv)
            vl = vt/vn
        if vl < best_vl:
            best_vl = vl
            torch.save({"model_state_dict": model.state_dict(),
                        "n_links": args.n_links, "hdim": hdim, "nlayer": nlayer},
                       out/"best_model.pt")
        sched.step()
        if ep % 50 == 0 or ep == 1:
            print(f"  Ep {ep:3d}: train={el/nb:.6f} val={vl:.6f} best={best_vl:.6f} [{time.time()-t0:.0f}s]")

    train_time = time.time() - t0
    ckpt = torch.load(out/"best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()

    # Benchmark on held-out trajectories
    print(f"\nBenchmark...")
    bench = {}
    for fam in ["uniform", "step", "chirp"]:
        if fam == "uniform":
            rs, ra = gen_data(sim, 100, T, tau_max, 9999)
        else:
            rs, ra = gen_ood(sim, 100, T, tau_max, 9999, ood_type=fam)
        mse = eval_chain(sim, model, rs, ra, tau_max, device)
        bench[fam] = mse
        print(f"  {fam:<12s}: {mse:.4e}")
    bench["_aggregate"] = float(np.mean(list(bench.values())))
    print(f"  {'_aggregate':<12s}: {bench['_aggregate']:.4e}")

    results = {"n_links": args.n_links, "nq": nq, "nu": nu,
               "hdim": hdim, "nlayer": nlayer, "n_params": npar,
               "n_traj": args.n_traj, "best_val": float(best_vl),
               "train_time": train_time, "benchmark": bench}
    with open(out/"results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone! Saved to {out}")

if __name__ == "__main__":
    main()
