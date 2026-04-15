#!/usr/bin/env python3
"""Test structured dynamics architecture across DOF scales.

Hypothesis: Factored MLP (τ = A(q,v)@error + b(q,v)) scales better with DOF
than flat MLP or diagonal residual PD, because it captures cross-joint coupling
through the full gain matrix A(q,v).

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.run_structured --n-links 2
    CUDA_VISIBLE_DEVICES=2 python -m scripts.run_structured --n-links 5
"""
import argparse, json, os, time
import mujoco
import numpy as np
import torch
import torch.nn as nn

from trackzero.config import Config
from scripts.models_structured import ARCH_REGISTRY


def build_chain_xml(n_links):
    """N-link planar chain XML (no contact)."""
    cfg = Config()
    p, s = cfg.pendulum, cfg.simulation
    L, m = p.link_length, p.link_mass
    ix, iy, iz = p.link_inertia
    damp, tau, g, dt = p.joint_damping, p.tau_max, p.gravity, s.dt

    body_lines, close_lines = [], []
    for i in range(1, n_links + 1):
        pre = "    " * (i + 1)
        pos = "0 0 0" if i == 1 else f"0 0 -{L}"
        body_lines.append(f'{pre}<body name="link{i}" pos="{pos}">')
        body_lines.append(f'{pre}  <joint name="joint{i}" type="hinge"/>')
        body_lines.append(
            f'{pre}  <inertial pos="0 0 -{L/2}" mass="{m}"'
            f' diaginertia="{ix} {iy} {iz}"/>'
        )
        body_lines.append(f'{pre}  <geom name="geom{i}" fromto="0 0 0 0 0 -{L}"/>')
        close_lines.append(f'{pre}</body>')

    bodies = "\n".join(body_lines) + "\n" + "\n".join(reversed(close_lines))
    actuators = "\n".join(
        f'    <motor name="motor{i}" joint="joint{i}" ctrllimited="true"'
        f' ctrlrange="{-tau} {tau}"/>'
        for i in range(1, n_links + 1)
    )
    return f"""\
<mujoco model="chain_{n_links}link">
  <option gravity="0 0 -{g}" timestep="{dt}" integrator="RK4"/>
  <default>
    <joint axis="0 1 0" damping="{damp}" limited="false"/>
    <geom type="capsule" size="0.02" rgba="0.4 0.6 0.8 1"/>
  </default>
  <worldbody>
{bodies}
  </worldbody>
  <actuator>
{actuators}
  </actuator>
</mujoco>
"""


def generate_data(xml, n_traj, traj_len, tau_max, seed=42):
    """Random rollout data for N-link chain."""
    rng = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    nq, nv, nu = model.nq, model.nv, model.nu
    sd = nq + nv

    all_s, all_sn, all_a = [], [], []
    for _ in range(n_traj):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = rng.uniform(-np.pi, np.pi, size=nq)
        data.qvel[:] = rng.uniform(-5, 5, size=nv)
        mujoco.mj_forward(model, data)
        for _ in range(traj_len):
            s = np.concatenate([data.qpos.copy(), data.qvel.copy()])
            tau = rng.uniform(-tau_max, tau_max, size=nu)
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            sn = np.concatenate([data.qpos.copy(), data.qvel.copy()])
            all_s.append(s)
            all_sn.append(sn)
            all_a.append(tau)

    S = np.array(all_s, dtype=np.float32)
    SN = np.array(all_sn, dtype=np.float32)
    A = np.array(all_a, dtype=np.float32)
    print(f"  {len(S):,} pairs in {sd}D state, {nu}D action")
    return S, SN, A


def generate_benchmark(xml, n_per, traj_len, tau_max, seed=9999):
    """3-family benchmark references."""
    rng = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    nq, nv, nu = model.nq, model.nv, model.nu
    dt = model.opt.timestep
    families = {}

    for fam in ["uniform", "step", "chirp"]:
        trajs = []
        for _ in range(n_per):
            mujoco.mj_resetData(model, data)
            data.qpos[:] = rng.uniform(-np.pi, np.pi, size=nq)
            data.qvel[:] = rng.uniform(-3, 3, size=nv)
            mujoco.mj_forward(model, data)

            T = traj_len
            actions = np.zeros((T, nu))
            if fam == "uniform":
                actions = rng.uniform(-tau_max, tau_max, size=(T, nu))
            elif fam == "step":
                for j in range(nu):
                    v = rng.uniform(-tau_max, tau_max)
                    sw = rng.integers(T // 4, 3 * T // 4)
                    actions[:sw, j] = v
                    actions[sw:, j] = rng.uniform(-tau_max, tau_max)
            elif fam == "chirp":
                t_arr = np.arange(T) * dt
                for j in range(nu):
                    f0 = rng.uniform(0.1, 1)
                    f1 = rng.uniform(2, 10)
                    amp = rng.uniform(0.5, 1.0) * tau_max
                    ph = t_arr * (f0 + (f1 - f0) * t_arr / (2 * t_arr[-1]))
                    actions[:, j] = amp * np.sin(2 * np.pi * ph)

            states = np.zeros((T + 1, nq + nv))
            states[0] = np.concatenate([data.qpos.copy(), data.qvel.copy()])
            for t in range(T):
                data.ctrl[:] = np.clip(actions[t], -tau_max, tau_max)
                mujoco.mj_step(model, data)
                states[t + 1] = np.concatenate([data.qpos.copy(), data.qvel.copy()])
            trajs.append((states, actions))
        families[fam] = trajs
    return families


def train_model(model, X, Y, device, epochs=200, bs=4096, lr=3e-4):
    """Train with cosine LR + weight decay, best-val checkpoint."""
    n = len(X)
    n_val = max(int(n * 0.1), 1000)

    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y)
    Xv, Yv = Xt[:n_val].to(device), Yt[:n_val].to(device)
    Xtr, Ytr = Xt[n_val:].to(device), Yt[n_val:].to(device)

    # Set normalization from training data
    xm = X[n_val:].mean(0)
    xs = X[n_val:].std(0) + 1e-8
    model.set_norm(xm, xs)
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val, best_sd = float("inf"), None
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        el, nb = 0.0, 0
        for i in range(0, len(Xtr), bs):
            idx = perm[i : i + bs]
            loss = nn.functional.mse_loss(model(Xtr[idx]), Ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            el += loss.item()
            nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(model(Xv), Yv).item()
        if vl < best_val:
            best_val = vl
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 20 == 0 or ep == 1:
            print(f"    Ep {ep:4d}: t={el/nb:.5f} v={vl:.5f} best={best_val:.5f}")

    model.load_state_dict(best_sd)
    model.eval()
    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.0f}s, best_val={best_val:.5f}")
    return model, best_val


def benchmark_model(model, xml, families, device, tau_max):
    """Closed-loop tracking benchmark."""
    mj = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(mj)
    nq = mj.nq
    results = {}

    for fam, trajs in families.items():
        mses = []
        for ref_s, ref_a in trajs:
            T = len(ref_a)
            mujoco.mj_resetData(mj, d)
            d.qpos[:] = ref_s[0, :nq]
            d.qvel[:] = ref_s[0, nq:]
            mujoco.mj_forward(mj, d)
            errs = []
            for t in range(T):
                s = np.concatenate([d.qpos.copy(), d.qvel.copy()])
                inp = np.concatenate([s, ref_s[t + 1]]).astype(np.float32)
                with torch.no_grad():
                    tau = model(
                        torch.tensor(inp).unsqueeze(0).to(device)
                    ).cpu().numpy()[0]
                d.ctrl[:] = np.clip(tau, -tau_max, tau_max)
                mujoco.mj_step(mj, d)
                actual = np.concatenate([d.qpos.copy(), d.qvel.copy()])
                errs.append(np.mean((actual - ref_s[t + 1]) ** 2))
            mses.append(np.mean(errs))
        results[fam] = float(np.mean(mses))

    results["AGGREGATE"] = float(
        np.mean([results[k] for k in ["uniform", "step", "chirp"]])
    )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-links", type=int, required=True)
    parser.add_argument("--n-traj", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    tau_max = cfg.pendulum.tau_max
    n = args.n_links
    nq = n
    sd = 2 * n  # state_dim = nq + nv

    out_dir = f"outputs/structured_{n}link"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"STRUCTURED DYNAMICS EXPERIMENT: {n}-link chain")
    print(f"Architectures: {list(ARCH_REGISTRY.keys())}")
    print(f"Data: {args.n_traj} traj, Training: {args.epochs} epochs")
    print(f"Network: {args.hidden}×{args.layers}, Device: {device}")
    print("=" * 60)

    # Generate data
    xml = build_chain_xml(n)
    print(f"\nGenerating {args.n_traj} training trajectories...")
    t0 = time.time()
    S, SN, A = generate_data(xml, args.n_traj, 500, tau_max)
    X = np.concatenate([S, SN], axis=1)  # (N, 2*sd)
    Y = A  # (N, nq)
    print(f"  Data generated in {time.time()-t0:.1f}s")

    print(f"\nGenerating benchmark references...")
    families = generate_benchmark(xml, 50, 500, tau_max)

    # Run each architecture
    all_results = {}
    for arch_name, ArchCls in ARCH_REGISTRY.items():
        print(f"\n{'='*60}")
        print(f"Architecture: {arch_name}")
        print(f"{'='*60}")

        model = ArchCls(sd, nq, hidden=args.hidden, layers=args.layers)
        npar = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {npar:,}")

        print(f"Training (in={X.shape[1]}, out={Y.shape[1]})...")
        model, val_loss = train_model(
            model, X, Y, device, epochs=args.epochs, bs=4096
        )

        print(f"  Benchmarking...")
        bench = benchmark_model(model, xml, families, device, tau_max)
        bench["val_loss"] = val_loss
        bench["params"] = npar
        all_results[arch_name] = bench

        print(f"  AGG={bench['AGGREGATE']:.4e}  "
              f"uni={bench['uniform']:.4e}  "
              f"step={bench['step']:.4e}  "
              f"chirp={bench['chirp']:.4e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {n}-link structured dynamics experiment")
    print(f"{'='*60}")
    header = f"{'Config':<20s} {'Params':>8s} {'AGG':>10s} {'uniform':>10s} {'step':>10s} {'chirp':>10s}"
    print(header)
    for name, r in all_results.items():
        print(f"{name:<20s} {r['params']:>8,} {r['AGGREGATE']:>10.4e} "
              f"{r['uniform']:>10.4e} {r['step']:>10.4e} {r['chirp']:>10.4e}")

    # Save
    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
