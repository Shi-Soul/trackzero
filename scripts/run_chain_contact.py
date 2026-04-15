#!/usr/bin/env python3
"""Stage 3B+: 5-link chain with ground contact.

Combines the two hardest challenges: high DOF (5-link) and contact dynamics.
Tests whether contact-aware input rescues performance at higher DOF.

Conditions:
1. 5-link no contact (baseline)
2. 5-link with contact, standard input
3. 5-link with contact, contact-aware input
"""
import argparse, json, os, time
import mujoco
import numpy as np
import torch
import torch.nn as nn

from trackzero.config import Config


def build_chain_contact_xml(n_links, with_contact=False, floor_z=None):
    """N-link chain with optional ground plane."""
    cfg = Config()
    p, s = cfg.pendulum, cfg.simulation
    L, m = p.link_length, p.link_mass
    ix, iy, iz = p.link_inertia
    damp, tau, g, dt = p.joint_damping, p.tau_max, p.gravity, s.dt

    # Floor at 60% of max reach
    if floor_z is None:
        floor_z = -(n_links * L * 0.6)

    contact_attr = ' condim="3" friction="0.5 0.5 0.005"' if with_contact else ""
    floor_xml = ""
    if with_contact:
        floor_xml = f'    <geom name="floor" type="plane" pos="0 0 {floor_z}" size="5 5 0.01" rgba="0.9 0.9 0.9 1" condim="3" friction="0.5 0.5 0.005"/>\n'

    # Build nested bodies
    body_lines, close_lines = [], []
    for i in range(1, n_links + 1):
        depth = i + 1
        pre = "    " * depth
        pos = "0 0 0" if i == 1 else f"0 0 -{L}"
        body_lines.append(f'{pre}<body name="link{i}" pos="{pos}">')
        body_lines.append(f'{pre}  <joint name="joint{i}" type="hinge"/>')
        body_lines.append(f'{pre}  <inertial pos="0 0 -{L/2}" mass="{m}" diaginertia="{ix} {iy} {iz}"/>')
        body_lines.append(f'{pre}  <geom name="geom{i}" fromto="0 0 0 0 0 -{L}"/>')
        close_lines.append(f'{pre}</body>')

    bodies = "\n".join(body_lines) + "\n" + "\n".join(reversed(close_lines))
    actuators = "\n".join(
        f'    <motor name="motor{i}" joint="joint{i}" ctrllimited="true" ctrlrange="{-tau} {tau}"/>'
        for i in range(1, n_links + 1)
    )

    return f"""\
<mujoco model="chain_{n_links}link{'_contact' if with_contact else ''}">
  <option gravity="0 0 -{g}" timestep="{dt}" integrator="RK4"/>
  <default>
    <joint axis="0 1 0" damping="{damp}" limited="false"/>
    <geom type="capsule" size="0.02" rgba="0.4 0.6 0.8 1"{contact_attr}/>
  </default>
  <worldbody>
{floor_xml}{bodies}
  </worldbody>
  <actuator>
{actuators}
  </actuator>
</mujoco>
"""


def generate_data(xml, n_traj, traj_len, rng, tau_max):
    """Random rollout data with contact flags."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    nq, nv, nu = model.nq, model.nv, model.nu

    all_s, all_sn, all_a, all_c = [], [], [], []
    for _ in range(n_traj):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = rng.uniform(-np.pi, np.pi, size=nq)
        data.qvel[:] = rng.uniform(-5, 5, size=nv)
        mujoco.mj_forward(model, data)
        for _ in range(traj_len):
            s = np.concatenate([data.qpos.copy(), data.qvel.copy()])
            tau = rng.uniform(-tau_max, tau_max, size=nu)
            c = float(data.ncon > 0)
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            sn = np.concatenate([data.qpos.copy(), data.qvel.copy()])
            all_s.append(s); all_sn.append(sn); all_a.append(tau); all_c.append(c)

    return np.array(all_s), np.array(all_sn), np.array(all_a), np.array(all_c)


def generate_refs(xml, n_per, traj_len, rng, tau_max):
    """3-family benchmark references."""
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
                    sw = rng.integers(T//4, 3*T//4)
                    actions[:sw, j] = v
                    actions[sw:, j] = rng.uniform(-tau_max, tau_max)
            elif fam == "chirp":
                t_arr = np.arange(T) * dt
                for j in range(nu):
                    f0, f1 = rng.uniform(0.1, 1), rng.uniform(2, 10)
                    amp = rng.uniform(0.5, 1.0) * tau_max
                    ph = t_arr * (f0 + (f1-f0)*t_arr/(2*t_arr[-1]))
                    actions[:, j] = amp * np.sin(2*np.pi*ph)

            states = np.zeros((T+1, nq+nv))
            states[0] = np.concatenate([data.qpos.copy(), data.qvel.copy()])
            for t in range(T):
                data.ctrl[:] = np.clip(actions[t], -tau_max, tau_max)
                mujoco.mj_step(model, data)
                states[t+1] = np.concatenate([data.qpos.copy(), data.qvel.copy()])
            trajs.append((states, actions))
        families[fam] = trajs
    return families


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=1024, layers=6):
        super().__init__()
        dims = [in_dim] + [hidden]*layers + [out_dim]
        mods = []
        for i in range(len(dims)-1):
            mods.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                mods.append(nn.ReLU())
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


def train(X, Y, device, epochs=100, bs=4096, lr=3e-4):
    n = len(X)
    n_val = max(int(n*0.1), 1000)
    Xt = torch.tensor(X.astype(np.float32))
    Yt = torch.tensor(Y.astype(np.float32))
    Xv, Yv = Xt[:n_val].to(device), Yt[:n_val].to(device)
    Xtr, Ytr = Xt[n_val:].to(device), Yt[n_val:].to(device)

    model = MLP(X.shape[1], Y.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val, best_sd = float('inf'), None
    for ep in range(1, epochs+1):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        el, nb = 0., 0
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i+bs]
            loss = nn.functional.mse_loss(model(Xtr[idx]), Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); nb += 1
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
    return model, best_val


def benchmark(model, xml, families, device, tau_max, use_contact=False):
    mj = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(mj)
    nq = mj.nq
    results = {}
    for fam, trajs in families.items():
        mses = []
        for ref_s, ref_a in trajs:
            T = len(ref_a)
            mujoco.mj_resetData(mj, d)
            d.qpos[:] = ref_s[0, :nq]; d.qvel[:] = ref_s[0, nq:]
            mujoco.mj_forward(mj, d)
            errs = []
            for t in range(T):
                s = np.concatenate([d.qpos.copy(), d.qvel.copy()])
                parts = [s, ref_s[t+1]]
                if use_contact:
                    parts.append(np.array([float(d.ncon > 0)]))
                inp = np.concatenate(parts).astype(np.float32)
                with torch.no_grad():
                    tau = model(torch.tensor(inp).unsqueeze(0).to(device)).cpu().numpy()[0]
                d.ctrl[:] = np.clip(tau, -tau_max, tau_max)
                mujoco.mj_step(mj, d)
                actual = np.concatenate([d.qpos.copy(), d.qvel.copy()])
                errs.append(np.mean((actual - ref_s[t+1])**2))
            mses.append(np.mean(errs))
        results[fam] = float(np.mean(mses))
    results["AGGREGATE"] = float(np.mean([results[k] for k in ["uniform","step","chirp"]]))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-links", type=int, default=5)
    parser.add_argument("--n-traj", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)
    tau_max = 5.0
    n = args.n_links
    out_dir = f"outputs/chain{n}_contact"
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    configs = [
        ("no_contact",       False, False),
        ("contact_baseline", True,  False),
        ("contact_aware",    True,  True),
    ]

    for label, with_contact, use_flag in configs:
        print(f"\n{'='*60}")
        print(f"{label}: {n}-link, contact={with_contact}, flag={use_flag}")
        print(f"{'='*60}")

        xml = build_chain_contact_xml(n, with_contact)
        rng = np.random.default_rng(42)
        print("Generating data...")
        t0 = time.time()
        S, SN, A, C = generate_data(xml, args.n_traj, 500, rng, tau_max)
        print(f"  {len(S)} pairs in {time.time()-t0:.1f}s")
        if with_contact:
            print(f"  Contact fraction: {C.mean():.1%}")

        X = np.concatenate([S, SN], axis=1)
        if use_flag:
            X = np.concatenate([X, C[:, None]], axis=1)

        print(f"Training (in={X.shape[1]}, out={A.shape[1]})...")
        model, vl = train(X, A, device, epochs=args.epochs)

        print("Generating benchmark...")
        bench_rng = np.random.default_rng(123)
        families = generate_refs(xml, 20, 500, bench_rng, tau_max)

        print("Benchmarking...")
        bench = benchmark(model, xml, families, device, tau_max, use_flag)
        for k, v in bench.items():
            print(f"    {k:12s}: {v:.4e}")
        results[label] = {**bench, "val_loss": vl}

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {n}-link chain contact experiment")
    print(f"{'='*60}")
    print(f"{'Config':<25s} {'AGG':>10s} {'uniform':>10s} {'step':>10s} {'chirp':>10s}")
    for label in ["no_contact", "contact_baseline", "contact_aware"]:
        r = results[label]
        print(f"{label:<25s} {r['AGGREGATE']:10.4e} {r['uniform']:10.4e} {r['step']:10.4e} {r['chirp']:10.4e}")

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir}/results.json")


if __name__ == "__main__":
    main()
