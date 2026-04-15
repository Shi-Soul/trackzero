#!/usr/bin/env python3
"""Stage 3B: Contact dynamics experiment.

Compares TRACK-ZERO performance on the same 2-link chain with and without
ground contact. Tests whether random-data supervised learning transfers
through contact discontinuities.

Key question: Does contact break the inverse dynamics learning approach?
"""
import argparse
import json
import os
import time

import mujoco
import numpy as np
import torch
import torch.nn as nn

from trackzero.config import Config


def build_chain_contact_xml(with_contact: bool, floor_z: float = -0.75):
    """2-link chain optionally with a ground plane at floor_z."""
    cfg = Config()
    p = cfg.pendulum
    s = cfg.simulation
    L, m = p.link_length, p.link_mass
    ix, iy, iz = p.link_inertia
    damp, tau, g, dt = p.joint_damping, p.tau_max, p.gravity, s.dt

    floor_xml = ""
    contact_attr = ""
    if with_contact:
        floor_xml = f'    <geom name="floor" type="plane" pos="0 0 {floor_z}" size="2 2 0.01" rgba="0.9 0.9 0.9 1" condim="3" friction="0.5 0.5 0.005"/>\n'
        contact_attr = ' condim="3" friction="0.5 0.5 0.005"'

    return f"""\
<mujoco model="chain_2link{'_contact' if with_contact else ''}">
  <option gravity="0 0 -{g}" timestep="{dt}" integrator="RK4"/>
  <default>
    <joint axis="0 1 0" damping="{damp}" limited="false"/>
    <geom type="capsule" size="0.02" rgba="0.4 0.6 0.8 1"{contact_attr}/>
  </default>
  <worldbody>
{floor_xml}    <body name="link1" pos="0 0 0">
      <joint name="joint1" type="hinge"/>
      <inertial pos="0 0 -{L/2}" mass="{m}" diaginertia="{ix} {iy} {iz}"/>
      <geom name="geom1" fromto="0 0 0 0 0 -{L}"/>
      <body name="link2" pos="0 0 -{L}">
        <joint name="joint2" type="hinge"/>
        <inertial pos="0 0 -{L/2}" mass="{m}" diaginertia="{ix} {iy} {iz}"/>
        <geom name="geom2" fromto="0 0 0 0 0 -{L}"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor1" joint="joint1" ctrllimited="true" ctrlrange="{-tau} {tau}"/>
    <motor name="motor2" joint="joint2" ctrllimited="true" ctrlrange="{-tau} {tau}"/>
  </actuator>
</mujoco>
"""


def generate_data(xml, n_traj, traj_len, rng, tau_max=5.0):
    """Generate random rollout data from the given model."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    nq, nv, nu = model.nq, model.nv, model.nu
    sd = nq + nv

    states = []
    next_states = []
    actions = []
    contact_flags = []

    for _ in range(n_traj):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = rng.uniform(-np.pi, np.pi, size=nq)
        data.qvel[:] = rng.uniform(-5, 5, size=nv)
        mujoco.mj_forward(model, data)

        for t in range(traj_len):
            s = np.concatenate([data.qpos.copy(), data.qvel.copy()])
            tau = rng.uniform(-tau_max, tau_max, size=nu)
            data.ctrl[:] = tau
            had_contact = data.ncon > 0

            for _ in range(model.opt.timestep > 0 and 1 or 1):
                mujoco.mj_step(model, data)

            s_next = np.concatenate([data.qpos.copy(), data.qvel.copy()])
            states.append(s)
            next_states.append(s_next)
            actions.append(tau)
            contact_flags.append(had_contact)

    return (np.array(states), np.array(next_states),
            np.array(actions), np.array(contact_flags))


def oracle_benchmark(xml, ref_trajectories, tau_max=5.0):
    """Run oracle (mj_inverse) on benchmark references."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    nq = model.nq

    total_mse = 0.0
    n_traj = 0
    for ref_states, ref_actions in ref_trajectories:
        T = len(ref_actions)
        errors = []
        for t in range(T - 1):
            # Finite-difference acceleration
            dt = model.opt.timestep
            q_t = ref_states[t, :nq]
            v_t = ref_states[t, nq:]
            v_next = ref_states[t + 1, nq:]
            qacc = (v_next - v_t) / dt

            data.qpos[:] = q_t
            data.qvel[:] = v_t
            data.qacc[:] = qacc
            mujoco.mj_inverse(model, data)
            oracle_tau = np.clip(data.qfrc_inverse.copy(), -tau_max, tau_max)

            # Roll forward with oracle torque
            data.qpos[:] = ref_states[t, :nq]
            data.qvel[:] = ref_states[t, nq:]
            data.ctrl[:] = oracle_tau
            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)

            actual = np.concatenate([data.qpos.copy(), data.qvel.copy()])
            errors.append(np.mean((actual - ref_states[t + 1]) ** 2))

        total_mse += np.mean(errors)
        n_traj += 1

    return total_mse / max(n_traj, 1)


def generate_benchmark_refs(xml, n_per_family, traj_len, rng, tau_max=5.0):
    """Generate benchmark reference trajectories for 3 families."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    nq, nv, nu = model.nq, model.nv, model.nu
    dt = model.opt.timestep
    families = {}

    for family in ["uniform", "step", "chirp"]:
        trajs = []
        for _ in range(n_per_family):
            mujoco.mj_resetData(model, data)
            data.qpos[:] = rng.uniform(-np.pi, np.pi, size=nq)
            data.qvel[:] = rng.uniform(-3, 3, size=nv)
            mujoco.mj_forward(model, data)

            T = traj_len
            actions = np.zeros((T, nu))
            if family == "uniform":
                actions = rng.uniform(-tau_max, tau_max, size=(T, nu))
            elif family == "step":
                for j in range(nu):
                    val = rng.uniform(-tau_max, tau_max)
                    switch = rng.integers(T // 4, 3 * T // 4)
                    actions[:switch, j] = val
                    actions[switch:, j] = rng.uniform(-tau_max, tau_max)
            elif family == "chirp":
                t_arr = np.arange(T) * dt
                for j in range(nu):
                    f0, f1 = rng.uniform(0.1, 1), rng.uniform(2, 10)
                    amp = rng.uniform(0.5, 1.0) * tau_max
                    phase = t_arr * (f0 + (f1 - f0) * t_arr / (2 * t_arr[-1]))
                    actions[:, j] = amp * np.sin(2 * np.pi * phase)

            # Roll out
            states = np.zeros((T + 1, nq + nv))
            states[0] = np.concatenate([data.qpos.copy(), data.qvel.copy()])
            for t in range(T):
                data.ctrl[:] = np.clip(actions[t], -tau_max, tau_max)
                mujoco.mj_step(model, data)
                states[t + 1] = np.concatenate([data.qpos.copy(), data.qvel.copy()])

            trajs.append((states, actions))
        families[family] = trajs

    return families


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=1024, layers=6):
        super().__init__()
        dims = [in_dim] + [hidden] * layers + [out_dim]
        mods = []
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mods.append(nn.ReLU())
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


def train_model(states, next_states, actions, device, epochs=100, bs=4096, lr=3e-4):
    """Train MLP to predict actions from (state, next_state)."""
    nq = states.shape[1] // 2
    in_dim = states.shape[1] + next_states.shape[1]
    out_dim = actions.shape[1]

    X = np.concatenate([states, next_states], axis=1).astype(np.float32)
    Y = actions.astype(np.float32)

    # Train/val split
    n = len(X)
    n_val = max(int(n * 0.1), 1000)
    X_val, Y_val = torch.tensor(X[:n_val]).to(device), torch.tensor(Y[:n_val]).to(device)
    X_tr, Y_tr = torch.tensor(X[n_val:]).to(device), torch.tensor(Y[n_val:]).to(device)

    model = MLP(in_dim, out_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = float("inf")
    best_sd = None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr), device=device)
        epoch_loss = 0.0
        n_batch = 0
        for i in range(0, len(X_tr), bs):
            idx = perm[i : i + bs]
            pred = model(X_tr[idx])
            loss = nn.functional.mse_loss(pred, Y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batch += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.mse_loss(model(X_val), Y_val).item()
        if val_loss < best_val:
            best_val = val_loss
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 20 == 0 or ep == 1:
            print(f"    Ep {ep:4d}: t={epoch_loss/n_batch:.5f} v={val_loss:.5f} best={best_val:.5f}")

    model.load_state_dict(best_sd)
    model.eval()
    return model, best_val


def benchmark_policy(model, xml, families, device, tau_max=5.0):
    """Closed-loop tracking benchmark for a trained policy."""
    mj_model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(mj_model)
    nq, nv = mj_model.nq, mj_model.nv

    results = {}
    for fam_name, trajs in families.items():
        mse_list = []
        for ref_states, ref_actions in trajs:
            T = len(ref_actions)
            mujoco.mj_resetData(mj_model, mj_data)
            mj_data.qpos[:] = ref_states[0, :nq]
            mj_data.qvel[:] = ref_states[0, nq:]
            mujoco.mj_forward(mj_model, mj_data)

            errors = []
            for t in range(T):
                s = np.concatenate([mj_data.qpos.copy(), mj_data.qvel.copy()])
                s_ref = ref_states[t + 1]
                inp = np.concatenate([s, s_ref]).astype(np.float32)
                with torch.no_grad():
                    tau = model(torch.tensor(inp).unsqueeze(0).to(device)).cpu().numpy()[0]
                tau = np.clip(tau, -tau_max, tau_max)
                mj_data.ctrl[:] = tau
                mujoco.mj_step(mj_model, mj_data)
                actual = np.concatenate([mj_data.qpos.copy(), mj_data.qvel.copy()])
                errors.append(np.mean((actual - s_ref) ** 2))
            mse_list.append(np.mean(errors))
        results[fam_name] = float(np.mean(mse_list))

    results["AGGREGATE"] = float(np.mean(list(results.values())))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-traj", type=int, default=2000)
    parser.add_argument("--traj-len", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n-bench", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(42)
    tau_max = 5.0
    out_dir = "outputs/contact_experiment"
    os.makedirs(out_dir, exist_ok=True)

    results = {}

    for with_contact in [False, True]:
        label = "contact" if with_contact else "no_contact"
        print("=" * 60)
        print(f"Condition: {label}")
        print("=" * 60)

        xml = build_chain_contact_xml(with_contact)
        print("Generating training data...")
        t0 = time.time()
        states, next_states, actions, cflags = generate_data(
            xml, args.n_traj, args.traj_len, rng, tau_max
        )
        print(f"  {len(states)} pairs in {time.time()-t0:.1f}s")
        if with_contact:
            print(f"  Contact fraction: {cflags.mean():.1%}")

        print("Generating benchmark references...")
        bench_rng = np.random.default_rng(123)
        families = generate_benchmark_refs(
            xml, args.n_bench, args.traj_len, bench_rng, tau_max
        )

        print(f"Training MLP ({label})...")
        model, val_loss = train_model(
            states, next_states, actions, device, epochs=args.epochs
        )
        print(f"  Best val loss: {val_loss:.6f}")

        print(f"Benchmarking ({label})...")
        bench = benchmark_policy(model, xml, families, device, tau_max)
        for k, v in bench.items():
            print(f"    {k:12s}: {v:.4e}")

        results[label] = {**bench, "val_loss": val_loss}

        # Oracle comparison
        print(f"Oracle benchmark ({label})...")
        oracle_mse = oracle_benchmark(xml, families["uniform"][:5], tau_max)
        results[label]["oracle_uniform"] = float(oracle_mse)
        print(f"    Oracle uniform MSE: {oracle_mse:.4e}")

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON: Contact vs No-Contact")
    print("=" * 60)
    for metric in ["AGGREGATE", "uniform", "step", "chirp"]:
        nc = results["no_contact"].get(metric, 0)
        c = results["contact"].get(metric, 0)
        ratio = c / nc if nc > 0 else float("inf")
        print(f"  {metric:12s}: no_contact={nc:.4e}  contact={c:.4e}  ratio={ratio:.2f}x")

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir}/results.json")


if __name__ == "__main__":
    main()
