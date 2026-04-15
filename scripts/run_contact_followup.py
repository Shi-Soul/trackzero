#!/usr/bin/env python3
"""Contact dynamics follow-up: does contact-aware input or more data help?

Tests two hypotheses:
1. More data (10K vs 2K) helps the baseline MLP
2. Adding contact flag to input helps the MLP

Uses the same 2-link chain + ground plane setup as run_contact_experiment.py.
"""
import argparse, json, os, time
import mujoco
import numpy as np
import torch
import torch.nn as nn

from scripts.run_contact_experiment import (
    build_chain_contact_xml,
    generate_benchmark_refs,
    MLP,
)


def generate_data_with_contact_info(xml, n_traj, traj_len, rng, tau_max=5.0):
    """Generate data with contact flag."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    nq, nv, nu = model.nq, model.nv, model.nu

    states, next_states, actions, contact_flags = [], [], [], []

    for _ in range(n_traj):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = rng.uniform(-np.pi, np.pi, size=nq)
        data.qvel[:] = rng.uniform(-5, 5, size=nv)
        mujoco.mj_forward(model, data)

        for t in range(traj_len):
            s = np.concatenate([data.qpos.copy(), data.qvel.copy()])
            tau = rng.uniform(-tau_max, tau_max, size=nu)
            data.ctrl[:] = tau
            has_contact = float(data.ncon > 0)
            mujoco.mj_step(model, data)
            s_next = np.concatenate([data.qpos.copy(), data.qvel.copy()])

            states.append(s)
            next_states.append(s_next)
            actions.append(tau)
            contact_flags.append(has_contact)

    return (np.array(states), np.array(next_states),
            np.array(actions), np.array(contact_flags))


def train_model(X_np, Y_np, device, epochs=100, bs=4096, lr=3e-4):
    """Train and return best model + val loss."""
    n = len(X_np)
    n_val = max(int(n * 0.1), 1000)

    X = torch.tensor(X_np.astype(np.float32))
    Y = torch.tensor(Y_np.astype(np.float32))
    X_val, Y_val = X[:n_val].to(device), Y[:n_val].to(device)
    X_tr, Y_tr = X[n_val:].to(device), Y[n_val:].to(device)

    in_dim, out_dim = X.shape[1], Y.shape[1]
    model = MLP(in_dim, out_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val, best_sd = float("inf"), None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr), device=device)
        eloss, nb = 0.0, 0
        for i in range(0, len(X_tr), bs):
            idx = perm[i:i + bs]
            loss = nn.functional.mse_loss(model(X_tr[idx]), Y_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            eloss += loss.item(); nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(model(X_val), Y_val).item()
        if vl < best_val:
            best_val = vl
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 20 == 0 or ep == 1:
            print(f"    Ep {ep:4d}: t={eloss/nb:.5f} v={vl:.5f} best={best_val:.5f}")

    model.load_state_dict(best_sd)
    model.eval()
    return model, best_val


def benchmark_policy(model, xml, families, device, tau_max, use_contact_flag=False):
    """Closed-loop benchmark with optional contact flag input."""
    mj_model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(mj_model)
    nq = mj_model.nq

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
                parts = [s, s_ref]
                if use_contact_flag:
                    parts.append(np.array([float(mj_data.ncon > 0)]))
                inp = np.concatenate(parts).astype(np.float32)
                with torch.no_grad():
                    tau = model(torch.tensor(inp).unsqueeze(0).to(device)).cpu().numpy()[0]
                tau = np.clip(tau, -tau_max, tau_max)
                mj_data.ctrl[:] = tau
                mujoco.mj_step(mj_model, mj_data)
                actual = np.concatenate([mj_data.qpos.copy(), mj_data.qvel.copy()])
                errors.append(np.mean((actual - s_ref) ** 2))
            mse_list.append(np.mean(errors))
        results[fam_name] = float(np.mean(mse_list))
    results["AGGREGATE"] = float(np.mean([results[k] for k in ["uniform", "step", "chirp"]]))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)
    tau_max = 5.0
    xml = build_chain_contact_xml(with_contact=True)
    out_dir = "outputs/contact_followup"
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    configs = [
        ("baseline_2K", 2000, False),
        ("baseline_10K", 10000, False),
        ("contact_aware_2K", 2000, True),
        ("contact_aware_10K", 10000, True),
    ]

    bench_rng = np.random.default_rng(123)
    print("Generating benchmark references...")
    families = generate_benchmark_refs(xml, 20, 500, bench_rng, tau_max)

    for label, n_traj, use_flag in configs:
        print(f"\n{'='*60}")
        print(f"Config: {label} (n_traj={n_traj}, contact_flag={use_flag})")
        print(f"{'='*60}")

        rng = np.random.default_rng(42)
        print("Generating data...")
        t0 = time.time()
        states, next_states, actions, cflags = generate_data_with_contact_info(
            xml, n_traj, 500, rng, tau_max
        )
        print(f"  {len(states)} pairs in {time.time()-t0:.1f}s")
        print(f"  Contact fraction: {cflags.mean():.1%}")

        # Build input
        if use_flag:
            X = np.concatenate([states, next_states, cflags[:, None]], axis=1)
        else:
            X = np.concatenate([states, next_states], axis=1)
        Y = actions

        print(f"Training (input_dim={X.shape[1]})...")
        model, val_loss = train_model(X, Y, device, epochs=args.epochs)
        print(f"  Best val loss: {val_loss:.6f}")

        print("Benchmarking...")
        bench = benchmark_policy(model, xml, families, device, tau_max, use_flag)
        for k, v in bench.items():
            print(f"    {k:12s}: {v:.4e}")

        results[label] = {**bench, "val_loss": val_loss}

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<25s} {'AGG':>10s} {'uniform':>10s} {'step':>10s} {'chirp':>10s}")
    for label in ["baseline_2K", "baseline_10K", "contact_aware_2K", "contact_aware_10K"]:
        r = results[label]
        print(f"{label:<25s} {r['AGGREGATE']:10.4e} {r['uniform']:10.4e} {r['step']:10.4e} {r['chirp']:10.4e}")

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir}/results.json")


if __name__ == "__main__":
    main()
