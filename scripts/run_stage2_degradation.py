#!/usr/bin/env python3
"""Stage 2A+2B: Degradation analysis under noisy references.

Evaluates how the best Stage 1 model and the oracle degrade when
reference trajectories are corrupted with increasing noise levels.
This establishes the "optimal degradation" baseline.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from trackzero.config import load_config
from trackzero.data.ood_references import generate_ood_reference_data
from trackzero.data.dataset import TrajectoryDataset
from trackzero.oracle import InverseDynamicsOracle
from trackzero.eval.harness import EvalHarness
from trackzero.policy.mlp import InverseDynamicsMLP, MLPPolicy, load_checkpoint

BENCH_SEED, N_PER = 12345, 50
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

def corrupt_references(states, actions, noise_std, rng):
    """Add Gaussian noise to reference states (not actions)."""
    noisy = states.copy()
    noisy[:, 1:, :2] += rng.normal(0, noise_std, noisy[:, 1:, :2].shape)
    noisy[:, 1:, 2:] += rng.normal(0, noise_std * 5, noisy[:, 1:, 2:].shape)
    return noisy, actions

def eval_policy_on_refs(harness, policy, ref_s, ref_a):
    sm = harness.evaluate_policy(policy, ref_s, ref_a, max_trajectories=N_PER)
    return float(sm.mean_mse_total)

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model-dir", default="outputs/arch_1024x6_wd0.0001_lr_cosine_10k")
    pa.add_argument("--device", default="cuda:0")
    args = pa.parse_args()

    cfg = load_config()
    tau_max = cfg.pendulum.tau_max
    device = torch.device(args.device)
    oracle = InverseDynamicsOracle(cfg)
    harness = EvalHarness(cfg)
    out = Path("outputs/stage2_degradation"); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    # Load best Stage 1 model
    model_path = Path(args.model_dir) / "best_model.pt"
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model = InverseDynamicsMLP(4, 2, 1024, 6).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    mlp_policy = MLPPolicy(model, tau_max=tau_max, device=str(device))
    oracle_policy = oracle.as_policy("finite_difference")

    # Generate clean benchmark refs (use 3 families for speed)
    families_to_test = ["multisine", "step", "random_walk"]
    clean_refs = {}
    ds = TrajectoryDataset("data/medium/test.h5")
    ms, ma = ds.get_all_states(), ds.get_all_actions()
    ds.close()
    clean_refs["multisine"] = (ms[:N_PER], ma[:N_PER])
    for nm in ["step", "random_walk"]:
        s, a = generate_ood_reference_data(cfg, N_PER, action_type=nm, seed=BENCH_SEED)
        clean_refs[nm] = (s, a)

    # Sweep noise levels
    results = {"noise_levels": NOISE_LEVELS, "families": {}}
    print(f"Sweeping {len(NOISE_LEVELS)} noise levels x {len(families_to_test)} families")
    for fname in families_to_test:
        rs, ra = clean_refs[fname]
        mlp_curve, oracle_curve = [], []
        for nl in NOISE_LEVELS:
            if nl == 0.0:
                ns, na = rs, ra
            else:
                ns, na = corrupt_references(rs, ra, nl, rng)
            m_mse = eval_policy_on_refs(harness, mlp_policy, ns, na)
            o_mse = eval_policy_on_refs(harness, oracle_policy, ns, na)
            mlp_curve.append(m_mse)
            oracle_curve.append(o_mse)
            print(f"  {fname} noise={nl:.2f}: MLP={m_mse:.4e} Oracle={o_mse:.4e}")
        results["families"][fname] = {
            "mlp": mlp_curve, "oracle": oracle_curve}

    # Aggregate
    agg_mlp = [np.mean([results["families"][f]["mlp"][i]
                        for f in families_to_test])
               for i in range(len(NOISE_LEVELS))]
    agg_orc = [np.mean([results["families"][f]["oracle"][i]
                        for f in families_to_test])
               for i in range(len(NOISE_LEVELS))]
    results["aggregate"] = {"mlp": agg_mlp, "oracle": agg_orc}

    print(f"\nAggregate degradation:")
    for i, nl in enumerate(NOISE_LEVELS):
        print(f"  σ={nl:.2f}: MLP={agg_mlp[i]:.4e}  Oracle={agg_orc[i]:.4e}  "
              f"ratio={agg_mlp[i]/max(agg_orc[i],1e-12):.1f}×")

    with open(out/"results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
