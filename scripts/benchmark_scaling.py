"""Benchmark new scaling + bangbang models only."""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from trackzero.config import load_config
from trackzero.data.dataset import TrajectoryDataset
from trackzero.data.ood_references import generate_ood_reference_data
from trackzero.eval.harness import EvalHarness
from trackzero.policy.mlp import InverseDynamicsMLP, MLPPolicy

BENCHMARK_SEED = 12345
N_PER_FAMILY = 100

MODELS = {
    # Baselines from v4 benchmark
    "active_512x4": ("outputs/stage1c_active_full/best_model.pt", 512, 4),
    "random_10k_512x4": ("outputs/stage1c_random_matched/best_model.pt", 512, 4),
    # New scaling models
    "random_20k_512x4": ("outputs/random_20k_512x4/best_model.pt", 512, 4),
    "random_20k_1024x6": ("outputs/random_20k_1024x6/best_model.pt", 1024, 6),
    "random_50k_1024x6": ("outputs/random_50k_1024x6/best_model.pt", 1024, 6),
    "random_100k_1024x6": ("outputs/random_100k_1024x6/best_model.pt", 1024, 6),
    # Targeted exploration
    "bangbang_aug_512x4": ("outputs/bangbang_augmented_512x4/best_model.pt", 512, 4),
}

def main():
    device = torch.device("cpu")  # benchmark is MuJoCo-bound, CPU is fine
    cfg = load_config()
    tau_max = cfg.pendulum.tau_max
    harness = EvalHarness(cfg)

    models = {k: v for k, v in MODELS.items() if os.path.exists(v[0])}
    print(f"Found {len(models)}/{len(MODELS)} models")

    # Generate benchmark data
    print("Generating benchmark trajectories...")
    families = {}
    ds = TrajectoryDataset("data/medium/test.h5")
    all_s, all_a = ds.get_all_states(), ds.get_all_actions()
    ds.close()
    families["multisine"] = (all_s[:N_PER_FAMILY], all_a[:N_PER_FAMILY])
    for name in ["chirp", "step", "random_walk", "sawtooth", "pulse"]:
        print(f"  {name}...")
        s, a = generate_ood_reference_data(cfg, N_PER_FAMILY, action_type=name, seed=BENCHMARK_SEED)
        families[name] = (s, a)
    print("Done generating benchmark data.")

    # Load oracle for comparison
    oracle_agg_mse = 1.85e-4
    if os.path.exists("outputs/oracle_benchmark.json"):
        with open("outputs/oracle_benchmark.json") as f:
            od = json.load(f)
            if "oracle" in od and "_aggregate" in od["oracle"]:
                oracle_agg_mse = od["oracle"]["_aggregate"]["mean_mse"]

    all_results = {}
    for name, (path, hd, nh) in models.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {name} ({hd}x{nh})")
        t0 = time.time()
        model = InverseDynamicsMLP(state_dim=4, action_dim=2, hidden_dim=hd, n_hidden=nh).to(device)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        policy = MLPPolicy(model, tau_max=tau_max)

        results = {}
        all_mse = []
        for fname, (ref_s, ref_a) in families.items():
            summary = harness.evaluate_policy(policy, ref_s, ref_a, max_trajectories=N_PER_FAMILY)
            mse = float(summary.mean_mse_total)
            median = float(summary.median_mse_total)
            worst = float(summary.max_mse_total)
            results[fname] = {"mean_mse": mse, "median_mse": median, "max_mse": worst}
            all_mse.extend([r.mse_total for r in summary.results])
            print(f"  {fname:12s}: mean={mse:.4e}  med={median:.4e}")

        results["_aggregate"] = {
            "mean_mse": float(np.mean(all_mse)),
            "median_mse": float(np.median(all_mse)),
            "max_mse": float(np.max(all_mse)),
        }
        agg = results["_aggregate"]
        ratio = agg["mean_mse"] / oracle_agg_mse
        print(f"  AGGREGATE: mean={agg['mean_mse']:.4e}  ({ratio:.1f}x oracle)  [{time.time()-t0:.0f}s]")
        all_results[name] = results

    # Save
    with open("outputs/scaling_benchmark.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Ranking
    print(f"\n{'='*60}")
    print("RANKING:")
    ranked = sorted(all_results.items(), key=lambda x: x[1]["_aggregate"]["mean_mse"])
    for i, (name, res) in enumerate(ranked, 1):
        agg = res["_aggregate"]
        ratio = agg["mean_mse"] / oracle_agg_mse
        print(f"  {i}. {name:30s} {agg['mean_mse']:.4e}  ({ratio:.1f}x oracle)")

if __name__ == "__main__":
    main()
