"""Quick benchmark for a single model - runs in ~2 minutes on GPU."""
import argparse, json, time, torch, numpy as np
from pathlib import Path
from trackzero.config import load_config
from trackzero.data.dataset import TrajectoryDataset
from trackzero.data.ood_references import generate_ood_reference_data
from trackzero.eval.harness import EvalHarness
from trackzero.policy.mlp import InverseDynamicsMLP, MLPPolicy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to best_model.pt")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-hidden", type=int, default=4)
    parser.add_argument("--auto-detect", action="store_true", default=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = load_config()
    device = torch.device(args.device)
    
    # Load model
    ckpt = torch.load(args.model_path, map_location=device, weights_only=True)
    sd = ckpt["model_state_dict"]
    
    if args.auto_detect:
        weight_keys = [k for k in sd if k.endswith('.weight')]
        args.n_hidden = len(weight_keys) - 1
        first_w = sd[weight_keys[0]]
        args.hidden_dim = first_w.shape[0]
        print(f"Auto-detected: {args.hidden_dim}x{args.n_hidden}")
    
    model = InverseDynamicsMLP(4, 2, args.hidden_dim, args.n_hidden).to(device)
    model.load_state_dict(sd)
    model.eval()
    policy = MLPPolicy(model, tau_max=cfg.pendulum.tau_max, device=device)
    harness = EvalHarness(cfg)
    
    # Load benchmark families
    ds = TrajectoryDataset("data/medium/test.h5")
    families = {"multisine": (ds.get_all_states()[:100], ds.get_all_actions()[:100])}
    ds.close()
    for name in ["chirp", "step", "random_walk", "sawtooth", "pulse"]:
        s, a = generate_ood_reference_data(cfg, 100, action_type=name, seed=12345)
        families[name] = (s, a)
    
    results = {}
    t0 = time.time()
    for fam, (s, a) in families.items():
        summary = harness.evaluate_policy(policy, s, a, max_trajectories=100)
        mse = float(summary.mean_mse_total)
        results[fam] = mse
        print(f"  {fam:12s}: {mse:.4e}")
    
    agg = np.mean(list(results.values()))
    results["_aggregate"] = agg
    print(f"  {'AGGREGATE':12s}: {agg:.4e}  ({time.time()-t0:.0f}s)")
    
    # Save alongside model
    out_path = Path(args.model_path).parent / "quick_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
