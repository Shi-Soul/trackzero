"""Analyze per-trajectory error distribution for the best model.
Identifies which trajectories cause the tail errors and their state characteristics."""
import sys, json, torch, numpy as np
sys.path.insert(0, ".")
from trackzero.config import load_config
from trackzero.policy.mlp import InverseDynamicsMLP, MLPPolicy
from trackzero.eval.harness import EvalHarness
from trackzero.data.dataset import TrajectoryDataset
from trackzero.data.ood_references import generate_ood_reference_data

N = 100
SEED = 42

def main():
    cfg = load_config()
    device = torch.device("cuda")
    tau_max = cfg.pendulum.tau_max
    harness = EvalHarness(cfg)

    # Load best model (active)
    model = InverseDynamicsMLP(state_dim=4, action_dim=2, hidden_dim=512, n_hidden=4).to(device)
    ckpt = torch.load("outputs/stage1c_active_full/best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    policy = MLPPolicy(model, tau_max=tau_max)

    # Generate benchmark families
    families = {}
    ds = TrajectoryDataset("data/medium/test.h5")
    all_s, all_a = ds.get_all_states(), ds.get_all_actions()
    ds.close()
    families["multisine"] = (all_s[:N], all_a[:N])
    for name in ["chirp", "step", "random_walk", "sawtooth", "pulse"]:
        s, a = generate_ood_reference_data(cfg, N, action_type=name, seed=SEED)
        families[name] = (s, a)

    print("Per-trajectory analysis (active model)")
    print("=" * 70)

    for fname, (ref_s, ref_a) in families.items():
        summary = harness.evaluate_policy(policy, ref_s, ref_a, max_trajectories=N)
        mses = np.array([r.mse_total for r in summary.results])

        # Sort and find outliers
        sorted_idx = np.argsort(mses)[::-1]
        p50 = np.median(mses)
        p90 = np.percentile(mses, 90)
        p99 = np.percentile(mses, 99)

        print(f"\n{fname}:")
        print(f"  p50={p50:.2e}  p90={p90:.2e}  p99={p99:.2e}  max={mses.max():.2e}")
        print(f"  Frac > 10×median: {(mses > 10*p50).sum()}/{len(mses)}")

        # Analyze worst 5 trajectories — their max velocities
        for rank, idx in enumerate(sorted_idx[:5]):
            traj_states = ref_s[idx]  # (T, 4)
            max_vel = np.abs(traj_states[:, 2:4]).max(axis=0)
            mean_vel = np.abs(traj_states[:, 2:4]).mean(axis=0)
            print(f"  Worst-{rank+1}: MSE={mses[idx]:.4e}  max_vel=[{max_vel[0]:.1f}, {max_vel[1]:.1f}]  mean_vel=[{mean_vel[0]:.1f}, {mean_vel[1]:.1f}]")

    print("\nDone.")

if __name__ == "__main__":
    main()
