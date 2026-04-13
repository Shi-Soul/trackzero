"""Train on 20K random trajectories with 1024x6 architecture.
Tests the scaling hypothesis: more data + bigger model → oracle-level performance.

Expected: ~1.26e-3 benchmark MSE based on N^{-0.85} scaling law.
"""
import argparse, json, time, torch, numpy as np
from pathlib import Path
from trackzero.config import load_config
from trackzero.data.random_rollout import generate_random_rollout_data
from trackzero.data.dataset import TrajectoryDataset
from trackzero.policy.mlp import InverseDynamicsMLP, MLPPolicy
from trackzero.eval.harness import EvalHarness
from trackzero.data.ood_references import generate_ood_reference_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-traj", type=int, default=20000)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--n-hidden", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/random_20k_1024x6")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    device = torch.device(args.device)
    tau_max = cfg.pendulum.tau_max
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Generate training data (GPU-accelerated if possible)
    print(f"Generating {args.n_traj} random trajectories...")
    t0 = time.time()
    train_s, train_a = generate_random_rollout_data(
        cfg, args.n_traj, action_type="mixed", seed=args.seed,
        use_gpu=True, gpu_device=args.device
    )
    print(f"  Generated in {time.time()-t0:.0f}s: {train_s.shape}")

    # Prepare training pairs
    s_t = train_s[:, :-1].reshape(-1, 4)
    s_tp1 = train_s[:, 1:].reshape(-1, 4)
    u_t = train_a.reshape(-1, 2)
    X = np.concatenate([s_t, s_tp1], axis=-1).astype(np.float32)
    Y = u_t.astype(np.float32)  # raw torques (MLPPolicy clips to [-tau_max, tau_max])
    print(f"Training pairs: {len(X)}")

    # Validation data
    val_ds = TrajectoryDataset("data/medium/test.h5")
    val_s, val_a = val_ds.get_all_states(), val_ds.get_all_actions()
    val_ds.close()
    vs_t = val_s[:, :-1].reshape(-1, 4)
    vs_tp1 = val_s[:, 1:].reshape(-1, 4)
    vu_t = val_a.reshape(-1, 2)
    X_val = np.concatenate([vs_t, vs_tp1], axis=-1).astype(np.float32)
    Y_val = vu_t.astype(np.float32)  # raw torques

    # Model
    model = InverseDynamicsMLP(4, 2, args.hidden_dim, args.n_hidden).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.hidden_dim}x{args.n_hidden} ({nparams:,} params)")

    # Move all data to GPU (avoids CPU→GPU transfer overhead per batch)
    X_train = torch.from_numpy(X).to(device)
    Y_train = torch.from_numpy(Y).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    Y_val_t = torch.from_numpy(Y_val).to(device)
    N_train, N_val = len(X_train), len(X_val_t)
    bs = args.batch_size
    bs_val = bs * 2

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = torch.nn.MSELoss()

    best_val, best_state = float("inf"), None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(N_train, device=device)
        total_loss, n = 0, 0
        for i in range(0, N_train, bs):
            idx = perm[i:i+bs]
            xb, yb = X_train[idx], Y_train[idx]
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            n += len(xb)
        train_loss = total_loss / n

        model.eval()
        total_loss, n = 0, 0
        with torch.no_grad():
            for i in range(0, N_val, bs_val):
                xb = X_val_t[i:i+bs_val]
                yb = Y_val_t[i:i+bs_val]
                loss = criterion(model(xb), yb)
                total_loss += loss.item() * len(xb)
                n += len(xb)
        val_loss = total_loss / n

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch >= 10:
                model.load_state_dict(best_state)
                torch.save({"model_state_dict": model.state_dict()}, out / "best_model.pt")
                model.train()

        # Save periodic checkpoints for post-hoc model selection
        if epoch % 50 == 0:
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                         "val_loss": val_loss}, out / f"checkpoint_ep{epoch}.pt")

        scheduler.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{args.epochs}  train={train_loss:.6f}  val={val_loss:.6f}  best_val={best_val:.6f}  time={elapsed:.0f}s")

    # Save best model
    if best_state:
        model.load_state_dict(best_state)
    torch.save({"model_state_dict": model.state_dict()}, out / "best_model.pt")
    print(f"\nSaved best model (val={best_val:.6f}) to {out}/best_model.pt")

    # Quick benchmark
    model.eval()
    policy = MLPPolicy(model, tau_max=tau_max, device=device)
    harness = EvalHarness(cfg)
    
    ds = TrajectoryDataset("data/medium/test.h5")
    families = {"multisine": (ds.get_all_states()[:100], ds.get_all_actions()[:100])}
    ds.close()
    for name in ["chirp", "step", "random_walk", "sawtooth", "pulse"]:
        s, a = generate_ood_reference_data(cfg, 100, action_type=name, seed=12345)
        families[name] = (s, a)
    
    results = {}
    for fam, (s, a) in families.items():
        summary = harness.evaluate_policy(policy, s, a, max_trajectories=100)
        mse = float(summary.mean_mse_total)
        results[fam] = mse
        print(f"  {fam}: {mse:.4e}")
    agg = np.mean(list(results.values()))
    results["_aggregate"] = agg
    print(f"  AGGREGATE: {agg:.4e}")
    
    with open(out / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
