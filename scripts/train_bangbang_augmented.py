"""Train with bangbang-heavy data to test velocity coverage hypothesis.

Generates 5K bangbang (short hold) + 5K mixed random trajectories,
training a 512x4 model. If this closes the step/random_walk gap,
it proves high-velocity coverage is the key missing piece."""
import argparse, json, time, torch, numpy as np
from pathlib import Path
from trackzero.config import load_config
from trackzero.data.random_rollout import generate_random_rollout_data
from trackzero.data.dataset import TrajectoryDataset
from trackzero.data.ood_references import generate_ood_reference_data
from torch.utils.data import TensorDataset, DataLoader
from trackzero.policy.mlp import InverseDynamicsMLP

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bangbang", type=int, default=5000)
    parser.add_argument("--n-mixed", type=int, default=5000)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-hidden", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/bangbang_augmented_512x4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    device = torch.device(args.device)
    tau_max = cfg.pendulum.tau_max
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Generate bangbang data (short hold times for high velocity)
    print(f"Generating {args.n_bangbang} bangbang trajectories...")
    t0 = time.time()
    bb_s, bb_a = generate_random_rollout_data(
        cfg, args.n_bangbang, action_type="bangbang", seed=args.seed,
        use_gpu=True, gpu_device=args.device,
        action_params={"min_hold": 1, "max_hold": 10}
    )
    print(f"  Generated in {time.time()-t0:.0f}s")

    # Generate mixed random data
    print(f"Generating {args.n_mixed} mixed random trajectories...")
    t1 = time.time()
    mix_s, mix_a = generate_random_rollout_data(
        cfg, args.n_mixed, action_type="mixed", seed=args.seed + 1000,
        use_gpu=True, gpu_device=args.device
    )
    print(f"  Generated in {time.time()-t1:.0f}s")

    # Combine
    train_s = np.concatenate([bb_s, mix_s], axis=0)
    train_a = np.concatenate([bb_a, mix_a], axis=0)
    print(f"Combined: {train_s.shape}")

    # Analyze velocity coverage
    vels = np.abs(train_s[:, :, 2:4]).reshape(-1, 2)
    for thresh in [5, 10, 15, 20]:
        f1 = (vels[:, 0] > thresh).mean()
        f2 = (vels[:, 1] > thresh).mean()
        print(f"  |vel|>{thresh}: j1={f1*100:.2f}% j2={f2*100:.2f}%")

    # Prepare training pairs
    s_t = train_s[:, :-1].reshape(-1, 4)
    s_tp1 = train_s[:, 1:].reshape(-1, 4)
    u_t = train_a.reshape(-1, 2)
    X = np.concatenate([s_t, s_tp1], axis=-1).astype(np.float32)
    Y = (u_t / tau_max).astype(np.float32)
    print(f"Training pairs: {len(X)}")

    # Validation data (same as other experiments)
    val_ds = TrajectoryDataset("data/medium/test.h5")
    val_s, val_a = val_ds.get_all_states(), val_ds.get_all_actions()
    val_ds.close()
    val_st = val_s[:, :-1].reshape(-1, 4)
    val_stp1 = val_s[:, 1:].reshape(-1, 4)
    val_ut = val_a.reshape(-1, 2)
    val_X = np.concatenate([val_st, val_stp1], axis=-1).astype(np.float32)
    val_Y = (val_ut / tau_max).astype(np.float32)

    model = InverseDynamicsMLP(
        state_dim=4, action_dim=2,
        hidden_dim=args.hidden_dim, n_hidden=args.n_hidden
    ).to(device)
    print(f"Model: {args.hidden_dim}x{args.n_hidden} ({sum(p.numel() for p in model.parameters()):,} params)")

    train_ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    val_ds_t = TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_Y))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds_t, batch_size=args.batch_size*2, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = torch.nn.MSELoss()

    best_val, best_state = float("inf"), None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
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
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
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

        scheduler.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{args.epochs}  train={train_loss:.6f}  val={val_loss:.6f}  best_val={best_val:.6f}  time={elapsed:.0f}s")

    if best_state:
        model.load_state_dict(best_state)
    torch.save({"model_state_dict": model.state_dict()}, out / "best_model.pt")
    print(f"\nSaved best model (val={best_val:.6f}) to {out}/best_model.pt")

if __name__ == "__main__":
    main()
