"""Training loop for supervised inverse dynamics learning."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from trackzero.policy.mlp import InverseDynamicsMLP, save_checkpoint


@dataclass
class TrainingConfig:
    hidden_dim: int = 256
    n_hidden: int = 3
    batch_size: int = 4096
    lr: float = 1e-3
    epochs: int = 20
    weight_decay: float = 0.0
    seed: int = 0
    output_dir: str = "outputs/stage1a"


@dataclass
class TrainingLog:
    epoch: int
    train_loss: float
    val_loss: float
    elapsed_s: float


def extract_pairs(states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract (input, target) pairs from trajectory data.

    Args:
        states: (N, T+1, 4) state trajectories
        actions: (N, T, 2) action sequences

    Returns:
        inputs: (N*T, 8) concatenation of [current_state, next_state]
        targets: (N*T, 2) torques
    """
    N, Tp1, sdim = states.shape
    T = Tp1 - 1

    current = states[:, :-1, :].reshape(-1, sdim)   # (N*T, 4)
    next_s = states[:, 1:, :].reshape(-1, sdim)      # (N*T, 4)
    targets = actions.reshape(-1, actions.shape[-1])  # (N*T, 2)

    inputs = np.concatenate([current, next_s], axis=1)  # (N*T, 8)
    return inputs, targets


def compute_normalization(inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std for input normalization."""
    mean = inputs.mean(axis=0)
    std = inputs.std(axis=0)
    std = np.maximum(std, 1e-8)  # avoid division by zero
    return mean, std


def train(
    train_states: np.ndarray,
    train_actions: np.ndarray,
    val_states: np.ndarray,
    val_actions: np.ndarray,
    cfg: Optional[TrainingConfig] = None,
    tau_max: float = 5.0,
    device: Optional[str] = None,
) -> tuple[InverseDynamicsMLP, list[TrainingLog]]:
    """Train an inverse dynamics MLP.

    Args:
        train_states: (N_train, T+1, 4)
        train_actions: (N_train, T, 2)
        val_states: (N_val, T+1, 4)
        val_actions: (N_val, T, 2)
        cfg: Training hyperparameters.
        tau_max: Torque limit (for metadata only).
        device: torch device string.

    Returns:
        (trained_model, training_logs)
    """
    if cfg is None:
        cfg = TrainingConfig()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f"Device: {device}")

    # Extract pairs
    print("Extracting training pairs...")
    train_inputs, train_targets = extract_pairs(train_states, train_actions)
    val_inputs, val_targets = extract_pairs(val_states, val_actions)
    print(f"  Train: {len(train_inputs):,} pairs")
    print(f"  Val:   {len(val_inputs):,} pairs")

    # Compute normalization from training data
    input_mean, input_std = compute_normalization(train_inputs)

    # Move all data to GPU upfront (40M pairs × 10 floats ≈ 1.5 GB — fits easily)
    train_x = torch.from_numpy(train_inputs).float().to(device)
    train_y = torch.from_numpy(train_targets).float().to(device)
    val_x = torch.from_numpy(val_inputs).float().to(device)
    val_y = torch.from_numpy(val_targets).float().to(device)

    # Free numpy arrays
    del train_inputs, train_targets, val_inputs, val_targets

    # Create model
    model = InverseDynamicsMLP(
        hidden_dim=cfg.hidden_dim,
        n_hidden=cfg.n_hidden,
    )
    model.set_normalization(input_mean, input_std)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=0
    )
    criterion = nn.MSELoss()

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Training for {cfg.epochs} epochs, batch_size={cfg.batch_size}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logs = []
    best_val_loss = float("inf")
    t0 = time.time()

    n_train = len(train_x)
    n_val = len(val_x)

    for epoch in range(1, cfg.epochs + 1):
        # Random batch order (generate permutation on CPU to save GPU memory)
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size].to(device, non_blocking=True)
            x_batch = train_x[idx]
            y_batch = train_y[idx]

            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches

        # Validate
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for i in range(0, n_val, cfg.batch_size * 4):
                x_batch = val_x[i : i + cfg.batch_size * 4]
                y_batch = val_y[i : i + cfg.batch_size * 4]
                pred = model(x_batch)
                val_loss_sum += criterion(pred, y_batch).item() * len(x_batch)
        val_loss = val_loss_sum / n_val

        scheduler.step()
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        log = TrainingLog(epoch=epoch, train_loss=train_loss, val_loss=val_loss, elapsed_s=elapsed)
        logs.append(log)

        print(f"  Epoch {epoch:3d}/{cfg.epochs}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"lr={lr:.2e}  [{elapsed:.1f}s]")

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, output_dir / "best_model.pt", metadata={
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "tau_max": tau_max,
                "training_config": asdict(cfg),
            })

    # Save final model too
    save_checkpoint(model, output_dir / "final_model.pt", metadata={
        "epoch": cfg.epochs,
        "train_loss": logs[-1].train_loss,
        "val_loss": logs[-1].val_loss,
        "tau_max": tau_max,
        "training_config": asdict(cfg),
    })

    print(f"Training complete. Best val loss: {best_val_loss:.6f}")
    print(f"Models saved to {output_dir}")

    return model, logs
