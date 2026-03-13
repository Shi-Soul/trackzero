"""MLP inverse dynamics network and policy wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class InverseDynamicsMLP(nn.Module):
    """MLP that maps (current_state, ref_next_state) -> torques.

    Input normalization stats are stored as registered buffers
    so they survive save/load and device transfers.
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        hidden_dim: int = 256,
        n_hidden: int = 3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        input_dim = 2 * state_dim

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)

        # Normalization buffers (set during training)
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))

    def set_normalization(self, mean: np.ndarray, std: np.ndarray):
        """Set input normalization from numpy arrays."""
        self.input_mean.copy_(torch.from_numpy(mean).float())
        self.input_std.copy_(torch.from_numpy(std).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with input normalization.

        Args:
            x: (batch, 8) concatenation of [current_state, ref_next_state]

        Returns:
            (batch, 2) predicted torques (unclamped)
        """
        x = (x - self.input_mean) / self.input_std
        return self.net(x)


class MLPPolicy:
    """Wraps an InverseDynamicsMLP as a numpy-in numpy-out policy callable."""

    def __init__(
        self,
        model: InverseDynamicsMLP,
        tau_max: float = 5.0,
        device: str = "cpu",
    ):
        self.model = model.to(device).eval()
        self.tau_max = tau_max
        self.device = device

    def __call__(
        self, current_state: np.ndarray, ref_next_state: np.ndarray
    ) -> np.ndarray:
        """Policy interface: (current_state, ref_next_state) -> action."""
        x = np.concatenate([current_state, ref_next_state]).astype(np.float32)
        with torch.no_grad():
            x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)
            action = self.model(x_t).squeeze(0).cpu().numpy()
        return np.clip(action, -self.tau_max, self.tau_max)


def save_checkpoint(
    model: InverseDynamicsMLP,
    path: str | Path,
    metadata: Optional[dict] = None,
):
    """Save model checkpoint with optional metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "state_dim": model.state_dim,
        "action_dim": model.action_dim,
        "config": {
            "hidden_dim": model.net[0].in_features,  # approximate
        },
    }
    if metadata:
        checkpoint["metadata"] = metadata
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> InverseDynamicsMLP:
    """Load model from checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # Infer architecture from state dict
    # Count hidden layers by looking at net.{i}.weight keys
    layer_keys = sorted([k for k in state_dict if k.startswith("net.") and k.endswith(".weight")])
    n_layers = len(layer_keys)
    hidden_dim = state_dict[layer_keys[0]].shape[0]
    state_dim = checkpoint.get("state_dim", 4)
    action_dim = checkpoint.get("action_dim", 2)
    n_hidden = n_layers - 1  # last layer is output

    model = InverseDynamicsMLP(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
    )
    model.load_state_dict(state_dict)
    return model.to(device)
