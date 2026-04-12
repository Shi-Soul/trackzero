"""Minimal PPO for max-entropy exploration.

The exploration policy pi(a|s) outputs torques to maximize
state visitation entropy. This is NOT the tracking policy —
it's the DATA COLLECTION policy from the proposal §5.2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ExplorationPolicy(nn.Module):
    """Gaussian policy for continuous torque output."""

    def __init__(self, obs_dim: int = 4, act_dim: int = 2,
                 hidden: int = 128, n_layers: int = 2, tau_max: float = 5.0):
        super().__init__()
        self.tau_max = tau_max
        layers = [nn.Linear(obs_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor):
        h = self.backbone(obs)
        mu = self.mu_head(h)
        std = self.log_std.exp().expand_as(mu)
        return Normal(mu, std)

    def get_action(self, obs: torch.Tensor):
        """Sample action, return (action, log_prob, value_placeholder)."""
        dist = self.forward(obs)
        raw = dist.rsample()
        action = torch.tanh(raw) * self.tau_max
        # Log prob with tanh correction
        log_prob = dist.log_prob(raw) - torch.log(
            1 - (action / self.tau_max).pow(2) + 1e-6)
        return action, log_prob.sum(-1)


class ValueNet(nn.Module):
    """State value function V(s)."""

    def __init__(self, obs_dim: int = 4, hidden: int = 128, n_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(obs_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
