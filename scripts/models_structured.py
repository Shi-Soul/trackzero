#!/usr/bin/env python3
"""Structured dynamics architectures for inverse dynamics learning.

Key insight: rigid body inverse dynamics is τ = M(q)q̈ + C(q,q̇)q̇ + g(q),
which is LINEAR in acceleration. We exploit this by factoring the network
into configuration-dependent gains and state-dependent bias.

Architectures:
  RawMLP:       τ = f(q, v, q_ref, v_ref)           — flat baseline
  ResidualPD:   τ = Kp·Δq + Kd·Δv + ff(x)           — diagonal PD + feedforward
  FactoredMLP:  τ = A(q,v) @ [Δq;Δv] + b(q,v)       — full gain matrix + bias
"""
import numpy as np
import torch
import torch.nn as nn


def _build_mlp(in_dim, out_dim, hidden=512, layers=4):
    """Helper to build a simple MLP."""
    mods = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden, hidden), nn.ReLU()]
    mods.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*mods)


class RawMLP(nn.Module):
    """Flat MLP: (q, v, q_ref, v_ref) → τ."""
    def __init__(self, state_dim, action_dim, hidden=512, layers=4):
        super().__init__()
        self.net = _build_mlp(2 * state_dim, action_dim, hidden, layers)
        self.register_buffer("mu", torch.zeros(2 * state_dim))
        self.register_buffer("sigma", torch.ones(2 * state_dim))

    def set_norm(self, m, s):
        self.mu.copy_(torch.from_numpy(m).float())
        self.sigma.copy_(torch.from_numpy(s).float())

    def forward(self, x):
        return self.net((x - self.mu) / self.sigma)


class ResidualPD(nn.Module):
    """Diagonal PD + feedforward: τ = Kp(x)·Δq + Kd(x)·Δv + ff(x)."""
    def __init__(self, state_dim, action_dim, hidden=512, layers=4):
        super().__init__()
        self.nq = action_dim
        self.sd = state_dim
        self.net = _build_mlp(2 * state_dim, 3 * action_dim, hidden, layers)
        self.register_buffer("mu", torch.zeros(2 * state_dim))
        self.register_buffer("sigma", torch.ones(2 * state_dim))

    def set_norm(self, m, s):
        self.mu.copy_(torch.from_numpy(m).float())
        self.sigma.copy_(torch.from_numpy(s).float())

    def forward(self, x):
        nq, sd = self.nq, self.sd
        q_t, v_t = x[:, :nq], x[:, nq:sd]
        q_ref, v_ref = x[:, sd:sd+nq], x[:, sd+nq:]
        out = self.net((x - self.mu) / self.sigma)
        Kp = nn.functional.softplus(out[:, :nq])
        Kd = nn.functional.softplus(out[:, nq:2*nq])
        ff = out[:, 2*nq:]
        return Kp * (q_ref - q_t) + Kd * (v_ref - v_t) + ff


class FactoredMLP(nn.Module):
    """Structured dynamics: τ = A(q,v) @ [Δq;Δv] + b(q,v).

    Exploits the linear-in-acceleration structure of inverse dynamics.
    A(q,v) is a full nq × 2nq matrix capturing cross-joint coupling.
    b(q,v) handles gravity + Coriolis (no target dependence).
    """
    def __init__(self, state_dim, action_dim, hidden=512, layers=4):
        super().__init__()
        self.nu = action_dim
        self.sd = state_dim
        # A: action_dim × state_dim — works for both fully- and under-actuated
        n_a_out = action_dim * state_dim
        self.gain_net = _build_mlp(state_dim, n_a_out, hidden, layers)
        self.bias_net = _build_mlp(state_dim, action_dim, hidden, layers)
        self.register_buffer("mu", torch.zeros(2 * state_dim))
        self.register_buffer("sigma", torch.ones(2 * state_dim))

    def set_norm(self, m, s):
        self.mu.copy_(torch.from_numpy(m).float())
        self.sigma.copy_(torch.from_numpy(s).float())

    def forward(self, x):
        nu, sd = self.nu, self.sd
        state_mu, state_sig = self.mu[:sd], self.sigma[:sd]
        state = (x[:, :sd] - state_mu) / state_sig
        error = x[:, sd:] - x[:, :sd]  # [Δq, Δv], dim = state_dim
        A = self.gain_net(state).reshape(-1, nu, sd)
        b = self.bias_net(state)
        return torch.bmm(A, error.unsqueeze(-1)).squeeze(-1) + b


class ContactFactoredMLP(nn.Module):
    """Factored dynamics with contact flag as auxiliary input.

    gain_net/bias_net receive (q, v, contact_flag) — contact mode
    informs the dynamics model.  Error is computed from (q, v) only.
    Input layout: [q, v, q_ref, v_ref, contact_flag].
    """
    def __init__(self, phys_state_dim, action_dim, n_contact=1,
                 hidden=512, layers=4):
        super().__init__()
        self.nu = action_dim
        self.sd = phys_state_dim
        self.n_contact = n_contact
        # A: action_dim × phys_state_dim
        n_a_out = action_dim * phys_state_dim
        cond_dim = phys_state_dim + n_contact
        self.gain_net = _build_mlp(cond_dim, n_a_out, hidden, layers)
        self.bias_net = _build_mlp(cond_dim, action_dim, hidden, layers)
        full_in = 2 * phys_state_dim + n_contact
        self.register_buffer("mu", torch.zeros(full_in))
        self.register_buffer("sigma", torch.ones(full_in))

    def set_norm(self, m, s):
        self.mu.copy_(torch.from_numpy(m).float())
        self.sigma.copy_(torch.from_numpy(s).float())

    def forward(self, x):
        nu, sd = self.nu, self.sd
        state_raw = x[:, :sd]
        target_raw = x[:, sd:2*sd]
        c_flag = x[:, 2*sd:]
        s_mu, s_sig = self.mu[:sd], self.sigma[:sd]
        state_n = (state_raw - s_mu) / s_sig
        c_mu, c_sig = self.mu[2*sd:], self.sigma[2*sd:]
        c_n = (c_flag - c_mu) / c_sig
        gi = torch.cat([state_n, c_n], dim=1)
        A = self.gain_net(gi).reshape(-1, nu, sd)
        b = self.bias_net(gi)
        error = target_raw - state_raw
        return torch.bmm(A, error.unsqueeze(-1)).squeeze(-1) + b


ARCH_REGISTRY = {
    "raw_mlp": RawMLP,
    "residual_pd": ResidualPD,
    "factored": FactoredMLP,
}
