"""PPO training loop for max-entropy exploration."""

from __future__ import annotations

import torch
import numpy as np

from trackzero.rl.ppo import ExplorationPolicy, ValueNet
from trackzero.rl.density import HistogramDensity


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < len(values) else 0.0
        mask = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_val * mask - values[t]
        advantages[t] = last_gae = delta + gamma * lam * mask * last_gae
    returns = advantages + values[:T]
    return advantages, returns


def ppo_update(policy, value_net, opt_pi, opt_v,
               obs_buf, act_buf, logp_buf, adv_buf, ret_buf,
               clip_eps=0.2, epochs=4, mb_size=2048):
    """Run PPO update on collected rollout buffer."""
    N = obs_buf.shape[0]
    adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)

    for _ in range(epochs):
        perm = torch.randperm(N, device=obs_buf.device)
        for start in range(0, N, mb_size):
            idx = perm[start:start + mb_size]
            obs_mb = obs_buf[idx]
            act_mb = act_buf[idx]
            logp_old = logp_buf[idx]
            adv_mb = adv_buf[idx]
            ret_mb = ret_buf[idx]

            dist = policy(obs_mb)
            raw = torch.atanh((act_mb / policy.tau_max).clamp(-0.999, 0.999))
            logp = dist.log_prob(raw) - torch.log(
                1 - (act_mb / policy.tau_max).pow(2) + 1e-6)
            logp = logp.sum(-1)

            ratio = (logp - logp_old).exp()
            clip_ratio = ratio.clamp(1 - clip_eps, 1 + clip_eps)
            pi_loss = -torch.min(ratio * adv_mb, clip_ratio * adv_mb).mean()
            # Entropy bonus
            ent = dist.entropy().sum(-1).mean()
            pi_loss = pi_loss - 0.01 * ent

            opt_pi.zero_grad()
            pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt_pi.step()

            v_pred = value_net(obs_mb)
            v_loss = ((v_pred - ret_mb) ** 2).mean()
            opt_v.zero_grad()
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
            opt_v.step()

    return pi_loss.item(), v_loss.item(), ent.item()
