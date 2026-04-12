"""GPU-side histogram density estimator for entropy reward.

Maintains a 4D histogram of visited states. Reward for a state is
r = 1/sqrt(count(bin(s)) + 1), encouraging visits to rare bins.

Math (from docs/math.md §5.2):
  r_t^cov = 1 / sqrt(N(b(x_t)) + 1)
where N(b) is the visit count of bin b.
"""

from __future__ import annotations

import torch


class HistogramDensity:
    """4D histogram density estimator on GPU."""

    def __init__(
        self,
        n_bins: int = 20,
        q_range: tuple[float, float] = (-3.14159, 3.14159),
        v_range: tuple[float, float] = (-15.0, 15.0),
        device: str = "cuda:0",
        decay: float = 0.999,
    ):
        self.n_bins = n_bins
        self.device = device
        self.decay = decay

        self.q_lo, self.q_hi = q_range
        self.v_lo, self.v_hi = v_range

        # 4D histogram: (n_bins, n_bins, n_bins, n_bins)
        self.counts = torch.zeros(
            n_bins, n_bins, n_bins, n_bins,
            dtype=torch.float32, device=device,
        )
        self.total = 0.0

    def _state_to_bins(self, states: torch.Tensor) -> torch.Tensor:
        """Convert (N, 4) states to (N, 4) bin indices."""
        s = states.clone()
        # q dims: wrap to [-pi, pi]
        s[:, 0] = torch.remainder(s[:, 0] + 3.14159, 6.28318) - 3.14159
        s[:, 1] = torch.remainder(s[:, 1] + 3.14159, 6.28318) - 3.14159
        # Clamp velocities
        s[:, 2] = s[:, 2].clamp(self.v_lo, self.v_hi)
        s[:, 3] = s[:, 3].clamp(self.v_lo, self.v_hi)

        # Map to [0, n_bins-1]
        b01 = ((s[:, :2] - self.q_lo) / (self.q_hi - self.q_lo) * self.n_bins)
        b23 = ((s[:, 2:] - self.v_lo) / (self.v_hi - self.v_lo) * self.n_bins)
        bins = torch.cat([b01, b23], dim=-1).long().clamp(0, self.n_bins - 1)
        return bins

    def update(self, states: torch.Tensor):
        """Update histogram with batch of states (N, 4)."""
        bins = self._state_to_bins(states)
        # Decay old counts slightly for non-stationarity
        self.counts *= self.decay
        # Scatter-add (CPU fallback for 4D indexing)
        for i in range(states.shape[0]):
            b = bins[i]
            self.counts[b[0], b[1], b[2], b[3]] += 1.0
        self.total = self.counts.sum().item()

    def update_batched(self, states: torch.Tensor):
        """Vectorized histogram update using flat indexing."""
        bins = self._state_to_bins(states)
        nb = self.n_bins
        flat_idx = (bins[:, 0] * nb**3 + bins[:, 1] * nb**2 +
                    bins[:, 2] * nb + bins[:, 3])
        self.counts *= self.decay
        counts_flat = self.counts.view(-1)
        counts_flat.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
        self.total = self.counts.sum().item()

    def reward(self, states: torch.Tensor) -> torch.Tensor:
        """Compute entropy reward: r = 1/sqrt(count + 1). Shape (N,)."""
        bins = self._state_to_bins(states)
        nb = self.n_bins
        flat_idx = (bins[:, 0] * nb**3 + bins[:, 1] * nb**2 +
                    bins[:, 2] * nb + bins[:, 3])
        counts = self.counts.view(-1)[flat_idx]
        return 1.0 / torch.sqrt(counts + 1.0)

    def coverage_stats(self) -> dict:
        """Return coverage statistics."""
        occupied = (self.counts > 0).sum().item()
        total_bins = self.n_bins ** 4
        return {
            "occupied_bins": int(occupied),
            "total_bins": total_bins,
            "coverage_pct": occupied / total_bins * 100,
            "total_visits": self.total,
            "entropy": self._entropy(),
        }

    def _entropy(self) -> float:
        """Compute empirical entropy of the histogram."""
        p = self.counts / (self.total + 1e-8)
        p = p[p > 0]
        return -(p * torch.log(p)).sum().item()
