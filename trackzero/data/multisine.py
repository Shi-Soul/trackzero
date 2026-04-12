"""Multisine reference signal generation.

A multisine is a sum of sinusoids with random frequencies, amplitudes, and phases.
These provide smooth, quasi-random torque sequences that drive the pendulum through
a diverse region of state space while remaining physically feasible.
"""

from __future__ import annotations

import numpy as np

from trackzero.config import DatasetConfig, PendulumConfig


def sample_multisine_params(
    rng: np.random.Generator,
    dataset_cfg: DatasetConfig,
    pendulum_cfg: PendulumConfig,
) -> dict:
    """Sample random parameters for a multisine signal.

    Args:
        rng: NumPy random generator (mutated in place).
        dataset_cfg: Dataset configuration (k_range, freq_range).
        pendulum_cfg: Pendulum configuration (tau_max for amplitude scaling).

    Returns:
        params dict with keys:
            K (int): number of sinusoid components
            freqs (2, K): frequencies in Hz
            amplitudes (2, K): per-component amplitudes
            phases (2, K): initial phases in [0, 2*pi)
    """
    k_min, k_max = dataset_cfg.multisine.k_range
    f_min, f_max = dataset_cfg.multisine.freq_range
    tau_max = pendulum_cfg.tau_max

    K = int(rng.integers(k_min, k_max + 1))

    # Log-uniform frequency sampling for even coverage across scales
    freqs = np.exp(
        rng.uniform(np.log(f_min), np.log(f_max), size=(2, K))
    )

    # Amplitudes scaled so per-component RMS ~ tau_max / sqrt(K).
    # This keeps the sum's RMS near tau_max, with occasional light clipping.
    # The bound also ensures max diff-per-step << tau_max (smoothness guarantee).
    amplitudes = rng.uniform(0.0, tau_max / np.sqrt(K), size=(2, K))

    phases = rng.uniform(0.0, 2.0 * np.pi, size=(2, K))

    return {"K": K, "freqs": freqs, "amplitudes": amplitudes, "phases": phases}


def evaluate_multisine(
    params: dict,
    times: np.ndarray,
    tau_max: float,
) -> np.ndarray:
    """Evaluate a multisine signal at given time points.

    Args:
        params: Parameter dict from sample_multisine_params.
        times: (T,) array of time points in seconds.
        tau_max: Torque limit; output is clipped to [-tau_max, tau_max].

    Returns:
        actions: (T, 2) array of torques.
    """
    freqs = params["freqs"]       # (2, K)
    amps = params["amplitudes"]   # (2, K)
    phases = params["phases"]     # (2, K)

    # Vectorised broadcast: (2, K, 1) x (1, 1, T) -> (2, T)
    t = times[np.newaxis, np.newaxis, :]     # (1, 1, T)
    f = freqs[:, :, np.newaxis]              # (2, K, 1)
    a = amps[:, :, np.newaxis]               # (2, K, 1)
    p = phases[:, :, np.newaxis]             # (2, K, 1)

    signal = np.sum(a * np.sin(2.0 * np.pi * f * t + p), axis=1)  # (2, T)
    return np.clip(signal.T, -tau_max, tau_max)  # (T, 2)


def generate_multisine_actions(
    rng: np.random.Generator,
    dataset_cfg: DatasetConfig,
    pendulum_cfg: PendulumConfig,
    dt: float,
) -> np.ndarray:
    """Generate a multisine torque sequence for one trajectory.

    Args:
        rng: NumPy random generator (mutated in place — q0/v0 must be drawn
             from the same rng before calling this to ensure reproducibility).
        dataset_cfg: Dataset configuration.
        pendulum_cfg: Pendulum configuration.
        dt: Control timestep in seconds.

    Returns:
        actions: (T, 2) torque sequence clipped to tau_max.
    """
    T = int(dataset_cfg.trajectory_duration / dt)
    times = np.arange(T) * dt
    params = sample_multisine_params(rng, dataset_cfg, pendulum_cfg)
    return evaluate_multisine(params, times, pendulum_cfg.tau_max)
