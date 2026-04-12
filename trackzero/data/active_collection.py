"""Active data-collection helpers for Stage 1C.

Implemented Stage 1C selectors:

1. ensemble disagreement
2. low-density / max-entropy proxy over 4D state bins
3. 4D bin rebalancing via rare-bin coverage score
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from trackzero.policy.mlp import InverseDynamicsMLP, save_checkpoint
from trackzero.policy.train import TrainingConfig, TrainingLog, train


STATE_RANGES = [
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-8.0, 8.0),
    (-8.0, 8.0),
]
BINS_PER_DIM = 10
N_BINS_TOTAL = BINS_PER_DIM ** 4


def bootstrap_trajectories(
    states: np.ndarray,
    actions: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample trajectories with replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(states), size=len(states))
    return states[idx], actions[idx]


def train_bootstrap_ensemble(
    train_states: np.ndarray,
    train_actions: np.ndarray,
    val_states: np.ndarray,
    val_actions: np.ndarray,
    ensemble_size: int,
    cfg: TrainingConfig,
    tau_max: float,
    device: str,
    output_dir: str | Path,
    seed: int,
) -> list[dict]:
    """Train bootstrap ensemble members and return model/log metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ensemble = []
    for member_idx in range(ensemble_size):
        member_seed = seed + member_idx
        member_states, member_actions = bootstrap_trajectories(
            train_states, train_actions, seed=member_seed
        )
        member_dir = output_dir / f"member_{member_idx:02d}"
        member_cfg = TrainingConfig(
            hidden_dim=cfg.hidden_dim,
            n_hidden=cfg.n_hidden,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            epochs=cfg.epochs,
            weight_decay=cfg.weight_decay,
            seed=member_seed,
            output_dir=str(member_dir),
        )
        _, logs = train(
            member_states,
            member_actions,
            val_states,
            val_actions,
            cfg=member_cfg,
            tau_max=tau_max,
            device=device,
        )
        model = torch.load(member_dir / "best_model.pt", map_location=device, weights_only=False)
        ensemble.append({
            "member_idx": member_idx,
            "seed": member_seed,
            "output_dir": str(member_dir),
            "best_model_path": str(member_dir / "best_model.pt"),
            "best_val_loss": min(log.val_loss for log in logs),
            "logs": [asdict(log) for log in logs],
            "checkpoint_metadata": model.get("metadata", {}),
        })
    return ensemble


def _predict_actions_batched(
    model: InverseDynamicsMLP,
    inputs: np.ndarray,
    device: str,
    batch_size: int = 65536,
) -> np.ndarray:
    model = model.to(device).eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(inputs[start:start + batch_size]).float().to(device)
            pred = model(batch).cpu().numpy()
            outputs.append(pred)
    return np.concatenate(outputs, axis=0)


def score_trajectory_disagreement(
    models: list[InverseDynamicsMLP],
    states: np.ndarray,
    device: str,
    batch_size: int = 65536,
) -> dict:
    """Score candidate trajectories by ensemble variance in predicted torque."""
    n_traj, t_plus_1, state_dim = states.shape
    current = states[:, :-1, :].reshape(-1, state_dim)
    next_s = states[:, 1:, :].reshape(-1, state_dim)
    inputs = np.concatenate([current, next_s], axis=1).astype(np.float32)

    preds = []
    for model in models:
        preds.append(_predict_actions_batched(model, inputs, device=device, batch_size=batch_size))
    pred_stack = np.stack(preds, axis=0)  # (E, N*T, 2)

    per_transition_var = np.var(pred_stack, axis=0).mean(axis=1)  # (N*T,)
    per_transition_std = np.sqrt(np.maximum(per_transition_var, 0.0))
    per_traj = per_transition_var.reshape(n_traj, t_plus_1 - 1)

    scores = per_traj.mean(axis=1)
    pct95 = np.percentile(per_traj, 95, axis=1)
    max_score = per_traj.max(axis=1)

    return {
        "trajectory_scores": scores,
        "trajectory_pct95_scores": pct95,
        "trajectory_max_scores": max_score,
        "mean_transition_std": float(per_transition_std.mean()),
        "max_transition_std": float(per_transition_std.max()),
    }


def select_top_trajectories(
    states: np.ndarray,
    actions: np.ndarray,
    scores: np.ndarray,
    n_select: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the top-scoring trajectories by disagreement."""
    if n_select <= 0 or n_select > len(states):
        raise ValueError(f"n_select must be in [1, {len(states)}], got {n_select}")
    order = np.argsort(scores)[::-1]
    chosen = order[:n_select]
    return states[chosen], actions[chosen], chosen


def _flatten_bin_indices(states: np.ndarray) -> np.ndarray:
    """Map each state to a flattened 4D histogram bin index."""
    flat = states.reshape(-1, states.shape[-1]).copy()
    mins = np.array([r[0] for r in STATE_RANGES], dtype=np.float64)
    maxs = np.array([r[1] for r in STATE_RANGES], dtype=np.float64)
    flat = np.clip(flat, mins, maxs)
    spans = np.maximum(maxs - mins, 1e-9)
    scaled = (flat - mins) / spans
    bins = np.floor(scaled * BINS_PER_DIM).astype(np.int64)
    bins = np.clip(bins, 0, BINS_PER_DIM - 1)
    return np.ravel_multi_index(bins.T, dims=(BINS_PER_DIM, BINS_PER_DIM, BINS_PER_DIM, BINS_PER_DIM))


def compute_bin_occupancy(states: np.ndarray) -> np.ndarray:
    """Count visits to each 4D state bin."""
    idx = _flatten_bin_indices(states)
    return np.bincount(idx, minlength=N_BINS_TOTAL).astype(np.float64)


def score_trajectories_low_density(
    seed_states: np.ndarray,
    candidate_states: np.ndarray,
) -> dict:
    """Score trajectories by inverse local state density.

    This is a simple proxy for maximum-entropy exploration:
    trajectories are preferred if they spend time in low-density bins.
    """
    occ = compute_bin_occupancy(seed_states)
    candidate_idx = _flatten_bin_indices(candidate_states).reshape(candidate_states.shape[0], -1)
    transition_scores = 1.0 / np.sqrt(occ[candidate_idx] + 1.0)
    trajectory_scores = transition_scores.mean(axis=1)
    return {
        "trajectory_scores": trajectory_scores,
        "trajectory_pct95_scores": np.percentile(transition_scores, 95, axis=1),
        "trajectory_max_scores": transition_scores.max(axis=1),
        "seed_occupied_bins": int((occ > 0).sum()),
        "seed_coverage": float((occ > 0).sum() / len(occ)),
        "mean_transition_score": float(transition_scores.mean()),
        "max_transition_score": float(transition_scores.max()),
    }


def score_trajectories_rebalance_bins(
    seed_states: np.ndarray,
    candidate_states: np.ndarray,
) -> dict:
    """Score trajectories by rare-bin breadth, not dwell time.

    Unique bins visited by a trajectory are scored with inverse occupancy,
    so trajectories that touch many under-visited bins are preferred.
    """
    occ = compute_bin_occupancy(seed_states)
    candidate_idx = _flatten_bin_indices(candidate_states).reshape(candidate_states.shape[0], -1)
    trajectory_scores = np.zeros(candidate_states.shape[0], dtype=np.float64)
    trajectory_pct95_scores = np.zeros(candidate_states.shape[0], dtype=np.float64)
    trajectory_max_scores = np.zeros(candidate_states.shape[0], dtype=np.float64)

    for i, row in enumerate(candidate_idx):
        uniq = np.unique(row)
        weights = 1.0 / (occ[uniq] + 1.0)
        trajectory_scores[i] = weights.sum()
        trajectory_pct95_scores[i] = np.percentile(weights, 95)
        trajectory_max_scores[i] = weights.max()

    return {
        "trajectory_scores": trajectory_scores,
        "trajectory_pct95_scores": trajectory_pct95_scores,
        "trajectory_max_scores": trajectory_max_scores,
        "seed_occupied_bins": int((occ > 0).sum()),
        "seed_coverage": float((occ > 0).sum() / len(occ)),
        "mean_transition_score": float(np.mean(1.0 / (occ[candidate_idx] + 1.0))),
        "max_transition_score": float(np.max(1.0 / (occ[candidate_idx] + 1.0))),
    }


def score_trajectories_hybrid_coverage(
    seed_states: np.ndarray,
    candidate_states: np.ndarray,
) -> dict:
    """Combine low-density dwell time and rare-bin breadth."""
    low = score_trajectories_low_density(seed_states, candidate_states)
    reb = score_trajectories_rebalance_bins(seed_states, candidate_states)

    def norm(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return (x - x.mean()) / max(x.std(), 1e-8)

    low_rank = norm(low["trajectory_scores"])
    reb_rank = norm(reb["trajectory_scores"])
    hybrid_scores = low_rank + reb_rank

    return {
        "trajectory_scores": hybrid_scores,
        "trajectory_pct95_scores": 0.5 * (norm(low["trajectory_pct95_scores"]) + norm(reb["trajectory_pct95_scores"])),
        "trajectory_max_scores": 0.5 * (norm(low["trajectory_max_scores"]) + norm(reb["trajectory_max_scores"])),
        "seed_occupied_bins": low["seed_occupied_bins"],
        "seed_coverage": low["seed_coverage"],
        "mean_transition_score": float(0.5 * (low["mean_transition_score"] + reb["mean_transition_score"])),
        "max_transition_score": float(max(low["max_transition_score"], reb["max_transition_score"])),
    }
