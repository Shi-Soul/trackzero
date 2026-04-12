"""Adversarial hard-example mining for Stage 1C.

Strategy: evaluate the current tracker on many candidate trajectories,
select the hardest ones (highest tracking error), and retrain.
This implements the proposal's "adversarial reference generation" idea
using hard-example mining rather than a learned generator network.
"""

from __future__ import annotations

import numpy as np

from trackzero.config import Config
from trackzero.eval.harness import EvalHarness
from trackzero.sim.simulator import Simulator


def score_trajectories_by_tracker_error(
    cfg: Config,
    policy,
    candidate_states: np.ndarray,
    candidate_actions: np.ndarray,
    max_eval: int | None = None,
) -> dict:
    """Score each candidate trajectory by tracker's closed-loop MSE.

    Args:
        cfg: Environment config.
        policy: MLPPolicy with __call__(current_state, ref_next_state).
        candidate_states: (N, T+1, 4) reference state trajectories.
        candidate_actions: (N, T, 2) reference action trajectories.
        max_eval: Limit number of trajectories to evaluate.

    Returns:
        Dict with trajectory_scores (N,), per_trajectory_results, stats.
    """
    harness = EvalHarness(cfg)
    N = len(candidate_states)
    if max_eval is not None:
        N = min(N, max_eval)

    scores = np.zeros(N)
    mse_q = np.zeros(N)
    mse_v = np.zeros(N)

    for i in range(N):
        result = harness.evaluate_trajectory(
            policy, candidate_states[i], candidate_actions[i], trajectory_idx=i
        )
        scores[i] = result.mse_total
        mse_q[i] = result.mse_q
        mse_v[i] = result.mse_v
        if (i + 1) % 200 == 0:
            print(f"  Scored {i+1}/{N} trajectories, "
                  f"mean_err={scores[:i+1].mean():.6f}, "
                  f"max_err={scores[:i+1].max():.6f}")

    return {
        "trajectory_scores": scores,
        "mse_q": mse_q,
        "mse_v": mse_v,
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "max_score": float(np.max(scores)),
        "pct90_score": float(np.percentile(scores, 90)),
        "pct99_score": float(np.percentile(scores, 99)),
    }


def select_hardest_trajectories(
    candidate_states: np.ndarray,
    candidate_actions: np.ndarray,
    scores: np.ndarray,
    n_select: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the n_select trajectories with highest tracker error.

    Returns:
        selected_states, selected_actions, selected_indices
    """
    idx = np.argsort(scores)[::-1][:n_select]
    return candidate_states[idx], candidate_actions[idx], idx


def adversarial_iterative_collection(
    cfg: Config,
    policy,
    generate_fn,
    n_rounds: int,
    candidates_per_round: int,
    select_per_round: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Multi-round adversarial hard-example mining.

    Each round:
    1. Generate fresh candidates
    2. Score by current tracker error
    3. Select hardest ones
    4. Accumulate

    Args:
        cfg: Environment config.
        policy: Current tracker policy (updated externally between rounds).
        generate_fn: Callable(n_traj, seed) -> (states, actions).
        n_rounds: Number of mining rounds.
        candidates_per_round: Candidates to generate each round.
        select_per_round: How many to keep each round.
        seed: Random seed base.

    Returns:
        accumulated_states, accumulated_actions, round_stats
    """
    all_states = []
    all_actions = []
    round_stats = []

    for r in range(n_rounds):
        print(f"\n--- Adversarial round {r+1}/{n_rounds} ---")
        states, actions = generate_fn(candidates_per_round, seed + r * 10000)

        info = score_trajectories_by_tracker_error(
            cfg, policy, states, actions
        )
        sel_s, sel_a, sel_idx = select_hardest_trajectories(
            states, actions, info["trajectory_scores"], select_per_round
        )
        all_states.append(sel_s)
        all_actions.append(sel_a)

        stats = {
            "round": r + 1,
            "candidates": candidates_per_round,
            "selected": select_per_round,
            "mean_candidate_error": info["mean_score"],
            "pct90_candidate_error": info["pct90_score"],
            "max_candidate_error": info["max_score"],
            "mean_selected_error": float(
                info["trajectory_scores"][sel_idx].mean()
            ),
            "min_selected_error": float(
                info["trajectory_scores"][sel_idx].min()
            ),
        }
        round_stats.append(stats)
        print(f"  Selected {select_per_round} hardest: "
              f"mean_err={stats['mean_selected_error']:.6f}, "
              f"min_sel={stats['min_selected_error']:.6f}")

    return (
        np.concatenate(all_states, axis=0),
        np.concatenate(all_actions, axis=0),
        round_stats,
    )
