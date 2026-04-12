"""Multisine trajectory dataset generation.

Generates (state, action) trajectories by:
  1. Sampling random initial conditions.
  2. Sampling multisine torque sequences.
  3. Rolling out the MuJoCo simulator.
  4. Saving to HDF5.

Reproducibility: trajectory i is generated with seed `base_seed + i`, and the
initial-condition RNG is shared with the multisine RNG (q0/v0 are drawn first,
then actions). This matches the round-trip test in tests/test_dataset.py.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional, Union

import h5py
import numpy as np

from trackzero.config import Config
from trackzero.data.multisine import generate_multisine_actions
from trackzero.sim.simulator import Simulator


def _generate_trajectory(args: tuple) -> tuple[int, np.ndarray, np.ndarray]:
    """Module-level worker function (must be picklable for multiprocessing).

    Args:
        args: (trajectory_index, base_seed, cfg)

    Returns:
        (index, states (T+1, 4), actions (T, 2))
    """
    idx, base_seed, cfg = args
    rng = np.random.default_rng(base_seed + idx)

    q0 = rng.uniform(
        cfg.dataset.initial_state.q_range[0],
        cfg.dataset.initial_state.q_range[1],
        size=2,
    )
    v0 = rng.uniform(
        cfg.dataset.initial_state.v_range[0],
        cfg.dataset.initial_state.v_range[1],
        size=2,
    )
    actions = generate_multisine_actions(
        rng, cfg.dataset, cfg.pendulum, cfg.simulation.control_dt
    )

    sim = Simulator(cfg)
    states = sim.rollout(actions, q0=q0, v0=v0)
    return idx, states, actions


def generate_dataset(
    cfg: Config,
    path: Union[str, Path],
    n_trajectories: int,
    seed: int = 42,
    num_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Generate a multisine trajectory dataset and save to HDF5.

    Args:
        cfg: Project configuration.
        path: Output HDF5 file path.
        n_trajectories: Number of trajectories to generate.
        seed: Base random seed (trajectory i uses seed + i for full reproducibility).
        num_workers: Number of parallel workers. Defaults to cfg.dataset.num_workers.
                     Pass 0 to force serial execution.
        progress_callback: Optional callable(n_done, n_total) called after each
                           completed trajectory (may be out-of-order in parallel mode).

    Returns:
        Path to the saved HDF5 file.
    """
    if num_workers is None:
        num_workers = cfg.dataset.num_workers

    T = int(cfg.dataset.trajectory_duration / cfg.simulation.control_dt)
    all_states = np.zeros((n_trajectories, T + 1, 4), dtype=np.float64)
    all_actions = np.zeros((n_trajectories, T, 2), dtype=np.float64)

    args_list = [(i, seed, cfg) for i in range(n_trajectories)]

    if num_workers > 0:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_generate_trajectory, args): args[0]
                for args in args_list
            }
            done = 0
            for future in as_completed(futures):
                idx, states, actions = future.result()
                all_states[idx] = states
                all_actions[idx] = actions
                done += 1
                if progress_callback:
                    progress_callback(done, n_trajectories)
    else:
        for i in range(n_trajectories):
            idx, states, actions = _generate_trajectory(args_list[i])
            all_states[idx] = states
            all_actions[idx] = actions
            if progress_callback:
                progress_callback(i + 1, n_trajectories)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("states", data=all_states)
        f.create_dataset("actions", data=all_actions)

    return path
