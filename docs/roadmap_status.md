# TRACK-ZERO Roadmap Status

## Current status

This repo has a **solid Stage 0 / Stage 1 scaffold**, but it is **not yet in a runnable research state**.

- **Implemented well:** MuJoCo double-pendulum simulator, inverse-dynamics oracle, evaluation harness, MLP policy/training core, plotting/playback helpers, GPU simulator prototype.
- **Partially implemented:** Stage 1 script layer exists for **1A supervised**, **1B random rollout**, **OOD eval**, and **1E ablations**.
- **Missing / blocking:** the whole `trackzero.data.*` package referenced by scripts and tests is absent (`multisine`, `dataset`, `generator`, `random_rollout`, `ood_references`), so the data path is broken end-to-end.
- **Environment baseline after `uv sync --extra dev`:** core packages now install successfully, but test collection still fails because `trackzero.data` is missing and the GPU path imports `mujoco_warp`, which is not declared as a project dependency.
- **Experiment evidence missing:** there is currently no `data/`, `outputs/`, or saved evaluation artifact in the repo, so no stage completion claim is backed by results yet.

## Progress estimate

These are rough engineering-progress estimates from the current repository state:

| Scope | Estimate | Notes |
| --- | --- | --- |
| **Overall roadmap (Stage 0-4)** | **~10-15%** | Mostly early infrastructure and research scaffolding; later stages are not present. |
| **Stage 0** | **~50-60%** | Simulator/oracle/eval core exists, but dataset generation/loading and reproducible completion evidence are missing. |
| **Stage 1** | **~15-25%** | Training/eval scripts are drafted, but the shared data pipeline is missing, so 1A/1B/1E are not actually executable end-to-end yet. |
| **Stage 2-4** | **~0%** | No implementation found. |

## What is done vs. what remains

### Done

1. Double-pendulum dynamics and MuJoCo wrapper.
2. Inverse-dynamics oracle with exact/shooting-style recovery.
3. Tracking evaluation harness and core metrics.
4. Basic inverse-dynamics MLP training loop.
5. Stage-labeled entry scripts for supervised/random-rollout/ablation/OOD evaluation.

### Remaining before Stage 0 can be considered complete

1. Recreate the missing `trackzero.data` package.
2. Make dataset generation/loading work end-to-end.
3. Decide whether GPU simulation is required now; if yes, add and validate the `mujoco_warp` dependency path.
4. Produce coverage statistics/plots for the multisine reference set.
5. Run and save oracle/open-loop validation results.
6. Generate reproducible baseline artifacts under `data/` and `outputs/`.

## Short-term todo list

1. **Restore the data layer first.** Implement or recover `trackzero.data.multisine`, `dataset`, `generator`, `random_rollout`, and `ood_references`; until this exists, most scripts/tests cannot run.
2. **Repair the post-sync test baseline.** After `uv sync --extra dev`, fix the current collection blockers: missing `trackzero.data` and missing `mujoco_warp` for GPU tests.
3. **Close Stage 0 with a small reproducible run.** Generate a tiny train/test dataset, verify oracle/open-loop evaluation, and save a minimal coverage report.
4. **Make Stage 1A actually executable.** Train the supervised baseline on the generated dataset and save checkpoint + eval JSON.
5. **Only then compare Stage 1B.** Train random-rollout baseline, run OOD evaluation, and use those outputs to decide whether Stage 1C/1D work is justified.
