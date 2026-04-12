# TRACK-ZERO Roadmap Status

_Last updated after Stage 1B/1C experiments._

## Current status

**Stage 0 and Stage 1A/1B are complete; Stage 1C has an initial negative-result pass.**

- **Stage 0 ✅**: Data pipeline, oracle, training loop, evaluation harness — all working. Multisine reference dataset generated; supervised baseline trained and benchmarked. Results in `docs/stage0_results.md`.
- **Stage 1A ✅**: Supervised inverse-dynamics MLP trained on 10k multisine trajectories (512×4, 100 epochs, GPU). ID MSE 1.22e-4. Results in `docs/stage1_results.md`.
- **Stage 1B ✅**: Random-rollout self-supervised dataset; coverage analysis, action-type and data-scaling ablations, OOD evaluation, learning curves. Results in `docs/stage_1b.md`.
- **Stage 1C ⚠️**: First active-collection pass implemented with ensemble disagreement. End-to-end pipeline works, but did not beat the strong random-rollout baseline. Results in `docs/stage_1c.md`.
- **Stage 1D 🧪**: Hindsight relabeling prototype scripts added, but no scaled experiment result is committed yet.
- **Stage 1E**: Ablation scripts exist; not complete at full scale.
- **Stage 2–4**: Not yet implemented.

## Progress estimate

| Scope | Estimate | Notes |
|-------|---------|-------|
| **Overall roadmap (Stage 0-4)** | **~35%** | Stage 0 and the first full Stage 1 research passes are in place. |
| **Stage 0** | **~100%** | Full pipeline + baseline results. |
| **Stage 1A** | **~100%** | Supervised model trained, evaluated, learning curves. |
| **Stage 1B** | **~100%** | Coverage, ablations, OOD eval, analysis doc written. |
| **Stage 1C** | **~50%** | First disagreement-based pass implemented and evaluated; result negative, iteration still needed. |
| **Stage 1D** | **~20%** | Hindsight relabeling prototype and training entrypoint exist; no committed result doc yet. |
| **Stage 1E** | **~50%** | Ablation scripts exist; capacity ablation not run at scale. |
| **Stage 2–4** | **~0%** | No implementation. |

## Key experimental results

### Stage 1A vs 1B (10k traj, 100 epochs, 512×4 MLP)

| Metric | Stage 1A (supervised) | Stage 1B (random rollout) |
|--------|----------------------|--------------------------|
| ID MSE total | **1.22e-4** | 1.76e-4 |
| OOD step MSE | 1.50e-1 | **4.71e-2** (3.2× better) |
| OOD random_walk MSE | 1.18e-1 | **1.38e-2** (8.5× better) |
| Val loss | 0.0357 | 0.0424 |

### Stage 1B coverage (no attractor concentration)
All action types achieve >87% q-space coverage on 50×50 grid. `mixed` recommended for best OOD performance.

### Data scaling (mixed, 40 ep): ID MSE ∝ n^{−1.1}
1250 traj → 1.51e-2 · 2500 → 4.89e-3 · 5000 → 2.09e-3 · 10000 → 8.24e-4

## What is done vs. what remains

### Done ✅
1. Double-pendulum dynamics and MuJoCo wrapper.
2. Inverse-dynamics oracle.
3. Full data pipeline (`trackzero.data.*` package).
4. Stage 0 supervised baseline + evaluation.
5. Stage 1A supervised training (10k/100ep).
6. Stage 1B: coverage analysis, action-type ablation, data-size ablation, OOD eval.
7. GPU training support (torch 2.5.1+cu124, 8× RTX 3080 Ti).
8. Visualization: learning curves, coverage grids, rollout vs reference trajectories.

### Short-term todo list

1. **Stage 1C second pass.** Replace raw trajectory-level disagreement with transition-level or hybrid learnable-coverage selectors.
2. **Stage 1D experiment run.** Use the hindsight pipeline to test whether achieved-trajectory relabeling beats the Stage 1B baseline on broader OOD families.
3. **Stage 1E — network capacity ablation at scale.** Current ablations are still smaller than the main Stage 1A/1B runs.
4. **Investigate failure modes.** The bimodal OOD error distribution (low median, high max) suggests some trajectories trigger chaotic rollouts. Curriculum or trajectory filtering may help.
5. **Stage 2 — begin design.** Define the infeasible-reference setting and its optimization baseline before adding training code.
