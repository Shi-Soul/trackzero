# Stage 1C: Entropy-Driven Coverage — Multi-Variant Synthesis

## Executive Summary

Stage 1C explored **five** coverage/diversity strategies from the proposal:

1. **Ensemble disagreement** — select high-uncertainty trajectories
2. **Bin rebalancing** — favor trajectories visiting rare 4D state bins
3. **Low-density / max-entropy proxy** — select trajectories in sparse regions
4. **Hybrid coverage** — z-score combination of density + rebalancing
5. **Hindsight relabeling** — roll out teacher on hard refs, relabel achieved motion

**Core finding: No Stage 1C method consistently beats the matched random baseline.**

The random-rollout baseline remains the strongest overall strategy. Each diversity
method helps on 1–2 specific OOD families but hurts on the critical `mixed_ood`
benchmark. This is a significant negative result — for this system, naive uniform
random data is hard to beat with post-hoc selection.

See `stage_1c_commands.md` for exact reproduction commands.
See `stage_1c_details.md` for per-variant failure-mode discussion.

---

## Experimental Setup

All Stage 1C variants share a matched budget:
- **Seed data:** 2000 random-rollout trajectories (mixed action types)
- **Candidate pool:** 12000 trajectories (except hindsight: 4000 teacher rollouts)
- **Selected:** 8000 trajectories merged with seed = 10000 total
- **Architecture:** MLP 512x4, 100 epochs, batch 65536, lr=1e-3, cosine schedule
- **Baseline:** `outputs/stage1c_random_matched/` (10000 pure random trajectories)
- **Reference:** `outputs/stage1b_scaled/` (old Stage 1B best)

---

## ID Results (Multisine Test Set)

| Method | mean MSE_total | max MSE_total | median MSE_total |
|---|---:|---:|---:|
| **random_matched** | **1.185e-4** | 2.156e-3 | 6.182e-5 |
| hybrid_coverage | 1.478e-4 | 2.759e-3 | 6.389e-5 |
| low_density | 1.532e-4 | 2.928e-3 | 7.221e-5 |
| disagreement | 1.662e-4 | 3.807e-3 | 8.321e-5 |
| rebalance_bins | 1.950e-4 | 7.620e-3 | 7.543e-5 |
| hindsight | 2.894e-4 | 1.816e-2 | 7.970e-5 |

Random baseline wins ID cleanly. Hindsight is worst (2.4x worse).

---

## OOD Results (mean MSE_total)

| OOD family | random | disagree | rebalance | density | hybrid | hindsight | stage1b | winner |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| chirp | **2.16e-4** | 3.01e-4 | 2.47e-4 | 2.49e-4 | 2.48e-4 | 3.32e-4 | 3.04e-4 | random |
| step | 3.34e-2 | **3.34e-2** | 3.92e-2 | 3.63e-2 | 3.68e-2 | 6.71e-2 | 4.71e-2 | disagree~random |
| random_walk | 1.74e-2 | 2.33e-2 | 1.60e-2 | 2.03e-2 | 2.44e-2 | 3.59e-2 | **1.38e-2** | stage1b |
| sawtooth | 1.56e-4 | 2.00e-4 | 1.98e-4 | **1.27e-4** | 1.41e-4 | 2.18e-4 | 2.37e-4 | density |
| pulse | 8.26e-5 | 1.23e-4 | 8.48e-5 | 9.52e-5 | 7.89e-5 | **7.28e-5** | 7.37e-5 | hindsight |
| mixed_ood | **4.83e-3** | 1.36e-2 | 2.97e-2 | 1.58e-2 | 1.67e-2 | 3.26e-2 | 9.69e-3 | random |

**Geometric mean across all 6 OOD families:**

| Method | Geomean MSE |
|---|---:|
| **random_matched** | **1.41e-3** |
| stage1b_scaled | 1.79e-3 |
| low_density | 1.81e-3 |
| hybrid_coverage | 1.86e-3 |
| rebalance_bins | 2.06e-3 |
| disagreement | 2.07e-3 |
| hindsight | 2.73e-3 |

---

## Key Research Findings

### 1. The random baseline is unreasonably strong

With mixed action types and 10k trajectories, random rollouts achieve broad 4D
state coverage that no selection method improves upon holistically.

### 2. Selection creates a coverage-specificity tradeoff

Each selector concentrates data in its target region at the expense of broad coverage:
- **Low-density** helps `sawtooth` but hurts `mixed_ood` 3.3x
- **Rebalance** helps `random_walk` slightly but hurts `mixed_ood` 6.2x
- **Hybrid** helps `sawtooth`+`pulse` but hurts `mixed_ood` 3.5x
- **Disagreement** hurts almost everything; curiosity trap

### 3. Hindsight is limited by teacher quality

Teacher has MSE=0.010 on mixed_ood refs. Achieved trajectories concentrate near
the teacher's comfort zone, not in the hard regions we need.

### 4. The binding constraint is likely capacity, not coverage

Val losses are similar (~0.002) across methods. The data *distribution* matters
more than visiting specific rare bins.

---

## Proposal Checklist

| Approach | Status | Outcome |
|---|---|---|
| State-space binning with rebalancing | Done | Negative |
| Maximum entropy / low-density | Done | Negative |
| Curiosity / ensemble disagreement | Done | Negative |
| Hindsight relabeling | Done | Negative |
| Adversarial reference generation | Not yet | Next candidate |

---

## Artifacts

| Directory | Description |
|---|---|
| `outputs/stage1c_random_matched/` | Matched random baseline |
| `outputs/stage1c_active_full/` | Disagreement active collection |
| `outputs/stage1c_rebalance_full/` | Bin rebalancing |
| `outputs/stage1c_density_full/` | Low-density selector |
| `outputs/stage1c_hybrid_full/` | Hybrid coverage |
| `outputs/stage1c_hindsight_full/` | Hindsight relabeling |
| `outputs/stage1c_ood_*` | OOD comparison runs |

---

## Next Steps: Stage 1D

Key insight from 1C: **reweighting existing rollout data doesn't help**.
Stage 1D should focus on **generating qualitatively new data**:
- Reachability-guided sampling
- Trajectory optimization as data generator
- Planning-based distillation (iLQR/CEM/MPPI)

Best model to carry forward: `outputs/stage1c_random_matched/best_model.pt`
