# Research Narrative: Data Distribution for Inverse Dynamics Learning

## Problem Statement

Given a fixed compute budget for training data collection, what distribution
of training data produces the best inverse dynamics model for a general
tracking task?

**System**: Double pendulum (4D state: q1, q2, v1, v2; 2D action: tau1, tau2)
**Task**: Track diverse reference trajectories (step, chirp, random walk, etc.)
**Metric**: Mean tracking MSE on a fixed 600-trajectory benchmark

## 39-Model Comprehensive Ranking

We evaluated 39 models spanning 3 categories:
- **Stage 1C** (11 models): Different data strategies, 512×4 arch, 10K traj
- **Ablation** (10 models): Data size and action type sweeps
- **Probe** (18 models): Coverage-quality probes, 256×3 arch, 2K traj

Top 10:

| # | Method | MSE | Category |
|---|--------|-----|----------|
| 1 | active | 1.86e-3 | Stage 1C |
| 2 | hybrid_select | 2.15e-3 | Stage 1C |
| 3 | rebalance | 2.18e-3 | Stage 1C |
| 4 | random | 2.67e-3 | Stage 1C |
| 5 | density | 3.17e-3 | Stage 1C |
| 6 | hybrid_weighted | 4.49e-3 | Stage 1C |
| 7 | adversarial | 5.01e-3 | Stage 1C |
| 8 | ablation n10k | 7.72e-3 | Ablation |
| 9 | hindsight | 8.13e-3 | Stage 1C |
| 10 | ablation n5k | 1.12e-2 | Ablation |

## Theoretical Lower Bound (Oracle)

The finite-difference inverse dynamics oracle (mj_inverse) establishes the performance floor:

| Family | Oracle | Active MLP | Gap |
|--------|--------|-----------|-----|
| multisine | 1.01e-4 | 1.71e-4 | 1.7× |
| chirp | 3.16e-4 | 3.11e-4 | 1.0× |
| step | 3.25e-4 | 7.64e-3 | **23.5×** |
| random_walk | 4.49e-5 | 2.76e-3 | **61.6×** |
| sawtooth | 1.93e-4 | 1.20e-4 | 0.6× |
| pulse | 1.29e-4 | 1.40e-4 | 1.1× |
| **AGGREGATE** | **1.85e-4** | **1.86e-3** | **10.0×** |

Key takeaway: easy families are at oracle level (gap ≤ 2×).
The entire 10× aggregate gap is concentrated on step (23×) and random_walk (62×).
Step and random_walk are NOT inherently hard (oracle MSE ~3e-4 and ~4e-5).
The gap is 100% from MLP approximation error on extreme state excursions.

## Key Findings

### 1. Coverage Predicts Performance (r = −0.946)

Within 16 probe models using learnable action types (same arch, same data size),
4D state coverage is a near-perfect predictor of benchmark performance.

**But bangbang breaks the rule**: highest coverage (75.5%), worst performance.
Discontinuous dynamics create unlearnable states.

### 2. Density > Coverage Across Methods

| Method | Coverage | Density/cell | Bench MSE |
|--------|----------|-------------|-----------|
| Random | 98.0% | 25,940 | 2.67e-3 |
| MaxEnt | 92.8% | 12,764 | 23.9e-3 |
| Active | ~98% | targeted | 1.86e-3 |

MaxEnt has lower coverage AND lower density → 9× worse.

### 3. Data Size Scaling is Log-Linear

1K→2K→5K→10K trajectories: each doubling → ~2× improvement.
Ablation n10k (7.7e-3) vs Stage 1C random (2.7e-3) gap is due to
fewer training epochs (40 vs 100), not data quality.

### 4. Action Diversity is Critical

mixed >> multisine > ou > uniform > gaussian > bangbang.
Bangbang is 25× worse than mixed. Single action types cannot cover
the full dynamics needed for diverse tracking.

## Theoretical Framework

Benchmark error decomposes as:
  E[MSE] = Σ_c p_bench(c) · error(c)

Optimal data allocation: p_train(c) ∝ p_bench(c) · difficulty(c)

## Running Experiments (epoch ~70 update)

| Experiment | best_val | Status |
|-----------|---------|--------|
| **hp_random_1024x6** | **0.001675** | ★ Below active benchmark! |
| hp_random_1024x4_wd | 0.002366 | Still improving |
| hp_random_2048x3 | 0.002790 | Steady |
| hp_random_512x4_wd | 0.006290 | Slow convergence |
| **dagger_512x4** | **0.001540** | ★ Best val_loss overall |
| dagger_1024x4 | 0.001681 | Excellent |
| hp_maxent_1024x6 | 0.034639 | Hopeless (15× worse) |
| hp_maxent_2048x4 | >0.095 | Still no epochs logged |

### New Finding: Error Concentration

Active learning model: top 10 of 100 step trajectories = 85.7% of step error.
Root cause: q2 excursions beyond training range (q2_max up to 25.5 vs training max 14.3).

Fixing 10 worst step + 10 worst random_walk trajectories → 78.9% aggregate improvement.
Theoretical floor (all at median): 1.37e-4 (13.5× below current best).

This explains DAgger's success: benchmark-focused data targets exactly these hard states.

### New Finding: Capacity Closes the Gap

hp_random_1024x6 (epoch 70): val_loss 0.001675 < active benchmark 0.001858.
→ Random data + large model (4.7M params) ≈ active learning + default model (1.3M params).
→ Implications: data quality matters, but so does model capacity. The "best" strategy
  depends on the compute-capacity tradeoff.

## Open Questions

1. Will DAgger beat active on standard benchmark? (val_loss says YES)
2. Does hp_random_1024x6's val_loss advantage translate to benchmark? 
3. Does DAgger improve across iterations or plateau at iter 0?
4. What's the cost-performance Pareto frontier (data quality × model capacity)?
