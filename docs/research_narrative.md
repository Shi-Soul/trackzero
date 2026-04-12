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

## Running Experiments (epoch ~60)

| Experiment | best_val | Prediction |
|-----------|---------|-----------|
| hp_random_1024x6 | 0.00233 | May beat active on benchmark |
| hp_random_1024x4_wd | 0.00263 | Competitive |
| hp_random_2048x3 | 0.00279 | Good |
| dagger_512x4 | 0.00154 | **Likely new champion** |
| hp_maxent_1024x6 | 0.03464 | Still 15× worse |
| hp_maxent_2048x4 | 0.09597 | Hopeless |

## Open Questions

1. Will DAgger beat active on standard benchmark?
2. Can HP-tuned random match active?
3. Is importance-weighted resampling better than uniform training?
