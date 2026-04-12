# Stage 1 Synthesis: Data Distribution for Inverse Dynamics

## Executive Summary

We evaluated 40+ training strategies for learning inverse dynamics of a
double pendulum. Key finding: **performance is determined by coverage of
task-relevant states**, not total state-space coverage.

Oracle (finite-difference): 1.85e-4 MSE. Best MLP: 1.86e-3 (10× gap).
The gap is concentrated on 2 of 6 benchmark families (step: 23.5×,
random_walk: 61.6×) where states exceed training range by 3.7×.

## The Benchmark

600 trajectories across 6 families (100 each):
- **Easy**: multisine, chirp, sawtooth, pulse — all methods achieve ≤2× oracle
- **Hard**: step, random_walk — drives 95%+ of aggregate error

## Approaches Tested

### Data Collection Strategies (Stage 1C, 512×4 arch)

| Rank | Method | MSE | vs Oracle |
|------|--------|-----|-----------|
| 1 | active | 1.86e-3 | 10.0× |
| 2 | hybrid_select | 2.15e-3 | 11.6× |
| 3 | rebalance | 2.18e-3 | 11.8× |
| 4 | random | 2.67e-3 | 14.4× |
| 5 | density | 3.17e-3 | 17.1× |
| ... | maxent_rl | 2.39e-2 | 129× |

### Capacity Scaling (Stage 1D, random data)

| Architecture | Params | Val-Loss | Bench MSE |
|-------------|--------|----------|-----------|
| 512×4 (base) | 1.3M | 0.00418 | 2.67e-3 |
| 1024×4+wd | 2.6M | 0.00230 | TBD |
| 1024×6 | 5.3M | 0.00168 | TBD |
| 2048×3 | 8.4M | 0.00274 | TBD |

Finding: depth matters more than width (1024×6 > 2048×3 despite fewer params).

### DAgger (Stage 1D, task-focused augmentation)

| Config | Iter | Val-Loss | Bench MSE |
|--------|------|----------|-----------|
| dagger_512x4 | 0 | 0.00120 | TBD |
| dagger_1024x4 | 0 | 0.00168 | TBD |

DAgger adds benchmark-like trajectories (step, random_walk, etc.) to training.
With just 1200 extra trajectories (12% increase), val_loss drops dramatically.

## Root Cause Analysis

### Why Step and Random_Walk Are Hard

1. **State range mismatch**: q2 max excursion 3.7× beyond training 99th percentile
2. **Error concentration**: Top 10 of 100 step trajectories = 85.7% of error
3. **Not inherently hard**: Oracle achieves 3.25e-4 on step (similar to easy families)

### Why Coverage Predicts Performance

Probe experiment (18 models, same architecture, same data size):
- Coverage vs benchmark MSE: r = −0.946 (excluding bangbang)
- Bangbang: highest coverage (75.5%) but worst performance → unlearnable dynamics

### Why MaxEnt Fails

- Spreads coverage across entire state space → low density in task-relevant regions
- Even 12.6M param model cannot compensate (val_loss > 0.028)
- 55× worse than random on easy families, 7× worse on hard families

## Conclusions

*To be completed when HP sweep and DAgger experiments finish.*

Preliminary conclusions:
1. **Data quality > model capacity** for this problem
2. DAgger (task-focused data) > active learning ≈ large random model
3. The practical recommendation: augment training with task-like references
4. Coverage of task-relevant states is the key predictor
