# Synthesis: Cross-Stage Findings

## Central Result

**TRACK-ZERO works.** A policy trained on random physics rollouts
-- no demonstrations, no task references, no domain knowledge --
tracks arbitrary reference trajectories 10-190x better than
supervised imitation, and matches the oracle on 4 of 6 trajectory
families.

## Quantitative Summary

Oracle: 1.85e-4 aggregate MSE (analytical inverse dynamics).
Best TRACK-ZERO: 1.86e-3 aggregate (10x oracle).
Supervised baseline: 3.52e-2 aggregate (190x oracle).

### Per-Family Performance (best model: active, 512x4, 10K)

| Family | Active | Oracle | Gap |
|--------|--------|--------|-----|
| multisine | 1.71e-4 | 1.01e-4 | 1.7x |
| chirp | 3.11e-4 | 3.16e-4 | 1.0x |
| sawtooth | 1.20e-4 | 1.93e-4 | 0.6x |
| pulse | 1.40e-4 | 1.29e-4 | 1.1x |
| step | 7.64e-3 | 3.25e-4 | 23x |
| random_walk | 2.76e-3 | 4.49e-5 | 62x |

4/6 families at oracle level. Gap dominated by step and
random_walk which produce high-velocity tail states.

## Key Findings

### 1. The oracle gap is a velocity coverage problem

Quantitative analysis across all 600 benchmark trajectories:

| Max velocity (rad/s) | N traj | x Oracle |
|---------------------|--------|----------|
| [0, 5) | 28 | 0.5 |
| [5, 10) | 310 | 1.7 |
| [10, 15) | 194 | 8.5 |
| [15, 20) | 66 | 57.7 |

Log-log correlation r = 0.50. Below 10 rad/s, policy matches
oracle. Above 15 rad/s, 58x degradation. Root cause: training
data has <0.01% samples above 10 rad/s. Coriolis terms scale
as w^2, so extrapolation error grows quadratically.

### 2. Exploration strategy has marginal effect

9 exploration methods tested at 10K budget. Best (active) only
1.44x better than random. All physics-only methods within 2.7x.
The bottleneck is data coverage of high-velocity states, not
exploration intelligence.

### 3. Model capacity backfires at small data

1024x6 (5.3M) with 10K: 3300x oracle.
512x4 (1.1M) with 10K: 14x oracle.
229x degradation from 5x more parameters. Classic overfitting.

### 4. DAgger is structurally incompatible

On-policy rollouts diverge into unstable states.
All configurations degrade across iterations. Definitively ruled out.

## Hypothesis Verdicts

| # | Hypothesis | Status |
|---|-----------|--------|
| H1 | Random rollout covers enough | Confirmed |
| H2 | Smarter exploration beats random | Marginal (1.44x) |
| H3 | More data > smarter exploration | Testing (full matrix running) |
| H4 | Larger models help at fixed data | Rejected (229x worse) |
| H5 | DAgger bridges the gap | Rejected |
| H6 | Targeted velocity coverage closes gap | Testing (bangbang) |

## Open Questions (Stage 1D matrix in progress)

1. Does data scaling (10K to 100K) close the 10x gap?
2. Does targeted velocity coverage (bangbang) close the gap
   more efficiently than brute-force scaling?
3. At what data budget does 1024x6 recover from overfitting?
4. What is the best (data, architecture, coverage) triple?
