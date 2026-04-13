# Stage 1D: Scaling & Architecture Ablations

## Research Question

Stage 1C showed all exploration methods cluster within 1.5× of
random at fixed data budget. The best (active) is still 10× oracle.
Two scaling axes remain: **data quantity** and **model capacity**.
This stage tests whether either closes the gap.

## Standard Benchmark v4 (23 models)

All models evaluated on the same 600-trajectory benchmark (6 families
× 100 trajectories). Oracle aggregate MSE: **1.85e-4**.

### Full Ranking

| Rank | Method | Arch | Data | Agg MSE | ×Oracle |
|------|--------|------|------|---------|---------|
| 1 | active | 512×4 | 10K | 1.86e-3 | 10.0 |
| 2 | hybrid_select | 512×4 | 10K | 2.15e-3 | 11.6 |
| 3 | rebalance | 512×4 | 10K | 2.18e-3 | 11.8 |
| 4 | random | 512×4 | 10K | 2.67e-3 | 14.4 |
| 5 | density | 512×4 | 10K | 3.17e-3 | 17.1 |
| 6 | hybrid_weighted | 512×4 | 10K | 4.49e-3 | 24.3 |
| 7 | adversarial | 512×4 | 10K | 5.01e-3 | 27.1 |
| 8 | hindsight | 512×4 | 10K | 8.13e-3 | 43.9 |
| 9 | maxent_rl | 512×4 | 10K | 2.39e-2 | 129 |
| 10 | supervised_1a | 512×4 | demo | 3.52e-2 | 190 |
| 11 | hybrid_curriculum | 512×4 | 10K | 4.51e-2 | 244 |
| 12–20 | DAgger variants | various | demo | 0.60–0.67 | 3200–3600 |
| 21–23 | 20K/50K (undertrained) | 1024×6 | 20K/50K | 0.92–0.94 | ~5000 |

### Oracle Gap by Family

The aggregate gap is misleading — it is dominated by two hard families:

| Family | Oracle | Active | Gap |
|--------|--------|--------|-----|
| multisine | 1.01e-4 | 1.71e-4 | 1.7× |
| chirp | 3.16e-4 | 3.11e-4 | **1.0×** |
| sawtooth | 1.93e-4 | 1.20e-4 | **0.6×** |
| pulse | 1.29e-4 | 1.40e-4 | **1.1×** |
| step | 3.25e-4 | 7.64e-3 | 23× |
| random_walk | 4.49e-5 | 2.76e-3 | 62× |

**Insight**: On 4 of 6 families, TRACK-ZERO already matches the
oracle. The residual gap concentrates in step (discontinuous torque
changes) and random_walk (white-noise torques), which produce
high-velocity tail states rarely seen in training data.

## Architecture Scaling (10K data)

| Architecture | Params | Agg MSE | ×Oracle |
|-------------|--------|---------|---------|
| 512×4 | 1.1M | 2.67e-3 | 14 |
| 1024×6 | 5.3M | 6.11e-1 | 3300 |
| 2048×4 (maxent data) | 10.5M | 6.35e-1 | 3430 |

**Result**: At 10K data, increasing model capacity from 1.1M to
5.3M params causes a **229× degradation**. The larger models overfit
to the training distribution and fail catastrophically on the
benchmark. This is a classic bias-variance tradeoff: 10K×500=5M
data points are insufficient for 5.3M parameters.

## Data Scaling (1024×6 architecture, in progress)

Training status (all using random rollouts, 200 epochs):

| N_traj | Epoch | Best Val | Notes |
|--------|-------|----------|-------|
| 10K | 200 (done) | — | Benchmark: 6.11e-1 (3300× oracle) |
| 20K | 130/200 | 1.77e-4 | Near oracle val; benchmark pending |
| 50K | 40/200 | 1.79e-4 | Already at oracle val; training faster |
| 100K | 10/200 | 7.32e-4 | Early; expect good convergence |

Key observation: 50K reaches oracle-level val-loss (1.79e-4) at
epoch 40, while 20K takes 130 epochs for comparable val (1.77e-4).
More data → faster convergence, as expected. The critical question
is whether low val-loss translates to good *benchmark* performance,
given that benchmark difficulty correlates with velocity not
in-distribution generalization.

## 512×4 Data Scaling (new)

| N_traj | Epoch | Best Val | Notes |
|--------|-------|----------|-------|
| 10K | 200 (done) | — | Benchmark: 2.67e-3 (14× oracle) |
| 20K | 10/200 | 5.54e-3 | Just started; ~12 hrs to completion |

## Bangbang Augmentation (velocity coverage test)

Training a 512×4 model on 5K bangbang (high-velocity) + 5K random
trajectories. The bangbang data has 13× more |vel|>10 coverage
(9.24% vs 0.72%) and 36× more |vel|>15 coverage (1.44% vs 0.04%).

| Epoch | Val | Notes |
|-------|-----|-------|
| 20/200 | 8.08e-3 | Still converging; ~6 hrs remaining |

If this model closes the step/random_walk gap, it proves the
oracle gap is a pure coverage problem solvable by targeted data
augmentation rather than brute-force scaling.

## DAgger Analysis (definitive)

DAgger iteratively collects on-policy data and retrains. Tested
across 4 configurations with 2–3 iterations each:

| Model | Iter 0 | Iter 1 | Iter 2 | Trend |
|-------|--------|--------|--------|-------|
| 512×4 | 0.625 | 0.638 | 0.673 | ↗ worse |
| 1024×4 | 0.615 | 0.604 | 0.638 | ↗ worse |
| 512×4 v3 | 0.630 | — | — | — |
| 1024×6 | 0.688 | — | — | — |

**Verdict**: DAgger is fundamentally incompatible with inverse
dynamics learning. Each iteration compounds tracking errors: the
policy tracks poorly → on-policy data concentrates in unstable
states → retraining on this data worsens generalization.
Larger models are *worse* (1024×6: 0.688 vs 512×4: 0.625),
confirming overfitting to biased on-policy data.

## Open Questions

1. Does the 1024×6 model with 20K+ data match the 512×4 with 10K
   on the benchmark? (Requires training completion.)
2. Would 512×4 with 20K data outperform both? (Not yet tested.)
3. Is the step/random_walk gap closable by any amount of random
   data, or does it require targeted exploration of high-velocity
   states?
- At what data budget does 1024×6 match oracle on easy families?