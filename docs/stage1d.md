# Stage 1D: Scaling and Architecture Ablations

## Research Question

Stage 1C showed all exploration methods cluster within 1.5x of
random at fixed 10K data budget. The best (active) is still 10x oracle.
Two scaling axes remain: **data quantity** and **model capacity**.
A third axis: **targeted coverage** (bangbang augmentation).

## Standard Benchmark Reference

600-trajectory benchmark (6 families x 100 trajectories).
Oracle aggregate MSE: **1.85e-4**.

### Stage 1C Ranking (10K data, 512x4)

| Rank | Method | Agg MSE | x Oracle |
|------|--------|---------|----------|
| 1 | active | 1.86e-3 | 10.0 |
| 2 | hybrid_select | 2.15e-3 | 11.6 |
| 3 | rebalance | 2.18e-3 | 11.8 |
| 4 | random | 2.67e-3 | 14.4 |
| 5 | density | 3.17e-3 | 17.1 |
| 10 | supervised_1a | 3.52e-2 | 190 |

All physics-only methods beat supervised by 10-190x.
DAgger variants all catastrophic (3200-3600x oracle).

### Per-Family Oracle Gap

| Family | Oracle | Active | Gap |
|--------|--------|--------|-----|
| multisine | 1.01e-4 | 1.71e-4 | 1.7x |
| chirp | 3.16e-4 | 3.11e-4 | 1.0x |
| sawtooth | 1.93e-4 | 1.20e-4 | 0.6x |
| pulse | 1.29e-4 | 1.40e-4 | 1.1x |
| step | 3.25e-4 | 7.64e-3 | 23x |
| random_walk | 4.49e-5 | 2.76e-3 | 62x |

4/6 families already at oracle. Gap is dominated by step and
random_walk (high-velocity tail states).

## Velocity-Error Analysis

| Max velocity (rad/s) | N traj | Mean MSE | x Oracle |
|---------------------|--------|----------|----------|
| [0, 5) | 28 | 1.01e-4 | 0.5 |
| [5, 10) | 310 | 3.16e-4 | 1.7 |
| [10, 15) | 194 | 1.57e-3 | 8.5 |
| [15, 20) | 66 | 1.07e-2 | 57.7 |

Log-log correlation r = 0.50. The oracle gap is a velocity
coverage problem: training data has <0.01% coverage above
10 rad/s for joint 1.

## Architecture at Fixed Data (10K)

| Architecture | Params | Agg MSE | x Oracle |
|-------------|--------|---------|----------|
| 512x4 | 1.1M | 2.67e-3 | 14 |
| 1024x6 | 5.3M | 6.11e-1 | 3300 |

At 10K, larger capacity causes 229x degradation. Classic
overfitting: 5M data points insufficient for 5.3M params.

## Full Experiment Matrix (in progress)

|  | 512x4 (1.1M) | 1024x6 (5.3M) |
|--|--------------|----------------|
| 10K random | 14x oracle (done) | 3300x oracle (done) |
| 20K random | GPU2, ep20, val 3.18e-3 | GPU0, ep70, val 3.09e-4 |
| 50K random | GPU3, data gen | GPU1, ep20, val 4.54e-4 |
| 100K random | GPU6, data gen | GPU7, ep10, val 8.10e-4 |
| 10K bangbang | GPU4, ep40, val 4.50e-3 | GPU5, ep1, val 2.01e-2 |

When training completes, this matrix answers:
1. How does benchmark MSE scale with data quantity?
2. Does 1024x6 recover at higher data budgets?
3. Does bangbang outperform random at equal budget?

## DAgger Analysis

DAgger is fundamentally incompatible with inverse dynamics:
each iteration compounds tracking errors. Benchmark MSE
worsens across iterations for all 4 architectures tested
(0.625 to 0.688). Definitively ruled out.
