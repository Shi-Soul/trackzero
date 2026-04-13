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

## Bugfix: tau_max Normalization (commit 1188bfa)

All previous scaling experiments had a critical bug: training
targets were divided by tau_max=5.0 (`Y = u_t / tau_max`), but
`MLPPolicy.__call__()` expects raw torques and clips to
[-tau_max, tau_max]. Result: models applied 1/5 the correct
torque, yielding ~5000x oracle on all configurations.

Fix: `Y = u_t.astype(np.float32)` (raw torques, no normalization).
All scaling experiments relaunched from scratch.

## Full Experiment Matrix (relaunched post-bugfix)

|  | 512x4 (1.1M) | 1024x6 (5.3M) |
|--|--------------|----------------|
| 10K random | 14× oracle (pre-bugfix, ok) | 3300× (pre-bugfix, ok) |
| 20K random | GPU 2 | GPU 1 |
| 50K random | GPU 4 | GPU 3 |
| 100K random | GPU 6 | GPU 5 |
| 10K bangbang | GPU 7 | — |

Note: 10K results are from Stage 1C (different training script,
no bug). All new experiments use the fixed `train_20k_random.py`.

When training completes, this matrix answers:
1. How does benchmark MSE scale with data quantity?
2. Does 1024×6 recover at higher data budgets?
3. Does bangbang outperform random at equal budget?

## DAgger Analysis

DAgger is fundamentally incompatible with inverse dynamics:
each iteration compounds tracking errors. Benchmark MSE
worsens across iterations for all 4 architectures tested
(0.625 to 0.688). Definitively ruled out.
