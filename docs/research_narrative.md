# Research Narrative: Data Distribution for Inverse Dynamics Learning

## Problem Statement

Given a fixed compute budget for training data collection, what distribution
of training data produces the best inverse dynamics model for a general
tracking task?

**System**: Double pendulum (4D state: q1, q2, v1, v2; 2D action: tau1, tau2)
**Task**: Track diverse reference trajectories (step, chirp, random walk, etc.)
**Metric**: Mean tracking MSE on a fixed 600-trajectory benchmark

## Methods Compared

| Method | Data Strategy | Data Size |
|--------|--------------|-----------|
| Random | Uniform random torques | 10k traj |
| Active | Uncertainty-based queries | 10k traj |
| MaxEnt RL | Maximum entropy exploration | 10k traj |
| Density | Low-density region targeting | 10k traj |
| DAgger | Random + benchmark-like references | 10k+1.2k traj |

All models: MLP (512x4 baseline, with HP sweep up to 2048x4).

## Key Findings

### 1. Active Learning Wins (But Barely)

Standard benchmark ranking:
1. active: 1.86e-3 (best)
2. hybrid_select: 2.16e-3
3. rebalance: 2.18e-3
4. random: 2.67e-3
5. maxent_rl: 23.9e-3 (9x worse)

Active is only 1.05x of theoretical best (1.78e-3), leaving little room
for improvement within the current paradigm.

### 2. Density Beats Coverage

Random data: 281 cells (0.6%), 77.5% benchmark coverage, 25,940 samples/cell
MaxEnt data: 1,684 cells (3.3%), 100% benchmark coverage, 12,764 samples/cell

Despite 4x more coverage, maxent performs 9x WORSE. The bottleneck is
sample density in task-relevant cells, not total coverage.

### 3. Hard Families Are Tail-Heavy, Not OOD

Step and random_walk contribute 93-99% of aggregate error. They are NOT
out-of-distribution (only 2-3% of states outside training range). They are
tail-heavy: q_std ~5 vs ~2 for easy families. The model sees these states
in training, just rarely.

### 4. Capacity Cannot Fix Bad Data

HP sweep results (epoch 30): maxent models stuck at val_loss ~0.127
regardless of architecture (512x4, 1024x6, 2048x4). The problem is
data distribution, not model capacity.

## Theoretical Framework

The tracking benchmark error decomposes as:

  E[MSE] = sum_c p_bench(c) * error(c)

where c = state-space cell, p_bench(c) = benchmark visitation frequency,
and error(c) depends on training sample density in cell c.

For a fixed data budget N:
- Random: concentrates N in 281 cells -> high density where it covers
- MaxEnt: spreads N across 1684 cells -> low density everywhere
- Active: targets cells where error(c) is high -> optimal allocation

The optimal strategy: allocate training samples proportional to
p_bench(c) * difficulty(c), where difficulty captures the local
function complexity.

## Running Experiments

8 GPU experiments testing:
1. HP sweep: Does larger capacity help random data? (preliminary: yes)
2. MaxEnt capacity: Can 2048x4 close the gap? (preliminary: no)
3. DAgger: Does adding benchmark-like references help? (preliminary: promising)

Results expected in 3-4 hours.

## Next Steps

1. Complete HP sweep + DAgger experiments
2. Run standard benchmark on all new models
3. Plot: coverage vs benchmark MSE (the key relationship)
4. Determine if DAgger can beat active learning
5. Write final synthesis with actionable conclusions
