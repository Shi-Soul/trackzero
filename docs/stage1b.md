# Stage 1B: Random Exploration Baseline

## Research Question

Can a policy trained on data from random-torque rollouts (no task
signal, no human demonstrations) track arbitrary reference trajectories?
If so, what is the relationship between dataset size, state-space
coverage, and benchmark performance?

## Hypothesis

Random exploration covers the full reachable set uniformly and therefore
produces a model that generalizes to all signal families, unlike the
supervised baseline which is confined to one family's state distribution.

## Setup

**Data generation**: At each timestep, sample torques u_t uniformly
from the full torque range. The simulator runs for 500 steps per
trajectory. This produces maximally diverse state-action pairs with no
task-specific bias.

**Coverage**: 10K random-torque trajectories yield 24.5% coverage
(2,449 of 10,000 bins occupied) versus 1.3% for the multisine reference
data. See Stage 0 for the coverage metric definition.

**Training**: Same architecture (MLP 512×4), optimizer (Adam lr=1e-3),
and evaluation protocol as Stage 1A.

## Benchmark Results

| Family | MSE | ×Oracle | vs Supervised |
|--------|-----|---------|---------------|
| multisine | 3.83e-3 | 37.9 | 20× worse |
| chirp | 1.80e-3 | 5.7 | 8.7× worse |
| sawtooth | 2.01e-3 | 10.4 | 6.7× worse |
| pulse | 1.06e-3 | 8.2 | 17.7× worse |
| step | 4.59e-3 | 14.1 | **25.7× better** |
| random_walk | 2.70e-3 | 60.1 | **34.2× better** |
| **Aggregate** | **2.67e-3** | **14.4** | **13.2× better** |

## Analysis

**Result 1: Random exploration enables universal tracking.** The random
model achieves 14.4× oracle aggregate — dramatically better than the
supervised model's 190× — despite never seeing any reference trajectory
during training.

**Result 2: Coverage explains generalization.** The supervised model
fails on step/random_walk because its training data lacks high-velocity
states. The random model covers these regions and succeeds. But it pays
a cost on smooth families (multisine, pulse) where the supervised model's
narrow-but-dense coverage gives better local accuracy.

**Result 3: The coverage–precision tradeoff.** Universal coverage comes
at the cost of per-region density. With a fixed data budget, spreading
data uniformly means fewer samples in any particular state region, which
increases local approximation error.

This tradeoff motivates two subsequent investigations:
1. Can smarter exploration improve on random? (Stage 1C)
2. Can scaling the dataset close the remaining gap? (Stage 1D)