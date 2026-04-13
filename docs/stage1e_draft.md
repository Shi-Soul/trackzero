# Stage 1E: Synthesis and Ablation Study

## Research Questions (from README)

1. Which mechanisms actually mattered (coverage strategy, data
   distribution, architecture, training procedure)?
2. What is the single best TRACK-ZERO configuration?
3. How does TRACK-ZERO compare to (a) supervised baseline,
   (b) inverse dynamics oracle?

## Experimental Design

### Axes of Variation

We test three independent axes that could close the oracle gap:

| Axis | Levels | Status |
|------|--------|--------|
| Data quantity | 10K, 20K, 50K, 100K random traj | Training |
| Architecture | 512×4 (1.1M), 1024×6 (5.3M) | Tested |
| Coverage strategy | random, active, bangbang-augmented | Training |

### Standard Benchmark

The evaluation is the 600-trajectory standard benchmark (6 signal
families × 100 trajectories), which represents the TASK distribution:
- Smooth families: multisine, chirp, sawtooth, pulse
- Hard families: step, random_walk (high-velocity tails)

The oracle achieves 1.85e-4 aggregate MSE. All results reported
as multiples of oracle.

## Results

### Completed: Stage 1C Exploration Methods (10K data, 512×4)

| Method | Agg (×oracle) | Step (×oracle) | Random Walk (×oracle) |
|--------|--------------|----------------|----------------------|
| active | 10.0 | 24 | 62 |
| hybrid_select | 11.7 | 27 | 82 |
| rebalance | 11.8 | 30 | 61 |
| random | 14.4 | 29 | 132 |
| density | 17.1 | 39 | 124 |
| supervised_1a | 190.5 | 364 | 2056 |

Key finding: all physics-only methods beat supervised by 10-190×.
Best exploration method (active) improves only 1.44× over random.

### Completed: Velocity-Error Analysis

| Max velocity | N traj | Mean MSE | ×Oracle |
|-------------|--------|----------|---------|
| [0, 5) | 28 | 1.01e-4 | 0.5 |
| [5, 10) | 310 | 3.16e-4 | 1.7 |
| [10, 15) | 194 | 1.57e-3 | 8.5 |
| [15, 20) | 66 | 1.07e-2 | 57.7 |

The oracle gap is a velocity coverage problem:
- Below 10 rad/s: at or below oracle (0.5-1.7×)
- Above 15 rad/s: 58× oracle
- Training data has <0.01% coverage above 10 rad/s (joint 1)

### Pending: Data Scaling (post-bugfix, training in progress)

All scaling experiments were relaunched after the tau_max
normalization bugfix (commit 1188bfa). Previous val-loss
numbers were invalid. Results TBD.

### Pending: Targeted Coverage (bangbang augmentation)

Bangbang augmented 512×4 training in progress (post-bugfix).
Result TBD.

## Analysis Framework (to fill when results arrive)

### Q1: Does more data close the gap?
Compare 10K vs 20K vs 50K at fixed architecture.
If yes: scaling law slope?

### Q2: Does targeted coverage close the gap?
Compare bangbang vs random at 10K data, same architecture.
If yes: how much improvement on step/random_walk specifically?

### Q3: Which matters more — quantity or targeting?
Compare 20K random vs 10K bangbang-augmented.
If bangbang 10K > random 20K: coverage quality dominates.
If random 20K > bangbang 10K: brute-force scaling is sufficient.

### Q4: Best single configuration?
Combine findings from Q1-Q3 to identify optimal
(data_quantity, architecture, coverage_strategy) triple.

## Stage 1 Completion Criteria Assessment

- [x] Match supervised on ID: all methods match within 2×
- [x] Beat supervised on OOD: 10-190× better
- [ ] Approach oracle broadly: currently 10× gap (tail-dominated)
- [ ] Clear understanding of essential mechanisms: pending experiments
