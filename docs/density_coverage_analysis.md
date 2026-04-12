# Sample Density vs Coverage: The Key Tradeoff

## Core Finding

**High coverage ≠ good performance.** What matters is *sample density in task-relevant cells*.

## Setup

- State space: 4D (q1, q2, v1, v2), discretized into 15^4 = 50,625 cells
- Benchmark: 600 trajectories (6 families × 100), visiting 249 unique cells
- Training data: random (10k traj), maxent (10k traj from max-entropy RL)

## Coverage vs Density

| Dataset | Cells covered | Benchmark cells hit | Mean density/cell | Median density/cell |
|---------|--------------|--------------------|--------------------|---------------------|
| Random  | 281 (0.6%)   | 193 / 249 (77.5%)  | 25,940             | 388                 |
| Maxent  | 1,684 (3.3%) | 249 / 249 (100%)   | 12,764             | 5,750               |

## The Paradox

- **Maxent covers 100% of benchmark cells** but performs 9× worse (MSE 2.39e-2 vs 2.67e-3)
- **Random misses 56 benchmark cells** but excels in the 193 it covers

Why? In the 193 shared cells:
- Random: **25,940** samples/cell (concentrated)
- Maxent: **15,923** samples/cell (spread thin)

Random's 5M samples concentrate in 281 cells → massive density where it counts.
Maxent's 5M samples spread across 1,684 cells → lower density everywhere.

## Benchmark Time Distribution

| Family      | % time in random cells | % time in maxent cells |
|-------------|----------------------|----------------------|
| multisine   | 100%                 | 100%                 |
| chirp       | 100%                 | 100%                 |
| sawtooth    | 100%                 | 100%                 |
| pulse       | 100%                 | 100%                 |
| step        | 98.1%                | 100%                 |
| random_walk | 96.7%                | 100%                 |

Even step/random_walk (the "hard" families) spend 96-98% of their time in random-covered
cells. The 2-3% in uncovered cells drives the error.

## Implications

1. **Density > Coverage** for any fixed data budget
2. Active learning wins (#1, MSE 1.86e-3) because it targets high-uncertainty regions within
   the relevant distribution — increasing density exactly where it's needed
3. DAgger (task-focused augmentation) should be even better — it explicitly adds samples in
   benchmark-relevant cells
4. The ideal strategy: start with random (high density in core), add targeted samples for tails

## NEW: Probe Model Correlation Analysis (18 models)

Trained 18 models with identical architecture (256×3, 2K trajectories) but different
action types. Measured 4D state-space coverage and standard benchmark MSE.

**Key correlations:**
- coverage vs bench_mse (excl. bangbang): **r = −0.946**
- coverage vs log(bench_mse) (excl. bangbang): **r = −0.910**
- coverage vs bench_mse (all 18): **r = +0.401** (bangbang reverses it!)

**The Bangbang Paradox:** Highest coverage (75.5%) but worst performance (1.06).
Discontinuous dynamics create states the model cannot learn to track.

**Takeaway:** Within learnable action types, coverage is a near-perfect predictor.
But coverage must be paired with **learnability** — bangbang breaks this.

## Predictions for Running Experiments

| Experiment | Prediction | Reasoning |
|-----------|-----------|-----------|
| HP random (larger model) | ↓ MSE, maybe beat active | Better interpolation in 193 covered cells |
| HP maxent (larger model) | Still ~10× worse | Coverage is fine, density is the bottleneck |
| DAgger (random + bench refs) | Best overall | 100% coverage + high density in relevant cells |
