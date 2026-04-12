# Oracle Gap Analysis

## Theoretical Lower Bound

The **finite-difference (FD) oracle** uses MuJoCo's `mj_inverse` to compute
exact inverse dynamics, then applies the same MLP input transform.
Its benchmark MSE is the irreducible error floor.

| Metric | Value |
|--------|-------|
| FD Oracle aggregate MSE | **1.85e-4** |
| Shooting Oracle MSE | 4.6e-8 |
| Best MLP (active) | 1.86e-3 |
| Gap: MLP / Oracle | **10.0×** |

## Per-Family Oracle Comparison

| Family | Oracle MSE | Active MSE | Gap |
|--------|-----------|-----------|-----|
| multisine | 1.01e-4 | 1.71e-4 | 1.7× |
| chirp | 3.16e-4 | 3.11e-4 | 1.0× |
| sawtooth | 1.93e-4 | 1.20e-4 | 0.6× |
| pulse | 1.29e-4 | 1.40e-4 | 1.1× |
| **step** | 3.25e-4 | 7.64e-3 | **23.5×** |
| **random_walk** | 4.49e-5 | 2.76e-3 | **61.6×** |

## Key Insight

**The entire 10× gap is from step + random_walk.**

Easy families are at 0.6–1.7× oracle — essentially solved.
If active matched oracle on step + random_walk alone, the aggregate
would drop to 1.85e-4 (exactly oracle level).

## Why Step and Random Walk Are Hard

Training data covers q2 ∈ [−7.2, 7.3] (1st–99th percentile).
Step benchmark trajectories reach q2 = 27.1 (**3.7× beyond training**).
Random walk reaches q2 = 27.6 (**3.8× beyond**).

These families produce sustained coherent forcing that drives the
pendulum far beyond the training distribution. Other families
(multisine, chirp, etc.) stay within q2 < 8.6.

## Error Concentration

Top 10 worst step trajectories account for 58.5% of total error.
Fixing top 10 step + top 10 random_walk → 78.9% improvement.
The error is highly concentrated in a small number of extreme
out-of-distribution trajectories.

## Strategies to Close the Gap

1. **Data scaling** (N^{−0.85} law): 20K traj → predicted 1.26e-3
2. **DAgger** (task-focused augmentation): val_loss 0.0012 at epoch 80
3. **Architecture** (depth > width): 1024×6 beats 2048×3
4. **Coverage targeting**: train on data that overlaps benchmark states
