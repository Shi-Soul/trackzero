# Coverage Analysis

## Coverage vs Benchmark Performance

The fundamental question: does higher training data coverage lead to better benchmark
performance? The answer is nuanced.

## Coverage Metrics (15 bins per dimension, 4D state space)

| Dataset | Total Cells | Total % | Benchmark Overlap | Overlap % | Missing |
|---------|------------|---------|-------------------|-----------|---------|
| random (train.h5) | 4,990 | 9.9% | 3,062/3,123 | **98.0%** | 61 |
| maxent_rl | 19,663 | 38.8% | 2,897/3,123 | 92.8% | 226 |
| combined (random+maxent) | 19,954 | 39.4% | 3,122/3,123 | **100.0%** | 1 |

Benchmark itself occupies 3,123 / 50,625 cells (6.2%).

## Key Insight: Targeted Coverage > Total Coverage

Maxent_rl has **4× more total coverage** than random (38.8% vs 9.9%), but
**5.2% less benchmark overlap** (92.8% vs 98.0%).

This explains the benchmark ranking:
- Random (rank #4, MSE=2.67e-3): high overlap (98%) → good benchmark performance
- Maxent (rank #9, MSE=2.39e-2): low overlap (92.8%) + data spread too thin → poor

The issue: maxent spreads its training budget across 4× more state space,
meaning each covered region gets ~4× fewer samples. For the benchmark-relevant
states, this dilution hurts accuracy.

## Implications for Data Collection Strategy

1. **Coverage alone is insufficient** — training data must overlap with the task.
2. **Random multisine is surprisingly well-targeted** for this benchmark.
3. **The ideal strategy combines**: broad coverage (maxent) + dense sampling in
   task-relevant regions (random).
4. This motivates **hybrid_weighted** (#6, MSE=4.49e-3) and future DAgger approaches
   that adaptively focus on task-relevant states.

## Why Active Learning Wins

Active learning (#1) selects training points based on model uncertainty.
This naturally concentrates data where it matters most — regions the model
currently handles poorly, which tend to be the hard benchmark families.

The active method achieves this without explicit coverage maximization.
It's a coverage-accuracy compromise that emerges from the uncertainty criterion.
