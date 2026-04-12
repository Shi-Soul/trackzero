# Coverage-Performance Relationship

## The Central Finding

Across 18 probe experiments with different action strategies:

| Subset | Pearson r | Spearman ρ | Interpretation |
|--------|-----------|-----------|----------------|
| All 18 probes | 0.061 | 0.335 | No correlation! |
| Excluding bangbang (16) | **−0.910** | **0.900** | Very strong |

**Coverage is the strongest predictor of benchmark performance**
(r = −0.91), but ONLY when the training data is learnable.

## The Bangbang Anomaly

| Probe | Coverage | Bench MSE | Rank |
|-------|----------|-----------|------|
| bangbang | **0.755** (highest) | **1.056** (worst) | 17/18 |
| bangbang_slow | 0.746 (2nd) | 1.080 (worst) | 18/18 |
| mixed_uniform | 0.626 | 0.053 (best) | 1/18 |

Bangbang achieves maximum state-space coverage by applying extreme
discontinuous torques. But these create **unlearnable training data**:
the transitions are so extreme that the MLP cannot generalize.

## Implications

### What Matters (in order)
1. **Learnability** — data must come from smooth, physical trajectories
2. **Coverage** — among learnable strategies, more coverage = better
3. **Relevance** — coverage of benchmark-relevant states matters most

### What Doesn't Work
- Maximizing coverage at the expense of learnability (bangbang)
- Maximizing entropy without physical constraints (maxent_rl)
- Narrow coverage even if highly learnable (gaussian_narrow)

### The Goldilocks Principle (Refined)
The best strategies occupy a sweet spot:
- Enough diversity to cover the benchmark state space
- Smooth enough actions that the MLP can learn the mapping
- Active learning wins because it targets the RELEVANT states

## Coverage Rankings (top 10 by benchmark)

| Rank | Strategy | Coverage | Bench MSE |
|------|----------|----------|-----------|
| 1 | mixed_uniform | 0.627 | 5.30e-2 |
| 2 | mixed_all_equal | 0.606 | 5.65e-2 |
| 3 | mixed_ou_heavy | 0.574 | 7.27e-2 |
| 4 | mixed_multisine_heavy | 0.562 | 7.61e-2 |
| 5 | ou_fast | 0.562 | 1.02e-1 |
| 6 | mixed_no_bang_gauss | 0.499 | 1.06e-1 |
| 7 | multisine | 0.507 | 1.07e-1 |
| 8 | mixed_smooth | 0.523 | 1.16e-1 |
| 9 | ou_medium_high | 0.603 | 1.19e-1 |
| 10 | ou_default | 0.516 | 1.35e-1 |

The pattern is clear: coverage 0.50–0.63 with diverse smooth actions
consistently outperforms both low-coverage (gaussian) and
high-coverage-unlearnable (bangbang) strategies.
