# Stage 1B: Random Rollout Self-Supervision

Roll out double pendulum under random torques, collect (s, a, s') tuples,
train inverse dynamics. No access to any reference dataset.

## Benchmark Result

| Family | MSE | ×Oracle |
|--------|-----|---------|
| multisine | 1.22e-4 | 1.2× |
| chirp | 2.49e-4 | 0.8× |
| sawtooth | 1.03e-4 | 0.5× |
| pulse | 8.73e-5 | 0.7× |
| step | 9.49e-3 | 29.2× |
| random_walk | 5.94e-3 | 132.4× |
| **Aggregate** | **2.67e-3** | **14.4×** |

vs supervised (1A): **13× better aggregate**, driven by hard families.
Easy families: both near-oracle. Hard families: random rollout 12-15× better.

---

## Coverage Analysis

18 action distributions tested. Key metric: 4D state-space coverage
(occupied bins / 10K total bins over (q1, q2, v1, v2)).

### Coverage Ranking (Top 6)

| Distribution | Coverage | Entropy | Benchmark? |
|-------------|----------|---------|------------|
| bangbang | 0.755 | 0.877 | Bad (unlearnable) |
| bangbang_slow | 0.746 | 0.864 | Bad |
| ou_medium_high | 0.603 | 0.830 | Moderate |
| mixed_uniform | 0.626 | 0.777 | **Best** |
| ou_fast | 0.562 | 0.818 | Good |
| mixed_all_equal | 0.606 | 0.769 | Good |

### The Coverage-Learnability Tradeoff

Coverage predicts benchmark quality (Pearson r = −0.823 with log MSE),
**but** the highest-coverage distributions (bangbang) produce the worst
models. Bangbang creates discontinuous torque transitions the MLP can't
fit — high coverage, low learnability.

The useful range: coverage 0.50–0.63 with smooth action distributions.

### Coverage vs Density

| Method | Coverage | Benchmark Overlap | Bench MSE |
|--------|---------|-------------------|-----------|
| random | 9.9% (193 cells) | 98% | 2.67e-3 |
| maxent_rl | 38.8% (751 cells) | 100% | 2.39e-2 |

MaxEnt covers everything but performs 9× worse — density spread too thin.
Random misses cells but has 51× higher density where it matters.
**Sample density in task-relevant cells > raw coverage.**

---

## Data Scaling Law

MSE ∝ N^{−0.85} (log-linear). More data is the single biggest lever.

| N (trajectories) | Benchmark MSE |
|-----------------|---------------|
| 1K | ~3.0e-2 |
| 5K | ~5.0e-3 |
| 10K | 2.67e-3 |
| 20K | TBD (val approaching oracle) |
| 50K | TBD (training in progress) |

Action type ranking: mixed > multisine > ou > uniform > gaussian > bangbang.

---

## Key Findings

1. **Random rollout works**: 14.4× oracle, 13× better than supervised
2. **Coverage predicts quality** but must be balanced with learnability
3. **Density > coverage**: task-relevant density matters more than breadth
4. **Scaling is the dominant lever**: 2× data → ~1.8× better benchmark

## Stage 1B Status: ✅ COMPLETE