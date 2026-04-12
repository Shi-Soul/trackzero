# Stage 1C: Entropy-Driven Coverage

Test whether smarter data selection can beat random rollout on the
standard benchmark.

## Benchmark Ranking (All Methods)

| Rank | Method | Bench MSE | ×Oracle |
|------|--------|-----------|---------|
| 1 | active (ensemble disagreement) | 1.86e-3 | 10.0× |
| 2 | hybrid_select | 2.15e-3 | 11.7× |
| 3 | rebalance (bin rebalancing) | 2.18e-3 | 11.8× |
| 4 | random (1B baseline) | 2.67e-3 | 14.4× |
| 5 | density (low-density selection) | 3.17e-3 | 17.1× |
| 6 | hybrid_weighted | 4.49e-3 | 24.3× |
| 7 | adversarial | 5.01e-3 | 27.1× |
| 8 | hindsight | 8.13e-3 | 44.0× |
| 9 | maxent_rl | 2.39e-2 | 129× |
| — | supervised (1A) | 3.52e-2 | 190× |

**Active learning is #1** — targets model uncertainty, allocates data to
hard-family-relevant states. But the margin over random is only 1.4×.

## Per-Family: Where the Gap Is

| Family | Best Method | ×Oracle |
|--------|------------|---------|
| multisine | hybrid_weighted | 0.7× |
| chirp | random | 0.8× |
| sawtooth | rebalance | 0.5× |
| pulse | active | 0.5× |
| step | active | **23.5×** |
| random_walk | rebalance | **60.7×** |

Easy families: **solved** by all methods. Hard families: **23-60× oracle**,
drive 95%+ of aggregate error. No 1C method substantially cracks this.

---

## Method Verdicts

| Method | Result | Why |
|--------|--------|-----|
| Active (disagreement) | Best overall | Targets uncertainty → concentrates on hard states |
| Rebalance | Close to active | Fills sparse bins, helps step/rw slightly |
| Density | Slightly worse than random | Low-density regions aren't task-relevant |
| Adversarial | 27× oracle | Finds hard references but doesn't improve coverage |
| Hindsight | 44× oracle | Teacher too weak early; data concentrates near rest |
| MaxEnt RL | 129× oracle | Uniform coverage spreads density too thin |

## The Goldilocks Principle

| Regime | Example | Coverage | Bench MSE |
|--------|---------|----------|-----------|
| Too narrow | restricted-velocity | 3.2% | catastrophic |
| Just right | random rollout | 9.9% | 2.67e-3 |
| Too broad | maxent_rl | 38.8% | 2.39e-2 |

Coverage must be balanced: enough to reach task-relevant states,
not so much that density is diluted everywhere.

---

## Conclusion

Data **selection strategy** gives at most 1.4× improvement (active vs random).
The remaining 10× oracle gap cannot be closed by selection alone — it
requires more data and more model capacity (→ Stage 1D).

## Hypothesis Verdicts

| # | Hypothesis | Verdict |
|---|-----------|---------|
| H1 | Random insufficient | Partial — ok easy, bad hard |
| H2 | Ensemble > density | Confirmed |
| H3 | Adversarial → boundary | Not confirmed |
| H4 | Hindsight useful | Not confirmed |
| H6 | TRACK-ZERO >> supervised | Confirmed (13-19×) |

## Stage 1C Status: ✅ COMPLETE