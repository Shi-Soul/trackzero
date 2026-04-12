# Stage 1C: Entropy-Driven Coverage

> README Goal: Address coverage gaps from 1B with diversity pressure.
> Explore: state-space binning, max-entropy, ensemble disagreement,
> hindsight relabeling, adversarial reference generation.

## Summary

9 methods tested on the standard benchmark (600 traj, 6 families).
**Active learning (ensemble disagreement) is the champion**, but the
advantage over random is modest (1.4×). The bigger lever is data quantity
+ model capacity (Stage 1D), not data selection strategy.

## Standard Benchmark Ranking (All 1C Methods)

| Rank | Method | MSE | vs Oracle | Category |
|------|--------|-----|-----------|----------|
| 1 | active (disagreement) | 1.86e-3 | 10.0× | Selection |
| 2 | hybrid_select | 2.15e-3 | 11.7× | Hybrid |
| 3 | rebalance | 2.18e-3 | 11.8× | Selection |
| 4 | random (1B baseline) | 2.67e-3 | 14.4× | Baseline |
| 5 | density | 3.17e-3 | 17.1× | Selection |
| 6 | hybrid_weighted | 4.49e-3 | 24.3× | Hybrid |
| 7 | adversarial | 5.01e-3 | 27.1× | Generation |
| 8 | hindsight | 8.13e-3 | 44.0× | Relabeling |
| 9 | maxent_rl | 2.39e-2 | 129× | RL coverage |
| — | supervised (1A) | 3.52e-2 | 190× | Baseline |

## Per-Family Oracle Gaps (Best Method per Family)

| Family | Best Method | MSE | vs Oracle |
|--------|------------|-----|-----------|
| multisine | hybrid_weighted | 1.35e-5 | 0.7× |
| chirp | random | 7.76e-6 | 0.4× |
| sawtooth | rebalance | 9.50e-5 | 0.5× |
| pulse | active | 8.88e-6 | 0.5× |
| step | active | 4.35e-3 | 23.5× |
| random_walk | rebalance | 1.12e-2 | 60.7× |

**Easy families solved** (≤ 2× oracle). **Hard families dominate** 95%+ error.

---

## Per-Method Analysis

### Active Learning (Ensemble Disagreement) — #1

Trains 3-member ensemble on seed data, selects high-disagreement trajectories.
Naturally concentrates data where the model is uncertain.

- Best overall (1.86e-3)
- Wins because it allocates more data to hard-family-relevant states
- Not explicit coverage maximization — uncertainty-driven

### Rebalance (Bin Rebalancing) — #3

Selects trajectories that fill under-visited 4D state bins.

- Strong on easy families but slightly hurts mixed_ood (6.15× geomean)
- Redistributes data from dense regions to sparse ones
- Sparse regions may not be task-relevant

### Density (Low-Density Selection) — #5

Selects trajectories visiting sparse state-space regions.

- Marginally worse than random (17.1× vs 14.4×)
- Low-density regions are often low-relevance regions

### Adversarial Reference Generation — #7

Generates references that maximize tracker error.

- 27.1× oracle — worse than random
- The adversarial generator finds hard references but doesn't improve coverage
- Concentrates training on edge cases that don't help broadly

### Hindsight Relabeling — #8

Rolls out tracker on hard references, relabels achieved trajectory as target.

- 44× oracle — teacher too weak in early stages
- Relabeled data concentrates near rest state (policy hangs down)
- Needs warm-starting from a stronger policy

### MaxEnt RL — #9

RL agent trained to maximize state entropy (KDE-based reward).

- 129× oracle — catastrophic
- Achieves near-uniform coverage (38.8%) but spreads data too thin
- Train loss 0.43 vs 0.007 for random — model can't fit extreme states

---

## Deep Analysis: The Goldilocks Principle

Three competing hypotheses for why random is hard to beat:

**H1: Smooth ID function** — inverse dynamics is nearly linear in
the training region, so ANY data with sufficient density works.
→ Partially true for easy families, doesn't explain hard family gap.

**H2: Non-uniform sensitivity** — some state regions have much higher
Coriolis/centrifugal forces, requiring more data there.
→ Confirmed: step and rw spend more time in high-sensitivity tails.

**H3: Closed-loop error compounding** — small open-loop errors cascade
in closed-loop tracking, requiring coverage of recovery states.
→ Confirmed by restricted-velocity experiment: 172× worse when
trained on restricted data despite good val-loss.

**The regime map:**

| Regime | Example | Coverage | Result |
|--------|---------|----------|--------|
| Too narrow | restricted-v | 3.2% | 172× (catastrophic) |
| Just right | random | 9.9% | 14.4× (best) |
| Too broad | maxent_rl | 38.8% | 129× (catastrophic) |
| No coherence | reachability | single-step | 1000-4000× |

### Why Step and Random_Walk Are Hard

These families are NOT far from training distribution — only 2.8-5.8%
of test states fall outside training range. The real problem:

- **Tail-heavy visitation**: step/rw spend 2× more time in high-velocity
  tails where Coriolis forces are large and the model has fewer samples
- **Sustained extremes**: unlike oscillatory references, step/rw hold
  extreme states for many timesteps, accumulating prediction errors
- The benchmark tests **tail accuracy** — rare-but-valid states

---

## Reproduction Commands

All from repo root with `uv run python`:

```bash
# Active (ensemble disagreement)
uv run python scripts/train_active.py \
  --config configs/medium.yaml --val-data data/medium/test.h5 \
  --output-dir outputs/stage1c_active_full \
  --seed-trajectories 2000 --candidate-trajectories 12000 \
  --select-trajectories 8000 --ensemble-size 3 \
  --final-hidden-dim 512 --final-n-hidden 4 --final-epochs 100 \
  --device cuda:0

# Rebalance
uv run python scripts/train_stage1c_selector.py \
  --selector rebalance_bins \
  --output-dir outputs/stage1c_rebalance_full \
  [same shared args as above]

# Density
uv run python scripts/train_stage1c_selector.py \
  --selector low_density \
  --output-dir outputs/stage1c_density_full \
  [same shared args]

# Hindsight
uv run python scripts/train_hindsight.py \
  --base-model outputs/stage1b_scaled/best_model.pt \
  --output-dir outputs/stage1c_hindsight_full \
  --hindsight-trajectories 4000 --reference-source mixed_ood

# Standard benchmark evaluation
uv run python scripts/standard_benchmark.py \
  --output outputs/standard_benchmark_v2.json
```

See `usage.md` for full argument details.

---

## Conclusion

Active learning is the best 1C method (10× oracle), but the gain over
random (14.4×) is modest. The fundamental limitation is **data quantity
and model capacity**, not data selection strategy:

- Easy families: ALL methods achieve ≤ 2× oracle (solved)
- Hard families: even the best method is 23-60× above oracle
- MaxEnt RL proves that uniform coverage is harmful (Goldilocks principle)
- Closed-loop error compounding requires broad-but-learnable coverage

This motivates Stage 1D: scale up data (20K, 50K) and model capacity
(1024×6, 2048×3) rather than pursuing more exotic selection strategies.

## README Hypothesis Verdicts from 1C

| # | Hypothesis | Verdict |
|---|-----------|---------|
| H1 | Random insufficient | PARTIAL — ok for easy, bad for hard |
| H2 | Ensemble > density | CONFIRMED — active #1, density #5 |
| H3 | Adversarial → boundary | NOT CONFIRMED — 27.1×, worse than random |
| H4 | Hindsight useful | NOT CONFIRMED — 44×, teacher too weak |
| H6 | TRACK-ZERO gap on OOD | CONFIRMED — 13-20× better than supervised |
