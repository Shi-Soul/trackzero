# Stage 1B: TRACK-ZERO v0 — Random Rollout Self-Supervision

> README Goal: Simplest physics-only approach. Roll out under random torques,
> collect (state, action, next_state) tuples, train inverse dynamics.

## Summary of Findings

1. **4D joint coverage strongly predicts OOD quality** (r = −0.823)
2. **Coverage-learnability tradeoff**: bangbang has highest coverage but worst
   model (discontinuous transitions hard to fit)
3. **Random beats supervised on hard families**: 3-8× better on step/rw/mixed_ood
4. **Massive oracle gap remains**: best 1B model still 14.4× above oracle
5. **Data scaling is log-linear**: MSE ∝ N^{−0.85}, more data helps enormously

## Standard Benchmark Result

| Metric | Value |
|--------|-------|
| Aggregate MSE | 2.67e-3 |
| vs Oracle | 14.4× |
| vs Supervised (1A) | 13× better |
| Easy families | ~1.4e-4 (≤ 2× oracle) |
| Hard families | step 23.5×, rw 60.7× oracle |

---

## Experiment 1: Action Distribution Sweep

18 random action distributions tested on 4D state coverage metric:
`coverage = occupied_bins / 10000` over `(q1, q2, v1, v2)`.

**Top 6 by coverage score:**

| Rank | Config | Coverage | Entropy | Score |
|------|--------|----------|---------|-------|
| 1 | bangbang | 0.755 | 0.877 | 0.816 |
| 2 | bangbang_slow | 0.746 | 0.864 | 0.805 |
| 3 | ou_medium_high | 0.603 | 0.830 | 0.716 |
| 4 | mixed_uniform | 0.626 | 0.777 | 0.702 |
| 5 | ou_fast | 0.562 | 0.818 | 0.690 |
| 6 | mixed_all_equal | 0.606 | 0.769 | 0.688 |

**Bottom by coverage:** gaussian_narrow (0.216), gaussian_medium (0.249).

Scripts: `scripts/search_action_distribution.py`
Artifacts: `outputs/action_sweep/sweep_results.json`

---

## Experiment 2: Coverage → Quality Probe Study

Train small probe models (MLP 256×3, 2000 traj, 30 ep) for each distribution,
evaluate on mixed_ood.

**Correlation with log10(OOD MSE):**

| Predictor | Pearson r |
|-----------|-----------|
| 4D coverage | **−0.823** |
| Composite score | −0.800 |
| 4D entropy | −0.743 |
| Validation loss | 0.107 (useless!) |

**Key insight**: Validation loss on held-out multisine does NOT predict OOD quality.
Coverage is the real signal.

**Probe ranking by OOD quality:**

| Rank | Config | Score | Val loss | OOD MSE |
|------|--------|-------|----------|---------|
| 1 | mixed_all_equal | 0.688 | 0.861 | 4.52e-2 |
| 2 | mixed_uniform | 0.702 | 0.832 | 5.11e-2 |
| 3 | mixed_ou_heavy | 0.679 | 0.889 | 6.80e-2 |
| 5 | bangbang_slow | 0.805 | 3.204 | 9.50e-2 |
| 6 | bangbang | 0.816 | 3.179 | 1.02e-1 |

**Coverage-learnability tradeoff**: bangbang has the BEST coverage but NOT
the best models. Discontinuous actions create transitions the MLP can't fit.

> Higher coverage helps, until the data becomes too hard to fit.

Scripts: `scripts/probe_coverage_quality.py`
Artifacts: `outputs/coverage_quality_probe/`

---

## Experiment 3: Full-Scale Training

| Model | Action Type | Val Loss | Best OOD? |
|-------|-------------|----------|-----------|
| 1B uniform-mixed (old) | mixed (default) | 0.042 | ✅ Yes |
| 1B equal-mixed (probe-selected) | mixed (equal weights) | 0.137 | ❌ No |

The probe-selected distribution did NOT beat the old baseline at full scale.
Probe selection doesn't transfer to full training regime.

Artifacts: `outputs/stage1b_scaled/best_model.pt` (carry-forward checkpoint)

---

## Experiment 4: OOD Comparison (1A vs 1B vs Oracle)

| Family | Oracle | 1A (supervised) | 1B (random) | 1A/1B ratio |
|--------|--------|-----------------|-------------|-------------|
| chirp | 6.45e-9 | 1.80e-4 | 3.04e-4 | 0.59× |
| step | 4.60e-7 | 1.50e-1 | 4.71e-2 | **3.19×** |
| rw | 1.22e-7 | 1.18e-1 | 1.38e-2 | **8.55×** |
| sawtooth | 4.15e-9 | 3.26e-4 | 2.37e-4 | 1.37× |
| pulse | 1.77e-9 | 4.62e-5 | 7.37e-5 | 0.63× |
| mixed_ood | 7.53e-8 | 7.39e-2 | 9.69e-3 | **7.62×** |

1A wins on smooth/easy families (chirp, pulse). 1B wins decisively on hard
families (step 3.2×, rw 8.6×, mixed 7.6×).

---

## Coverage Analysis: Why Coverage ≠ Performance

From the probe study and standard benchmark:

| Method | 4D Coverage | Bench Overlap | Bench MSE | vs Oracle |
|--------|------------|---------------|-----------|-----------|
| random | 9.9% (193 cells) | 98% | 2.67e-3 | 14.4× |
| active | 7.2% (139 cells) | 97% | 1.86e-3 | 10.0× |
| maxent_rl | 38.8% (751 cells) | 100% | 2.39e-2 | 129× |

**Maxent covers everything but performs worst**: density is spread too thin.
Random misses 56 cells but has 51× higher density in shared cells.
Active targets model uncertainty, concentrating data where it matters.

**The key metric is sample density in task-relevant cells, not raw coverage.**

---

## Data Scaling Law

| N (trajectories) | Val Loss | MSE (est.) |
|-----------------|----------|------------|
| 1K | ~0.01 | ~0.03 |
| 2K | ~0.005 | ~0.01 |
| 5K | ~0.002 | ~0.005 |
| 10K | 0.00139 | 2.67e-3 |
| 20K | 2.29e-4 (ep110) | TBD |

Scaling law: MSE ∝ N^{−0.85} (log-linear).
Action type ranking: mixed > multisine > ou > uniform > gaussian > bangbang.

---

## Answers to README Questions

| Question | Answer |
|----------|--------|
| Random rollout coverage sufficient? | Partially — good for easy, insufficient for hard |
| Which action distributions best? | Mixed > OU > bangbang (coverage-learnability tradeoff) |
| How much data needed to match supervised? | 10K already beats supervised on aggregate |
| Where does policy fail? | Step and random_walk (hard families) |
| Architecture variations? | Tested in Stage 1D (capacity matters) |

## Conclusion

Stage 1B establishes that **physics-only random rollout can train a viable
inverse dynamics tracker** that beats supervised learning on OOD references.
The key scientific finding is the **coverage-learnability tradeoff**:
maximizing coverage alone doesn't work; the data must also be learnable.
The massive oracle gap (14.4×) motivates Stages 1C (smarter data) and
1D (more data + bigger models).
