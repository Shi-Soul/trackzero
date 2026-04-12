# Stage 1C: Exploration Method Comparison

## Research Question

Can a more principled exploration strategy outperform random-torque
rollouts on the standard benchmark? (README Stage 1C)

## Hypothesis

The random baseline is a "dumb" explorer — it covers the state space
uniformly but without any intelligence. We hypothesize that at least
one of the following strategies can improve on it:
- **Active selection**: discard low-information trajectories
- **Density-based rebalancing**: oversample underrepresented state regions
- **Maximum-entropy RL**: learn an exploration policy that maximizes
  state-space entropy
- **DAgger**: iteratively collect data at the states the policy
  actually visits

## Experimental Design

All methods use the same total data budget (10K trajectories × 500
steps = 5M state-action pairs), the same architecture (MLP 512×4,
3.1M params), and the same training protocol (Adam, lr=1e-3, 30
epochs, batch 65536). Evaluation uses the standard benchmark (see
Stage 0). Each method is trained once with fixed seed; cross-seed
variance is not yet measured (see Limitations).

**Methods tested (9 TRACK-ZERO + 1 supervised baseline):**

| Method | Data source | Key mechanism |
|--------|-----------|---------------|
| random | Uniform random torques | No selection |
| active | Random torques → keep high-variance trajectories | Variance-based filtering |
| density | Random torques → reject from high-density bins | State-space rebalancing |
| rebalance | Random torques → weight loss by inverse density | Loss reweighting |
| hybrid_select | Random × multisine → select diverse subset | Cross-source selection |
| hybrid_weighted | Random × multisine → density-weighted loss | Cross-source weighting |
| hybrid_curriculum | Multisine → random curriculum | Staged introduction |
| adversarial | Maximize model error regions | Error-seeking exploration |
| hindsight | Random rollouts → relabel as tracking data | Hindsight relabeling |
| maxent_rl | SAC-style entropy maximization | RL-based exploration |

## Results

| Rank | Method | Aggregate MSE | ×Oracle | vs Random |
|------|--------|-------------|---------|-----------|
| 1 | active | 1.86e-3 | 10.0 | 1.44× better |
| 2 | hybrid_select | 2.15e-3 | 11.7 | 1.24× better |
| 3 | rebalance | 2.18e-3 | 11.8 | 1.23× better |
| 4 | random | 2.67e-3 | 14.4 | — |
| 5 | density | 3.17e-3 | 17.1 | 0.84× |
| 6 | hybrid_weighted | 4.49e-3 | 24.3 | 0.59× |
| 7 | adversarial | 5.01e-3 | 27.1 | 0.53× |
| 8 | hindsight | 8.13e-3 | 44.0 | 0.33× |
| 9 | maxent_rl | 2.39e-2 | 129 | 0.11× |
| — | supervised_1a | 3.52e-2 | 190 | 0.08× |
| — | dagger_512×4 | 6.09e-1 | 3294 | 0.004× |
| — | dagger_1024×4 | 6.22e-1 | 3361 | 0.004× |

**Per-family breakdown (top 4 + key failures):**

| Method | multisine | chirp | sawtooth | pulse | step | random_walk |
|--------|-----------|-------|----------|-------|------|-------------|
| *Oracle* | *1.01e-4* | *3.16e-4* | *1.93e-4* | *1.29e-4* | *3.25e-4* | *4.49e-5* |
| active | 1.71e-4 | 3.11e-4 | 1.20e-4 | 1.40e-4 | 7.64e-3 | 2.76e-3 |
| random | 1.22e-4 | 2.49e-4 | 1.03e-4 | 8.74e-5 | 9.49e-3 | 5.95e-3 |
| maxent_rl | 8.39e-3 | 9.52e-3 | 7.82e-3 | 7.13e-3 | 5.67e-2 | 5.38e-2 |
| supervised | 1.90e-4 | 2.08e-4 | 2.99e-4 | 5.98e-5 | 1.18e-1 | 9.23e-2 |

## Analysis

**Finding 1: Selection provides modest gains.** The best method
(active) is only 1.44× better than random. All top methods use the
same data source (random torques) and differ only in which subset
they keep or how they weight the loss. The shared performance plateau
suggests that the quality of any 10K random-torque dataset is already
high, and selection can only extract incremental improvements.

**Finding 2: The performance bottleneck is step and random_walk.**
Across all methods, the aggregate MSE is dominated by these two
families. Even the best method (active) achieves only 23.5× and
61.6× oracle on step and random_walk respectively, while all smooth
families are within 2× oracle. The remaining 10× aggregate gap is
almost entirely attributable to high-velocity tail states.

**Finding 3: More coverage ≠ better performance.** Maxent_rl achieves
the highest state-space coverage of all methods but ranks 9th. Its
RL-trained exploration policy visits extreme states that rarely appear
in the benchmark trajectories, wasting network capacity on irrelevant
regions. This directly contradicts the naive hypothesis that "more
coverage is always better."

**Finding 4: DAgger catastrophically fails.** DAgger's compounding
error makes it unusable for this problem: iteration 0 uses the random
policy to collect data, but the resulting model is so poor that
iteration 1 collects data from catastrophically bad trajectories.

## Conclusion

At fixed data budget (10K trajectories), no exploration method
achieves more than 1.5× improvement over random on the standard
benchmark. The dominant factor in performance is not how the data is
collected, but how much of the reachable state space it covers at
sufficient density. This motivates Stage 1D: scaling the dataset
size to close the remaining 10× gap to oracle.