# Stage 1C: Entropy-Driven Coverage — Multi-Variant Synthesis

## Executive Summary

Stage 1C explored **five** coverage/diversity strategies from the proposal:

1. **Ensemble disagreement** — select high-uncertainty trajectories
2. **Bin rebalancing** — favor trajectories visiting rare 4D state bins
3. **Low-density / max-entropy proxy** — select trajectories in sparse regions
4. **Hybrid coverage** — z-score combination of density + rebalancing
5. **Hindsight relabeling** — roll out teacher on hard refs, relabel achieved motion

**Core finding: No Stage 1C method consistently beats the matched random baseline.**

The random-rollout baseline remains the strongest overall strategy. Each diversity
method helps on 1–2 specific OOD families but hurts on the critical `mixed_ood`
benchmark. This is a significant negative result — for this system, naive uniform
random data is hard to beat with post-hoc selection.

See `stage_1c_commands.md` for exact reproduction commands.
See `stage_1c_details.md` for per-variant failure-mode discussion.

---

## Experimental Setup

All Stage 1C variants share a matched budget:
- **Seed data:** 2000 random-rollout trajectories (mixed action types)
- **Candidate pool:** 12000 trajectories (except hindsight: 4000 teacher rollouts)
- **Selected:** 8000 trajectories merged with seed = 10000 total
- **Architecture:** MLP 512x4, 100 epochs, batch 65536, lr=1e-3, cosine schedule
- **Baseline:** `outputs/stage1c_random_matched/` (10000 pure random trajectories)
- **Reference:** `outputs/stage1b_scaled/` (old Stage 1B best)

---

## ID Results (Multisine Test Set)

| Method | mean MSE_total | max MSE_total | median MSE_total | Category |
|---|---:|---:|---:|---|
| **random_matched** | **1.185e-4** | 2.156e-3 | 6.182e-5 | Ergodic |
| hybrid_coverage | 1.478e-4 | 2.759e-3 | 6.389e-5 | Mild |
| low_density | 1.532e-4 | 2.928e-3 | 7.221e-5 | Mild |
| disagreement | 1.662e-4 | 3.807e-3 | 8.321e-5 | Mild |
| rebalance_bins | 1.950e-4 | 7.620e-3 | 7.543e-5 | Mild |
| hindsight | 2.894e-4 | 1.816e-2 | 7.970e-5 | Mild |
| **maxent_rl** | **9.788e-3** | 2.573e-1 | 4.579e-3 | Too broad |
| **adversarial_v2** | **2.239e-4** | 7.244e-3 | — | Targeted |
| restricted_v | 2.047e-2 | 1.700 | 8.330e-4 | Too narrow |
| reachability_v5 | 1.205e-1 | 2.326 | 3.355e-2 | No coherence |
| reachability_v15 | 5.114e-1 | — | — | No coherence |

**Three regimes emerge:**
- Mild perturbations from ergodic (1.03–2.44×): tolerable
- Distribution shift (maxent RL 82.6×, restricted-v 172×): catastrophic
- No trajectory coherence (reachability 1,017–4,314×): complete failure

---

## OOD Results (mean MSE_total)

| OOD family | random | disagree | rebalance | density | hybrid | hindsight | stage1b | winner |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| chirp | **2.16e-4** | 3.01e-4 | 2.47e-4 | 2.49e-4 | 2.48e-4 | 3.32e-4 | 3.04e-4 | random |
| step | 3.34e-2 | **3.34e-2** | 3.92e-2 | 3.63e-2 | 3.68e-2 | 6.71e-2 | 4.71e-2 | disagree~random |
| random_walk | 1.74e-2 | 2.33e-2 | 1.60e-2 | 2.03e-2 | 2.44e-2 | 3.59e-2 | **1.38e-2** | stage1b |
| sawtooth | 1.56e-4 | 2.00e-4 | 1.98e-4 | **1.27e-4** | 1.41e-4 | 2.18e-4 | 2.37e-4 | density |
| pulse | 8.26e-5 | 1.23e-4 | 8.48e-5 | 9.52e-5 | 7.89e-5 | **7.28e-5** | 7.37e-5 | hindsight |
| mixed_ood | **4.83e-3** | 1.36e-2 | 2.97e-2 | 1.58e-2 | 1.67e-2 | 3.26e-2 | 9.69e-3 | random |

**Geometric mean across all 6 OOD families:**

| Method | Geomean MSE |
|---|---:|
| **random_matched** | **1.41e-3** |
| stage1b_scaled | 1.79e-3 |
| low_density | 1.81e-3 |
| hybrid_coverage | 1.86e-3 |
| rebalance_bins | 2.06e-3 |
| disagreement | 2.07e-3 |
| hindsight | 2.73e-3 |

---

## Key Research Findings

### 1. The Goldilocks Principle (Main Result)

Random rollouts produce a naturally optimal training distribution. Any deviation
hurts — whether narrowing (restricted-v: 172×) or broadening (maxent RL: 82.6×).

The ergodic measure of random dynamics provides:
- **Sufficient breadth**: 92% state coverage for closed-loop robustness
- **Natural concentration**: capacity focused on dynamically natural states
- **Trajectory coherence**: temporal structure preserved (critical!)

### 2. Max-Entropy RL: More Uniform ≠ Better

Despite achieving 94.6% coverage (vs 92.0%) and H=10.99 nats (vs 8.46),
the max-entropy policy produces data that's **82.6× worse** for tracking.
The model cannot fit the ID function uniformly (train loss 0.43 vs 0.007).

### 3. Trajectory Coherence Is Essential

Reachability-guided single-step data fails catastrophically even with
matched velocity range (1,017×). The training loss converges (0.011) but
val loss monotonically increases (5.98 → 14.06). Single-step patterns
don't transfer to sequential tracking.

### 4. Closed-Loop Error Compounding

Restricted-velocity shows the DAgger (Ross et al. 2011) phenomenon:
the model's errors change which states are visited, requiring robustness
to states OUTSIDE the test distribution. P99 error ratio grows to 154×.

---

## Proposal Checklist

| Approach | Status | Outcome |
|---|---|---|
| State-space binning with rebalancing | Done | Negative (1.64×) |
| Maximum entropy / low-density scoring | Done | Negative (1.29×) |
| **Max-entropy RL exploration** | **Done** | **Negative (82.6×)** |
| Curiosity / ensemble disagreement | Done | Negative (1.40×) |
| Hindsight relabeling | Done | Negative (2.44×) |
| Adversarial reference generation | Done | Negative (1.89×) |
| Restricted-velocity baseline | Done | Catastrophic (172×) |
| Reachability-guided (v=±15) | Done | Catastrophic (4,314×) |
| Reachability-guided (v=±5) | Done | Catastrophic (1,017×) |

---

## Artifacts

| Directory | Description |
|---|---|
| `outputs/stage1c_random_matched/` | Matched random baseline (BEST) |
| `outputs/stage1c_active_full/` | Disagreement active collection |
| `outputs/stage1c_rebalance_full/` | Bin rebalancing |
| `outputs/stage1c_density_full/` | Low-density selector |
| `outputs/stage1c_hybrid_full/` | Hybrid coverage |
| `outputs/stage1c_hindsight_full/` | Hindsight relabeling |
| `outputs/stage1c_maxent_rl/` | Max-entropy RL exploration |
| `outputs/stage1c_adversarial_full/` | Adversarial reference generation |
| `outputs/stage1c_restricted_v/` | Restricted-velocity baseline |
| `outputs/stage1d_reachability_5M/` | Reachability v=±15 |
| `outputs/stage1d_reachability_5M_v5/` | Reachability v=±5 |
| `outputs/stage1c_ood_*` | OOD comparison runs |

---

## Next Steps: Stage 1D (Revised)

Given the Goldilocks principle, Stage 1D should NOT pursue:
- ❌ Uniform coverage (maxent RL already proven suboptimal)
- ❌ Single-step reachability data (no trajectory coherence)

Instead, Stage 1D should explore:
- **DAgger-style iterative refinement**: Train on random → track → collect
  error states → augment → retrain. Directly addresses distributional shift.
- **Trajectory optimization**: iLQR/CEM generates coherent trajectories
  that explore specific state-space regions as needed.
- **Curriculum learning**: Start with easy references, progressively
  increase difficulty to build robustness incrementally.

Best model to carry forward: `outputs/stage1c_random_matched/best_model.pt`
