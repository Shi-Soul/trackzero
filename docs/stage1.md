# Stage 1: Learning Inverse Dynamics From Physics Alone

## Research Question

Can a neural network trained exclusively on physics-simulator rollouts
(no human demonstrations, no task-specific references) learn to track
arbitrary reference trajectories? What are the dominant factors that
determine tracking quality?

**System**: double pendulum (2 DOF, 4D state, 2D action, torque-limited).
**Benchmark**: 6 families × 100 trajectories (multisine, chirp, step,
random_walk, sawtooth, pulse). **Oracle**: MuJoCo `mj_inverse` with
finite-difference acceleration.

---

## 1A: Supervised Baseline

**Setup**: 10K multisine reference trajectories, MLP 512×4 (3.1M params).

| Family | MSE | ×Oracle |
|--------|-----|---------|
| multisine | 1.90e-4 | 1.9 |
| step | 1.18e-1 | 364 |
| random_walk | 9.23e-2 | 2056 |
| **Aggregate** | **3.52e-2** | **190** |

Near-oracle on families resembling training data (smooth oscillatory).
Catastrophic failure on step/random_walk: the multisine training
distribution has zero coverage of the sustained high-velocity states
these families require.

**Conclusion**: Training on one reference family cannot generalize.
Universal tracking requires covering the full reachable state space.

---

## 1B: Random Rollout Baseline

**Setup**: 10K random-torque rollouts (uniform u ~ [-τ_max, τ_max]
at each timestep). Same 512×4 architecture.

| Family | MSE | ×Oracle | vs Supervised |
|--------|-----|---------|---------------|
| multisine | 3.83e-3 | 37.9 | 20× worse |
| step | 4.59e-3 | 14.1 | **25.7× better** |
| random_walk | 2.70e-3 | 60.1 | **34.2× better** |
| **Aggregate** | **2.67e-3** | **14.4** | **13.2× better** |

**Finding**: Random rollouts achieve 13× better aggregate tracking
than supervised learning, despite never seeing any reference trajectory
during training. Coverage of the reachable state space (24.5% bin
occupancy vs 1.3% for multisine) explains the generalization.

**Tradeoff**: Random data trades per-family precision (20× worse on
multisine) for universal coverage (25× better on step). With a fixed
data budget, spreading data uniformly reduces local density.

---

## 1C: Exploration Method Comparison

**Question**: Can smarter exploration beat random rollouts at fixed
10K budget?

**Methods tested** (9 TRACK-ZERO + DAgger + supervised):

| Rank | Method | AGG MSE | ×Oracle | vs Random |
|------|--------|---------|---------|-----------|
| 1 | active (variance-based selection) | 1.86e-3 | 10.0 | 1.44× |
| 2 | hybrid_select (cross-source) | 2.15e-3 | 11.7 | 1.24× |
| 3 | rebalance (density-weighted loss) | 2.18e-3 | 11.8 | 1.23× |
| 4 | random (baseline) | 2.67e-3 | 14.4 | 1.00× |
| 5 | density (bin rejection) | 3.17e-3 | 17.1 | 0.84× |
| 9 | maxent_rl (SAC entropy) | 2.39e-2 | 129 | 0.11× |
| — | supervised | 3.52e-2 | 190 | 0.08× |
| — | DAgger | 6.09e-1 | 3294 | 0.004× |

**Finding 1**: All methods cluster within 1.5× of random. The quality
of any 10K random-torque dataset is already high; selection provides
only marginal improvement.

**Finding 2**: Maximum-entropy RL achieves highest state-space coverage
but ranks 9th. More coverage ≠ better performance — covering irrelevant
extreme states wastes network capacity.

**Finding 3**: DAgger catastrophically fails. Policy errors compound:
small torque errors → state divergence → data in unrecoverable regions.
Structurally incompatible with inverse dynamics learning.

**Conclusion**: At 10K budget, the exploration strategy bottleneck is
resolved. The remaining 10× oracle gap requires either more data or
better learning algorithms.

---

## 1D: Training Optimization

**Question**: Is the bottleneck data collection or training recipe?

### Data Engineering (all fail)

| Method | AGG | ×Baseline | Finding |
|--------|-----|-----------|---------|
| mixed random (baseline) | 7.87e-4 | 1.00× | Reference |
| wider initial velocity | 9.04e-4 | 1.15× | No effect |
| bangbang torques | 6.35e-2 | 80.7× | Catastrophic |
| max-coverage selection | 9.88e-4 | 1.26× | No gain |
| coverage-weighted loss | 1.19e-3 | 1.51× | Harmful |

Bangbang (±τ_max only) achieves 4× higher velocity coverage but
catastrophically fails because it lacks action diversity.

### Architecture (bigger is better)

| Architecture | Params | AGG | ×Baseline |
|---|---|---|---|
| 1024×6 | 5.3M | 7.87e-4 | 1.00× |
| 512×4 | 794K | 1.02e-3 | 1.29× |
| 256×3 | 134K | 2.44e-3 | 3.10× |

Per-family nuance: step benefits from smaller models (regularization),
random_walk benefits from larger (capacity for chaotic dynamics).

### Training Recipe (dominant lever)

| Innovation | AGG | ×Base | Key effect |
|---|---|---|---|
| baseline (Adam lr=3e-4) | 7.87e-4 | 1.00× | — |
| cosine LR | 5.64e-4 | 0.72× | random_walk −47% |
| weight decay (1e-4) | 7.17e-4 | 0.91× | step −34% |
| **cosine + WD** | **4.19e-4** | **0.53×** | **Both improved** |
| Huber loss | 1.01e-3 | 1.29× | No benefit |
| dropout (0.1) | 1.97e-3 | 2.50× | Harmful |

**Key finding**: Cosine LR + weight decay = 47% improvement (synergistic).
Cosine annealing helps random_walk (fine-tuning at low LR), WD helps
step (regularization prevents overfitting). Neither alone achieves both.

Cosine only helps large models — 512×4+cosine is 15% worse than without.

---

## Stage 1 Summary

| Axis | Best Method | Effect | Verdict |
|------|------------|--------|---------|
| Data source | Random rollout | 13× vs supervised | ✅ Core thesis |
| Exploration strategy | Active | <1.5× vs random | ❌ Not bottleneck |
| Data engineering | — | 0% gain | ❌ All fail |
| Architecture | 1024×6 | 3× vs 256×3 | ✅ Bigger better |
| **Training recipe** | **Cosine + WD** | **47% gain** | **✅ Dominant** |
| Data scaling | 20K | 59% gain | ✅ More data helps |

**Best Stage 1 result**: 1024×6, cosine LR, WD=1e-4, 10K data.
AGG = 4.19e-4 (11,277× oracle). Per-family: step=1.80e-3,
random_walk=6.45e-4, smooth families ≈ 1e-5.

**Open question**: The 11,000× oracle gap is dominated by step and
random_walk. Training optimization closes 47% (log-space) but cannot
eliminate it. Closing further requires architectural innovation
(Stage 2) or fundamentally different learning approaches.
