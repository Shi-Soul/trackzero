# Stage 1C Deep Analysis: Why Is Random Hard to Beat?

## First Principles Decomposition

### The Core Contradiction

We want to learn `f(s_t, s_{t+1}) → u_t` (inverse dynamics).
The proposal hypothesizes that *better state-space coverage → better ID model*.
Yet every Stage 1C method fails to beat random rollouts.

**Why?** Three competing hypotheses below.

---

## Hypothesis 1: The ID Function Is Too Smooth

The double pendulum ID function is:

```
u = M(q)·q̈ + C(q,q̇)·q̇ + g(q) + D·q̇
```

This is a composition of sinusoidal terms (through `cos(q2)` in M,
`sin(q1)`, `sin(q1+q2)` in gravity). It has **bounded, continuous
derivatives everywhere**. An MLP with 512×4 neurons has far more
capacity than needed.

**Prediction**: If true, ANY covering distribution works equally well.
Coverage optimization is pointless.

**Evidence for**: Mass matrix condition number varies only 2.65–44.98.
Determinant ratio is only 2.3×. The function is globally smooth.

**Evidence against**: Stage 1B showed that data SIZE matters (more data
→ better up to a point). This means the function is not trivially
learnable — optimization landscape matters.

---

## Hypothesis 2: Sensitivity Is Non-Uniform

We measured the ID function's Jacobian norm across 6 state-space regions:

| Region | Sensitivity | cond(M) |
|---|---:|---:|
| fast_spin (v=10,10) | **48.2** | 45.3 |
| origin (q=0, v=0) | 13.6 | 45.3 |
| hanging (q=π, v=0) | 13.6 | 45.3 |
| extreme (q=π/2, v=±10) | 10.7 | 6.9 |
| inverted (q=0,π) | 9.0 | 2.8 |
| double_inverted | 9.0 | 2.8 |

**5.4× sensitivity ratio** between hardest and easiest regions.
The Coriolis term `C(q,q̇)·q̇` grows quadratically with velocity,
making high-velocity states the hardest to approximate.

**Key finding**: Random rollouts spend only **6.4%** of time in the
high-sensitivity region (|v| > 10). Velocity marginal entropy is
2.12 vs uniform 3.00 — significantly biased toward low velocities.

**Prediction**: Methods that increase high-velocity data should help.
Max-entropy RL (uniform state coverage) would give ~5× more high-v data.

---

## Hypothesis 3: State Coverage ≠ Transition Coverage

The ID model input is a TRANSITION `(s_t, s_{t+1})`, not a single state.
For fixed `s_t`, the reachable `s_{t+1}` under `u ∈ [-τ,τ]²` forms a
2D parallelogram in 4D state space (since `q̈ = M⁻¹(u - bias)`).

Random actions sample **uniformly** within this parallelogram.
A max-entropy STATE policy might concentrate on specific actions that
push toward unvisited states → potentially LESS transition diversity
within each state.

**Evidence**: Reachable acceleration volume varies only 2.3× across
state space. Transition diversity is fairly uniform — hard to improve.

---

## Empirical Test: Coverage vs Entropy

GPU measurements (20-bin 4D histogram, same step budget ~200M):

| Metric | Random | Max-Entropy RL |
|---|---:|---:|
| Coverage (% bins occupied) | ~91.9% | ~91.3% |
| Histogram entropy (nats) | 8.46 | **10.90** |
| Entropy gap | — | **+2.44 nats** |

**Striking**: Random and max-entropy achieve nearly IDENTICAL coverage!
Both fill ~91% of bins. But max-entropy distributes visits **~10× more
uniformly** (2.44 nats difference ≈ e^2.44 ≈ 11.5× more uniform).

This means the question reduces to: **does visit UNIFORMITY matter
when coverage is already saturated?**

---

## The Missing Piece: Test Distribution Analysis

### Test set covers only 16% of state space!

| Distribution | Coverage | Entropy | frac(|v|>10) |
|---|---:|---:|---:|
| Test set (multisine) | **16.0%** | 8.61 | 1.09% |
| Random rollouts | 92.0% | 8.46 | 6.43% |
| Max-entropy RL | ~92% | 10.90 | ~25% (uniform) |

The test trajectories stay in a SMALL region of state space:
- **Velocities < 10 rad/s** for 99% of time
- **Position range** concentrated near origin
- Entropy 8.61 nats vs max possible 11.98

### Resolution of the 1C Puzzle

**Coverage methods push data AWAY from the test-relevant region.**

Random rollouts already cover the test-relevant 16% well — since both
use the same physics and moderate velocities dominate. Diversity methods
redistribute data toward the IRRELEVANT 76% of covered bins, effectively
DILUTING model capacity for the region that matters at test time.

This is the **Coverage Dilution Hypothesis**:
- More uniform coverage → less data per bin in the test-relevant region
- MLP has finite capacity → wasted on never-tested states
- Result: random ≈ best for the test distribution

### Implication for Research Direction

This means 1C's premise ("maximize coverage") is wrong for this system.
The correct objective is not `max H[p(s)]` but rather:

```
min_{p_train} E_{p_test}[ error(f, s, s') ]
```

This points toward:
1. **Test-distribution-aware training** (but test dist is unknown)
2. **Adversarial/curriculum** (find hard cases WITHIN test region)
3. **Robust optimization** (minimax over plausible test distributions)

---

## CRITICAL UPDATE: Restricted-Velocity Experiment REFUTES Dilution Hypothesis

### Experimental Design

Direct test of Coverage Dilution Hypothesis: if spreading data into
irrelevant state regions hurts, then **concentrating data in the test-relevant
region should HELP**.

- **Restricted-v**: 10k trajectories with |v| < 10 filter, action ∈ [-3,3]²
- Result: mean velocity magnitude 2.17, only 0.12% time at |v| > 10
- This EXACTLY matches the test distribution profile

### Result: Catastrophic Failure (172× worse)

| Metric | Restricted-v | Random | Ratio |
|---|---:|---:|---:|
| Mean MSE | **2.05e-2** | 1.19e-4 | **172.7×** |
| Median MSE | 8.33e-4 | 6.18e-5 | 13.5× |
| Max MSE | **1.70** | 2.16e-3 | 788.5× |
| Mean max_q_err | 0.126 rad | 0.017 rad | 7.2× |

### Percentile Scaling Reveals Error Compounding

| Percentile | Restricted-v | Random | Ratio |
|---|---:|---:|---:|
| P25 | 3.27e-4 | 3.80e-5 | 8.6× |
| P50 | 8.33e-4 | 6.18e-5 | 13.5× |
| P75 | 2.84e-3 | 1.24e-4 | 22.8× |
| P90 | 9.08e-3 | 2.64e-4 | 34.4× |
| P95 | 1.54e-2 | 3.67e-4 | 42.1× |
| P99 | 9.75e-2 | 6.32e-4 | **154.4×** |

The ratio grows from 8.6× at P25 to 154× at P99. This is the classic
signature of **closed-loop error compounding**: once the model makes a
mistake that pushes the system outside its training distribution, it
has ZERO recovery capability, leading to catastrophic divergence.

### Catastrophic failure distribution

- 46% of trajectories have MSE > 0.001 (vs 1% for random)
- 10% have MSE > 0.01 (vs 0% for random)
- 6 trajectories diverge past 0.5 rad position error (vs 0 for random)
- 3 trajectories completely fail with > 1.0 rad error (vs 0 for random)

### Why Concentration Fails: Closed-Loop Distributional Shift

The restricted-v model learns the low-velocity ID function perfectly
(train loss 0.007 — much lower than random baseline's train loss).
But in closed-loop tracking:

1. Small torque prediction errors cause velocity overshoots
2. System briefly enters |v| > 10 states during correction
3. Model has NO training data in this region → garbage predictions
4. Garbage predictions compound → further deviation → catastrophe

This is directly analogous to **DAgger** (Ross et al., 2011):
the observation distribution under the learned policy DIFFERS from the
training distribution precisely because imperfect predictions change
what states are visited.

### Revised Understanding

The Coverage Dilution Hypothesis is **REFUTED**. The correct explanation:

> **Robustness Requirement**: Closed-loop control demands that the ID
> model handle states arising from its own prediction errors, not just
> states in the test distribution. Broad coverage provides this robustness.
> Random rollouts are near-optimal because they cover a naturally wide
> range while concentrating in dynamically natural regions.

This resolves the 1C paradox:
- Coverage methods don't HURT (methods are 1-2× worse, not 172×)
- Coverage methods don't HELP (random already provides sufficient breadth)
- Concentration KILLS (restricting distribution causes catastrophic failure)

**Random rollouts hit a Goldilocks zone**: broad enough for robustness,
concentrated enough in natural dynamics to be sample-efficient.

See `stage_1c_maxent_results.md` for max-entropy RL results (pending).
