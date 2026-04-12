# OOD Robustness Analysis — Key Findings

## 1. The Degradation Ratio Discovery

We evaluated 5 trained models on 6 settings: 1 in-distribution (ID) + 5 out-of-distribution (OOD).
The **degradation ratio** = OOD_MSE / ID_MSE measures how much a model's performance degrades 
when the reference trajectory distribution shifts.

### Absolute Performance (MSE, lower is better)

| Method | ID multisine | chirp | step | random_walk | sawtooth | pulse |
|--------|-------------|-------|------|-------------|----------|-------|
| random_512x4 | **1.19e-4** | 2.16e-4 | **3.34e-2** | **1.74e-2** | 1.56e-4 | 8.26e-5 |
| maxent_512x4 | 9.79e-3 | 9.95e-3 | 9.53e-2 | 7.48e-2 | 8.15e-3 | 7.08e-3 |
| supervised | 1.22e-4 | **1.80e-4** | 1.50e-1 | 1.18e-1 | 3.26e-4 | **4.62e-5** |
| hybrid | 1.48e-4 | 2.48e-4 | 3.68e-2 | 2.44e-2 | **1.41e-4** | 7.89e-5 |
| adversarial | 2.24e-4 | 2.66e-4 | 4.96e-2 | 2.50e-2 | 2.35e-4 | 1.93e-4 |

### Degradation Ratios (OOD_MSE / ID_MSE)

| Method | step/ID | random_walk/ID | chirp/ID | sawtooth/ID | pulse/ID |
|--------|---------|----------------|----------|-------------|----------|
| **maxent_rl** | **9.7×** | **7.6×** | **1.0×** | 0.8× | 0.7× |
| random | 282× | 147× | 1.8× | 1.3× | 0.7× |
| hybrid | 249× | 165× | 1.7× | 1.0× | 0.5× |
| adversarial | 222× | 112× | 1.2× | 1.1× | 0.9× |
| supervised | 1235× | 970× | 1.5× | 2.7× | 0.4× |

### Key Insight

**Maxent RL is 29× more robust to hard distribution shift than the random baseline.**

On the most challenging OOD settings (step functions, random walks):
- Maxent degrades only **9.7×** from ID to step
- Random degrades **282×** — a 29× difference in robustness
- Supervised catastrophically fails: **1235×** degradation

## 2. OOD Difficulty Taxonomy

The 5 OOD types fall into two distinct categories:

### "Easy" OOD (< 3× degradation for good methods)
- **chirp**: Frequency-swept sinusoids — similar to multisine, just different frequencies
- **sawtooth**: Linear ramps — smooth, predictable
- **pulse**: Step-like but brief — transient response

### "Hard" OOD (100-1000× degradation)
- **step**: Sustained constant-torque changes — drives system to equilibria far from training
- **random_walk**: Cumulative random changes — explores wide state space over time

The hard OOD types force the system into **novel configurations** (extreme angles, high velocities)
that the training distribution may not cover.

## 3. Why Maxent Is Robust but Inaccurate

### The coverage-accuracy tradeoff

| Property | Random | Maxent |
|----------|--------|--------|
| State range q | ±15° | ±60° |
| State range v | ±23 rad/s | ±43 rad/s |
| Coverage volume | 1× | ~4× |
| Effective samples/region | 5M/V | 5M/4V ≈ 1.25M/V |
| ID accuracy | 1.19e-4 | 9.79e-3 |
| Hard OOD robustness | 282× degradation | 9.7× degradation |

Maxent explores 4× the state space with the same 5M samples, meaning **4× fewer samples 
per unit volume**. This hurts ID accuracy but provides coverage where step/random_walk 
trajectories actually go.

### The theoretical bound

If maxent could match random's ID accuracy while keeping its low degradation ratio:
- Predicted step MSE: 1.19e-4 × 9.7 = **1.15e-3** (vs random's 3.34e-2)
- Predicted rw MSE: 1.19e-4 × 7.6 = **9.06e-4** (vs random's 1.74e-2)
- **29× better on step, 19× better on random_walk**

## 4. Hypothesis: Hybrid Training

The degradation ratio analysis suggests a clear research direction: 
**combine maxent's broad coverage with random's efficient ID learning**.

### Strategies Under Test

1. **Concat**: 10M pairs (5M random + 5M maxent), standard training
2. **Curriculum**: Pretrain on maxent (broad features), finetune on random (ID accuracy)
3. **Reverse Curriculum**: Pretrain on random (ID accuracy), finetune on combined (add OOD)
4. **Weighted Mix**: Emphasize random data while including some maxent for coverage

### Expected Outcomes

The best strategy should achieve:
- ID accuracy close to random baseline (< 2× degradation from random)
- OOD robustness inheriting maxent's low degradation ratio (< 20×)

## 5. Why Previous "Hybrid" Didn't Work

The Stage 1C "hybrid" method selected a subset from the random data pool, enriched with 
high-value states. It did NOT include actual maxent RL data. Its degradation ratio (249×) 
is nearly identical to pure random (282×), confirming that subset selection from a narrow 
distribution cannot provide true OOD coverage.

The new hybrid strategies use the full maxent dataset, which covers fundamentally different 
state-space regions.

---

*Analysis based on outputs/ood_benchmark_existing.json*
*Generated during HP sweep experiments (8 GPUs, ongoing)*
