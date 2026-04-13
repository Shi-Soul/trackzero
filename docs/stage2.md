# Stage 2: Noise Robustness and Degradation Analysis

## Research Question

How does the TRACK-ZERO MLP policy behave when reference trajectories
are imperfect? Specifically:
1. Does the learned policy degrade gracefully under noisy references?
2. How does degradation compare to the analytical oracle?
3. Can explicit noise augmentation during training improve robustness?

## Experimental Setup

**System**: double pendulum (2 DOF), same as Stage 1.
**Best Stage 1 model**: 1024×6 MLP, cosine LR + WD=1e-4, trained on 10K random rollouts.
**Benchmark**: 6-family standard benchmark (multisine, chirp, step, random_walk, sawtooth, pulse).

### 2A: Degradation Under Gaussian Reference Noise

We corrupt reference trajectories at evaluation time by adding
Gaussian noise to joint positions (σ) and velocities (5σ).
Seven noise levels: σ ∈ {0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5}.
Three benchmark families (multisine, step, random_walk) × 100 episodes each.

Both the MLP policy and the analytical oracle (mj_inverse) are evaluated
under identical noisy conditions.

### 2C: Noise-Augmented Training

Train a new model on the same 10K random rollout data, but corrupt 50%
of training pairs by adding N(0, σ=0.05) to reference positions and
N(0, 5σ=0.25) to reference velocities. The target action remains the
CLEAN (uncorrupted) action, teaching the policy to be robust.

Same architecture and training recipe as Stage 1 best (1024×6, cosine+WD).

## Results

### Finding 1: MLP Degrades Slower Than Oracle

| σ | MLP Tracking MSE | Oracle Tracking MSE | MLP/Oracle |
|---|-----------------|--------------------:|------------|
| 0.00 | 6.80e-4 | 7.63e-5 | 8.9× (MLP worse) |
| 0.01 | 1.37e-2 | 6.99e-2 | **0.20× (MLP 5× better)** |
| 0.02 | 8.44e-2 | 2.22e-1 | 0.38× |
| 0.05 | 2.66e-1 | 6.40e-1 | 0.42× |
| 0.10 | 5.76e-1 | 1.08 | 0.53× |
| 0.20 | 9.63e-1 | 1.43 | 0.67× |
| 0.50 | 1.86 | 2.34 | 0.79× |

**Crossover at σ ≈ 0.005**: below this, oracle wins (exact dynamics);
above this, MLP wins by a widening margin.

**Mechanism**: The oracle computes exact inverse dynamics for the given
(noisy) state and reference. When the reference is inconsistent
(corrupted position without consistent velocity), the oracle's exact
computation amplifies noise into extreme torques. The MLP's smooth
function approximation acts as an implicit low-pass filter.

### Per-Family Breakdown (σ=0 → σ=0.01)

| Family | MLP 0→0.01 | Oracle 0→0.01 | MLP advantage at σ=0.01 |
|--------|-----------|--------------|-------------------------|
| multisine | 1.07e-5 → 3.07e-3 (287×) | 9.18e-5 → 8.78e-3 (96×) | 2.9× |
| step | 8.15e-4 → 9.72e-3 (12×) | 8.63e-5 → 6.94e-2 (804×) | **7.1×** |
| random_walk | 1.22e-3 → 2.83e-2 (23×) | 5.06e-5 → 1.31e-1 (2596×) | **4.6×** |

Step and random_walk families show the strongest MLP advantage: the oracle's
degradation is catastrophic (800× and 2600× worse), while the MLP degrades
only 12× and 23× respectively.

### Finding 2: Noise Augmentation Trades Precision for Robustness

| Method | Clean (σ=0) | σ=0.05 | σ=0.10 | σ=0.20 |
|--------|-------------|--------|--------|--------|
| Standard MLP | 6.80e-4 | 2.66e-1 | 5.76e-1 | 9.63e-1 |
| Noise-aug MLP | 5.29e-2 | 5.90e-2 | 7.67e-2 | 2.04e-1 |
| Oracle | 7.63e-5 | 6.40e-1 | 1.08 | 1.43 |

The noise-augmented model is **78× worse at clean tracking** (5.29e-2 vs 6.80e-4)
but **4.5× better under σ=0.05 noise** (5.90e-2 vs 2.66e-1).

Key observation: the noise-augmented model's degradation curve is nearly flat
from σ=0 to σ=0.10 (1.45× increase), while the standard model degrades 847×
over the same range. This suggests the noise-augmented model learned a
different strategy — prioritizing robustness over precision.

## Finding 3: Residual (Oracle-Informed) Architecture

A separate experiment tested feeding oracle torque as additional MLP input:
input = [state(4), ref_next(4), oracle_torque(2)] = 10D.

| Method | Agg MSE | vs Stage 1 Best | vs Oracle |
|--------|---------|-----------------|-----------|
| Stage 1 best MLP | 4.19e-4 | 1.0× | 5.5× worse |
| Oracle (mj_inverse) | 7.63e-5 | — | 1.0× |
| Residual MLP | **1.95e-5** | **21.5× better** | **3.9× better** |

The residual MLP outperforms even the oracle, suggesting it learns to
correct the oracle's discretization artifacts. However, this requires
oracle computation at inference time (impractical for complex systems).

## Interpretation

1. **The MLP learns more than point-wise inverse dynamics.** Its smooth
   function approximation provides natural noise filtering — an emergent
   property not explicitly trained for.

2. **Noise augmentation is effective but blunt.** It shifts the operating
   point from "precise but fragile" to "approximate but robust." An
   adaptive approach (choosing noise level based on application) would
   be more practical.

3. **The residual architecture confirms learnability.** The 21.5×
   improvement shows the remaining oracle gap is learnable given
   richer input features. The challenge is achieving this without
   oracle access at inference.

## Implications for Later Stages

- For Stage 3 (scaling): the MLP's implicit robustness may be more
  important in higher-DOF systems where reference quality is lower.
- For Stage 4 (humanoid): retargeted motion capture references will
  inevitably be imperfect, so noise robustness is a practical advantage.
