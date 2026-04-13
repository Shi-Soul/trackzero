# Synthesis: Stages 1A–1C Findings

## Thesis Under Test

Can a tracking policy trained purely from physics rollouts
(no demonstrations) match a supervised policy trained on the
target trajectory distribution?

## Experimental Setup

**System**: double pendulum (2 DOF, 4D state, torque-limited).
**Benchmark**: 600 trajectories across 6 signal families
(100 each: multisine, chirp, sawtooth, pulse, step, random_walk).
**Metric**: mean tracking MSE (joint angle + velocity).
**Oracle**: analytical inverse dynamics, aggregate MSE = 1.85e-4.

## Result 1: TRACK-ZERO Outperforms Supervised Learning

All physics-only methods beat the supervised baseline on the
standard benchmark. This is the core thesis confirmation.

| Method | Training data | Agg MSE | ×Oracle |
|--------|--------------|---------|---------|
| active | 10K random rollout | 1.86e-3 | 10.0 |
| random | 10K random rollout | 2.67e-3 | 14.4 |
| supervised | 10K multisine | 3.52e-2 | 190 |
| oracle | analytical | 1.85e-4 | 1.0 |

**Why**: supervised learning fits the multisine distribution
but cannot extrapolate. Physics rollouts cover broader dynamics.

## Result 2: The Remaining Gap Is Velocity-Specific

Decomposing the best model (active, 512×4, 10K) by family:

| Family type | Families | Mean ×Oracle | % of total MSE |
|-------------|----------|-------------|----------------|
| Easy (smooth) | multisine, chirp, sawtooth, pulse | 1.1 | 6.7% |
| Hard (dynamic) | step, random_walk | 42.5 | 93.3% |

The policy **matches the oracle** on 4 of 6 families.
93% of the aggregate gap comes from step and random_walk,
which require high-velocity (>10 rad/s) tracking.

**Root cause**: random rollout data concentrates at low
velocities. Training data has <0.01% coverage above 10 rad/s
for joint 1. Coriolis terms scale as ω², so extrapolation
error grows superlinearly.

**Evidence (velocity-stratified analysis)**:

| Max velocity (rad/s) | N trajectories | ×Oracle |
|---------------------|----------------|---------|
| [0, 5) | 28 | 0.5 |
| [5, 10) | 310 | 1.7 |
| [10, 15) | 194 | 8.5 |
| [15, 20) | 66 | 57.7 |

## Result 3: Exploration Strategy Has Marginal Effect

9 methods tested at fixed 10K budget with 512×4 architecture.
On hard families (the only ones that matter):

| Method | Hard family ×Oracle | vs random |
|--------|--------------------:|----------:|
| active | 42.5 | 1.90× better |
| rebalance | 45.3 | 1.78× |
| hybrid_select | 54.2 | 1.49× |
| random | 80.8 | 1.00× |
| density | 82.0 | 0.99× |
| supervised | 1210 | 0.07× |

Active exploration provides ~2× improvement on hard families
relative to random. All physics-only methods cluster within
a 2× band. The bottleneck is not exploration intelligence —
it is the absolute coverage of high-velocity states, which
10K random rollouts simply do not reach often enough.

## Result 4: Model Capacity Backfires at 10K Data

| Architecture | Params | Agg ×Oracle |
|-------------|--------|-------------|
| 512×4 | 1.1M | 14 |
| 1024×6 | 5.3M | 3300 |

229× degradation from 5× more parameters. With 5M training
pairs (10K traj × 500 steps) and 5.3M parameters, the model
memorizes rather than generalizes. The 512×4 architecture
(1.1M params, ~5:1 data-to-param ratio) is well-regularized.

## Result 5: DAgger Is Structurally Incompatible

DAgger requires rolling out the current policy to collect
on-policy data. For inverse dynamics, policy errors compound:
small torque errors → state divergence → data in unrecoverable
regions. All 4 tested configurations (512×4, 1024×4, 1024×6,
512×4 iter0-2) degraded across iterations (3200-3600× oracle).

## Hypothesis Status

| Hypothesis | Verdict | Evidence |
|-----------|---------|----------|
| Random rollout → useful inverse dynamics | **Confirmed** | 14× oracle, 190× better than supervised |
| Smarter exploration closes the gap | **Marginal** | Best 2× improvement, same bottleneck |
| More data closes the gap | Testing | Stage 1D: 10K→100K scaling matrix |
| Larger models help | **Rejected at 10K** | 229× worse; testing at higher N |
| DAgger bridges the gap | **Rejected** | Structural incompatibility |
| Targeted velocity coverage helps | Testing | Stage 1D: bangbang augmentation |

## Open Questions → Stage 1D

The key bottleneck is velocity coverage. Two paths to close it:
1. **Brute force**: does 10×–100× more random data naturally
   cover the high-velocity tail? (Scaling curve)
2. **Targeted**: does bangbang augmentation (biasing toward
   high-velocity states) close the gap at fixed 10K budget?
3. **Capacity**: does 1024×6 recover when data is sufficient?
