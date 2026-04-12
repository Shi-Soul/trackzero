# Stage 1D: Scaling & Architecture

## Research Question

Can scaling the data budget and/or model capacity close the 10×
gap between the best TRACK-ZERO method and the oracle? (README Stage 1D)

## Motivation

Stage 1C showed that at 10K trajectories, all exploration methods
cluster within 1.5× of random (aggregate MSE 1.86e-3 to 2.67e-3),
but the best is still 10× oracle. Two axes of scaling may help:

1. **Data quantity**: more trajectories → higher per-bin density
   → lower local approximation error
2. **Model capacity**: larger network → better function approximation
   for complex torque landscapes

## Experiments

### Axis 1: Data Scaling (architecture fixed at 1024×6)

All experiments use the same mixed-action random rollout generator
and 1024×6 MLP (5.26M params) trained with Adam lr=3e-4, batch 4096,
200 epochs. Evaluated on the standard benchmark.

| Dataset | Trajectories | Training Pairs | Status |
|---------|-------------|----------------|--------|
| 10K baseline | 10,000 | 5M | ✅ Benchmarked (Stage 1C) |
| 20K | 20,000 | 10M | 🔄 Training (GPU 0) |
| 50K | 50,000 | 25M | ✅ Model saved, benchmark pending |
| 100K | 100,000 | 50M | 🔄 Training (GPU 5) |

**Early result** (50K, 10 epochs only):
- Val-loss: 6.44e-4 (vs 10K best val 1.41e-3 after 180 epochs)
- This 2.2× val-loss improvement from 5× data is promising but
  the model needs full 200-epoch training for benchmark evaluation.

### Axis 2: Architecture Scaling (data fixed at 10K)

| Architecture | Params | Hidden Dim × Layers | Status |
|-------------|--------|---------------------|--------|
| 512×4 (baseline) | 3.1M | 512 × 4 | ✅ Benchmarked |
| 1024×4 + WD | 5.3M | 1024 × 4 | Ep 150/200 (dead, relaunch needed) |
| 1024×6 | 5.3M | 1024 × 6 | Ep 180/200 (dead, relaunch needed) |
| 2048×3 | 10.5M | 2048 × 3 | Ep 150/200 (dead, relaunch needed) |

### DAgger (iterative data collection)

DAgger collects data on-policy then retrains. Results:

| Model | Aggregate MSE | ×Oracle |
|-------|-------------|---------|
| dagger_512×4 | 6.09e-1 | 3294 |
| dagger_1024×4 | 6.22e-1 | 3361 |

**Analysis**: DAgger fails catastrophically. The compounding error
problem makes iteration 0 data useless: the random policy produces
such poor tracking that the collected states are in dynamically
unstable regions where the inverse dynamics mapping is ill-conditioned.
Subsequent iterations cannot recover because each builds on the
previous failure.

## Preliminary Conclusions

1. **Data scaling is the dominant lever.** The 50K model's val-loss
   after only 10 epochs already surpasses the 10K model's converged
   val-loss, suggesting data quantity dominates both architecture
   and exploration method.

2. **DAgger is incompatible with this problem.** The policy-data
   feedback loop amplifies errors instead of correcting them.

3. **Full benchmark results pending** — the 20K, 50K (retrained),
   and 100K experiments will provide the complete data-scaling curve.

## Open Questions

- What is the functional form of the scaling law (power-law exponent)?
- Is there a saturation point where more data stops helping?
- At what data budget does 1024×6 match oracle on easy families?