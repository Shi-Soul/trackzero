# TRACK-ZERO Roadmap Status

_Updated: epoch ~80, oracle benchmark computed._

## Current Status

**Stage 1D in progress. Oracle lower bound established.**

- **Stage 0 ✅**: Data pipeline, oracle, training loop, evaluation harness.
- **Stage 1A ✅**: Supervised ID MLP baseline.
- **Stage 1B ✅**: Random-rollout self-supervised; coverage/scaling analysis.
- **Stage 1C ✅**: 13 data collection strategies + 18 coverage probes + 10 ablations.
- **Stage 1D 🧪**: HP sweeps + DAgger running on all 8 GPUs.

## Theoretical Bounds

| Level | MSE | Description |
|-------|-----|-------------|
| **Shooting oracle** | 4.6e-8 | Perfect inverse dynamics (impractical) |
| **FD oracle** | **1.85e-4** | Finite-difference oracle (model target) |
| **Best MLP (active)** | 1.86e-3 | 10.0× above oracle |
| **Random baseline** | 2.67e-3 | 14.4× above oracle |

**The 10× MLP-to-oracle gap is concentrated on 2 of 6 families:**
- Step: 23.5× gap (q2 excursion 3.7× beyond training range)
- Random_walk: 61.6× gap (similar extrapolation issue)
- Easy families (multisine, chirp, sawtooth, pulse): gap ≤ 2× (solved)

## Running Experiments (epoch ~80)

| GPU | Experiment | best_val | Notes |
|-----|-----------|----------|-------|
| 0 | hp_random_1024x6 | **0.001675** | Below active benchmark! |
| 1 | hp_random_2048x3 | 0.002744 | Good |
| 2 | hp_random_512x4_wd | 0.006252 | Slow |
| 3 | hp_maxent_1024x6 | 0.028030 | Hopeless |
| 4 | hp_maxent_2048x4 | >0.095 | Very slow |
| 5 | dagger_512x4 | **0.001540** | Best val_loss |
| 6 | dagger_1024x4 | 0.001681 | Excellent |
| 7 | hp_random_1024x4_wd | 0.002301 | Strong |

## Key Findings So Far

1. **Coverage ↔ performance**: r = −0.946 within learnable action types
2. **Oracle gap**: 10× between best MLP and FD oracle, 95% on step + random_walk
3. **Error concentration**: Top 10 of 100 step trajectories = 85.7% of step error
4. **Root cause**: q2 exceeds training 99th percentile by 3.7× in hard families
5. **Depth > width**: 1024×6 (5.3M) beats 2048×3 (8.4M) on same data
6. **Capacity closes gap**: hp_random_1024x6 val_loss < active benchmark MSE
7. **Maxent fails**: Even 12.6M param model → 15× worse (data quality issue)
8. **Bangbang paradox**: Highest coverage (75.5%) but worst performance (unlearnable)

## Remaining Steps

1. **Benchmark all HP/DAgger models** when training completes (~3-4h)
2. **Analyze DAgger iteration curve** (does task-focused data augmentation help?)
3. **Final unified ranking** (47+ models)
4. **Coverage vs benchmark plot** across ALL models
5. **Stage 1E**: Synthesis document — what strategy wins and why
6. **Stage 2**: Infeasible reference tracking (future)
7. **Stage 3-4**: Real-world transfer (future)
