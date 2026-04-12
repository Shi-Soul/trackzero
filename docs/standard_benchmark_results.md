# Standard Benchmark Results

## Benchmark Design

Fixed, method-agnostic benchmark: 600 tracking trajectories across 6 signal families.
Each family contributes 100 trajectories, all deterministically seeded (seed=12345).

| Family | Description | Difficulty |
|--------|-------------|-----------|
| multisine | In-distribution (from test set) | Easy |
| chirp | Frequency-sweeping sinusoids | Easy |
| sawtooth | Ramping torque signals | Easy |
| pulse | Square pulse torques | Easy |
| step | Sudden torque changes → large excursions | Hard |
| random_walk | Cumulative random torques → drift | Hard |

**Single metric**: mean tracking MSE across ALL 600 trajectories.

## Comprehensive Ranking (39 models)

### Tier 1: Best methods (< 0.01)

| Rank | Method | Aggregate MSE | Category |
|------|--------|--------------|----------|
| 1 | active | 1.858e-3 | Stage 1C |
| 2 | hybrid_select | 2.155e-3 | Stage 1C |
| 3 | rebalance | 2.176e-3 | Stage 1C |
| 4 | random | 2.666e-3 | Stage 1C |
| 5 | density | 3.172e-3 | Stage 1C |
| 6 | hybrid_weighted | 4.492e-3 | Stage 1C |
| 7 | adversarial | 5.011e-3 | Stage 1C |
| 8 | ablation n10k | 7.717e-3 | Ablation |
| 9 | hindsight | 8.130e-3 | Stage 1C |

### Tier 2: Moderate (0.01 - 0.05)

| Rank | Method | Aggregate MSE | Category |
|------|--------|--------------|----------|
| 10 | ablation n5k | 1.117e-2 | Ablation |
| 11 | ablation n2k | 1.658e-2 | Ablation |
| 12 | maxent_rl | 2.390e-2 | Stage 1C |
| 13 | ablation mixed | 2.501e-2 | Ablation |
| 14 | supervised_1a | 3.522e-2 | Stage 1A |
| 15 | hybrid_curriculum | 4.509e-2 | Stage 1C |

### Tier 3: Probe models (0.05+, 256×3 arch, 2K traj)

| Rank | Model | Coverage | MSE |
|------|-------|----------|-----|
| 16 | probe mixed_uniform | 0.627 | 5.30e-2 |
| 17 | probe mixed_all_equal | 0.606 | 5.65e-2 |
| 18-25 | other mixed/ou probes | 0.50-0.57 | 7e-2 to 1.5e-1 |
| 35-36 | gaussian probes | 0.22-0.28 | 2.6e-1 to 2.9e-1 |
| 37-39 | bangbang probes | 0.75 | 1.06 to 1.08 |

## Key Findings

1. **Active learning is the champion** on the standard benchmark (1.86e-3)
2. **Coverage predicts performance** within learnable action types (r=-0.946)
3. **Bangbang breaks coverage**: highest coverage but worst performance
4. **Data scaling is log-linear**: 10× data → 8× improvement
5. **Action diversity critical**: mixed >> single type, bangbang 25× worse

## Pending: HP Sweep + DAgger Results

8 experiments running (epoch ~60/200). Will update when complete.
