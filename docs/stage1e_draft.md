# Stage 1E: Synthesis

## Research Questions

1. Which mechanisms actually close the oracle gap?
2. What is the best TRACK-ZERO configuration?
3. How does TRACK-ZERO compare to supervised baseline and oracle?

## Complete Stage 1 Results

### Stage 1A–1C: Exploration Methods (10K data, 512×4)

| Method | AGG MSE | ×Oracle | Key Insight |
|--------|---------|---------|-------------|
| active | 1.86e-3 | 10× | Best exploration, only 1.4× over random |
| random | 2.67e-3 | 14× | Naive baseline |
| supervised | 3.52e-2 | 190× | No physics access |

All 9 exploration methods cluster within 1.5× of random. Exploration
strategy is **not** the bottleneck.

### Stage 1D: Architecture, Data Engineering, Training Optimization

Tested on 1024×6 architecture with 10K mixed-torque data:

| Category | Best Method | AGG MSE | ×Baseline | Verdict |
|----------|-----------|---------|-----------|---------|
| **Training optimization** | **cosine+wd** | **4.19e-4** | **0.53×** | **Dominant lever** |
| Training optimization | cosine LR only | 5.64e-4 | 0.72× | Helps random_walk |
| Training optimization | weight decay only | 7.17e-4 | 0.91× | Helps step |
| Architecture (bigger) | 1024×6 baseline | 7.87e-4 | 1.00× | Reference |
| Architecture (smaller) | 512×4 | 1.02e-3 | 1.29× | Worse at 10K |
| Data engineering | various | 0.90–1.92e-1 | 1.1–192× | All negative |
| Data scaling | 512×4 @ 20K | 3.24e-4 | 0.41× | More data helps |

### Per-Family Structure

The oracle gap concentrates in 2 of 6 benchmark families:

| Family | Oracle | Best (cosine+wd) | Gap | Nature |
|--------|--------|-------------------|-----|--------|
| multisine | ~1e-8 | 1.76e-5 | ~1000× | Smooth periodic |
| chirp | ~1e-8 | 2.97e-5 | ~1000× | Frequency sweep |
| sawtooth | ~1e-8 | 1.27e-5 | ~1000× | Sharp periodic |
| pulse | ~1e-8 | 7.01e-6 | ~1000× | Step-like |
| **step** | ~1e-8 | **1.80e-3** | **~10⁵×** | Discontinuous |
| **random_walk** | ~1e-8 | **6.45e-4** | **~10⁴×** | Chaotic |

Step and random_walk have **opposite capacity requirements**:
- Step: benefits from regularization (WD cuts error 53%)
- Random_walk: benefits from fine-tuning (cosine cuts error 47%)
- Cosine+WD addresses both simultaneously (synergistic)

## Answers to Research Questions

### Q1: Which mechanisms close the oracle gap?

**Ranked by effect size (at 10K data):**

1. **Training optimization (cosine+WD): 47% reduction** — the only
   intervention that materially improves aggregate performance.
2. **Data quantity (10K→20K): 59% reduction** — brute-force scaling
   with smaller model (512×4).
3. **Exploration strategy: <5% effect** — all methods equivalent.
4. **Data engineering (selection, weighting, augmentation): 0%** —
   no method beats naive random.

### Q2: Best TRACK-ZERO configuration?

At 10K data: **1024×6 + cosine LR (T_max=200, η_min=1e-6) +
weight_decay=1e-4 + Adam lr=3e-4 + MSE loss + 200 epochs**.
Aggregate MSE: 4.19e-4 (11,277× oracle).

### Q3: How does TRACK-ZERO compare?

| Configuration | AGG MSE | ×Oracle | ×Supervised |
|--------------|---------|---------|-------------|
| Oracle | 3.72e-8 | 1× | — |
| 512×4 20K | 3.24e-4 | 8,710× | 109× better |
| **1024×6 cosine+wd 10K** | **4.19e-4** | **11,277×** | **84× better** |
| 1024×6 baseline 10K | 7.87e-4 | 21,163× | 45× better |
| Supervised (Stage 1A) | 3.52e-2 | 946,000× | 1× |

TRACK-ZERO is **45–109× better than supervised** across all
configurations. However, an 11,000× gap to oracle remains.

## Remaining Oracle Gap Analysis

The ~10⁴× gap to oracle is dominated by step and random_walk families.
These involve high-velocity transient states where:
1. Training data density is low (< 0.01% above 10 rad/s)
2. The dynamics are nonlinear (large Coriolis/centripetal terms)
3. The inverse map has high local curvature

Closing this gap likely requires moving beyond i.i.d. supervised
learning on random rollouts — this motivates **Stage 2** (online
learning, adaptive data collection, or model-based approaches).

## Stage 1 Completion Assessment

- ✅ Established that physics-only data >> supervised (45–109×)
- ✅ Identified that exploration strategy is not the bottleneck
- ✅ Found that training optimization (cosine+WD) is the dominant lever
- ✅ Characterized the remaining gap (step + random_walk families)
- ✅ Ruled out: DAgger, bangbang torques, data selection/weighting
- ⚠️ Remaining gap: 11,277× oracle — requires Stage 2 approaches
