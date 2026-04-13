# Experiment Status

## Stage Progress

| Stage | Status | Key Result | Doc |
|-------|--------|------------|-----|
| 0 | ✅ | Infrastructure: GPU simulator, oracle, eval harness | stage0.md |
| 1A | ✅ | Supervised baseline: AGG=3.52e-2 (190× oracle) | stage1a.md |
| 1B | ✅ | Random rollout: AGG=2.67e-3 (14× oracle) | stage1b.md |
| 1C | ✅ | 9 exploration methods: all within 1.5× of random | stage1c.md |
| 1D | ✅ | Cosine+WD: AGG=4.19e-4 (47% improvement) | stage1d.md |
| 2A | ✅ | MLP 5× better than oracle under noise (σ≥0.01) | stage2.md |
| 2C | ✅ | Noise augmentation: precision-robustness tradeoff | stage2.md |
| 3A | 🔄 | 3-link: 172× oracle (vs 9× for 2-link). 5-link pending | stage3.md |

## Best 2-Link Results

| Method | AGG MSE | vs Oracle |
|--------|---------|-----------|
| Oracle (mj_inverse) | 7.63e-5 | 1.0× |
| Residual MLP (oracle input) | 1.95e-5 | 0.26× (better) |
| Stage 1 best (1024×6+cosine+WD) | 4.19e-4 | 5.5× |
| Noise-augmented MLP | 1.39e-1 | 1,823× |

## Key Findings

1. **MLP has implicit noise robustness**: smooth function approximation
   filters noisy references better than exact inverse dynamics.
2. **Oracle gap widens with DOF**: 9× for 2-link → 172× for 3-link.
3. **Residual architecture achieves 3.9× better than oracle** (but
   requires oracle at inference time).
4. **Noise augmentation works but is blunt**: clean precision drops 78×,
   noisy tracking improves 4.5×.

## Active Experiments

- 5-link chain training (ETA ~20 min)
