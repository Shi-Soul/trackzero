# Experiment Status

## Completed Stages

| Stage | Status | Key Result |
|-------|--------|------------|
| 0 | ✅ | Infrastructure: GPU simulator, oracle, eval harness |
| 1A | ✅ | Supervised baseline: AGG=3.52e-2 (190× oracle) |
| 1B | ✅ | Random rollout: AGG=2.67e-3 (14× oracle, 512×4) |
| 1C | ✅ | 9 exploration methods: all within 1.5× of random |
| 1D | ✅ | Training optimization: **cosine+WD = 47% improvement** |

## Stage 1D Final Results (All 10K data, 1024×6 unless noted)

| Method | AGG MSE | ×Baseline | Status |
|--------|---------|-----------|--------|
| **1024×6+cosine+wd** | **4.19e-4** | **0.53×** | ✅ Best |
| 1024×6+cosine | 5.64e-4 | 0.72× | ✅ |
| 1024×6+wd1e-4 | 7.17e-4 | 0.91× | ✅ |
| 1024×6 baseline | 7.87e-4 | 1.00× | ✅ Reference |
| 512×4 (10K) | 1.02e-3 | 1.29× | ✅ |
| 1024×6+huber | 1.01e-3 | 1.29× | ✅ |
| 512×4+cosine | 1.17e-3 | 1.48× | ✅ |
| 1024×6+dropout | 1.97e-3 | 2.50× | ✅ |
| 256×3 (10K) | 2.44e-3 | 3.10× | ✅ |
| 512×4 (20K) [ref] | 3.24e-4 | 0.41× | ✅ |

## Key Findings

1. **Training optimization > data engineering**: Cosine LR + weight
   decay achieves 47% aggregate improvement — more than any exploration
   or data-selection strategy from Stage 1C.

2. **Cosine+WD is synergistic**: Cosine helps random_walk (-47%), WD
   helps step (-34%), combined effect (-47% AGG) exceeds either alone.

3. **Bigger model is better at 10K**: 1024×6 > 512×4 > 256×3. The
   earlier "3300× overfitting" was a tau_max bug (commit 1188bfa).

4. **Data engineering fails**: Wide v₀, bangbang torques, coverage
   selection, weighted training — all worse than naive random.

## Next Steps

- Stage 1E: Synthesis across all Stage 1 findings
- Stage 2: Beyond supervised learning (README roadmap)
