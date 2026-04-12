# Stage 1 Synthesis

## The Two Metrics That Matter

1. **Standard benchmark MSE** (6 families × 100 traj, closed-loop tracking)
2. **Dataset coverage/entropy** (4D state histogram, occupied bins / 10K)

Everything else (val-loss, ID/OOD splits, degradation ratios) proved
unreliable or misleading. Val-loss doesn't even rank methods correctly
(DAgger: best val-loss, worst benchmark).

---

## Full Ranking

| # | Model | Bench MSE | ×Oracle | Stage |
|---|-------|-----------|---------|-------|
| 1 | active | 1.86e-3 | 10× | 1C |
| 2 | hybrid_select | 2.15e-3 | 12× | 1C |
| 3 | rebalance | 2.18e-3 | 12× | 1C |
| 4 | random | 2.67e-3 | 14× | 1B |
| 5 | density | 3.17e-3 | 17× | 1C |
| 6 | hybrid_weighted | 4.49e-3 | 24× | 1C |
| 7 | adversarial | 5.01e-3 | 27× | 1C |
| 8 | hindsight | 8.13e-3 | 44× | 1C |
| 9 | maxent_rl | 2.39e-2 | 129× | 1C |
| 10 | supervised | 3.52e-2 | 190× | 1A |
| 11 | dagger_512×4 | 6.09e-1 | 3295× | 1D |
| 12 | dagger_1024×4 | 6.22e-1 | 3362× | 1D |
| — | 20K random | TBD | TBD | 1D |
| — | 50K random | TBD | TBD | 1D |

## What We Learned

| Finding | Evidence |
|---------|----------|
| Random rollout beats supervised 13× | benchmark: 2.67e-3 vs 3.52e-2 |
| Selection strategy gives ≤1.4× | active 1.86e-3 vs random 2.67e-3 |
| Data quantity gives ≥6× | 20K val 2.29e-4 vs 10K val 1.39e-3 |
| Uniform coverage is harmful | maxent_rl 129× oracle (Goldilocks) |
| Val-loss is unreliable cross-method | DAgger: val 1.14e-3, bench 0.639 |
| Hard families = tail accuracy | step 23×, rw 61× — only 3-6% outside range |

## Completion Criteria (from README)

| Criterion | Status |
|-----------|--------|
| Match supervised on easy families | ✅ All methods ≤ 2× oracle |
| Beat supervised on hard families | ✅ 13-19× better |
| Approach oracle broadly | ❌ Best: 10× (pending 1D scaling) |
| Understand mechanisms | 🔄 Coverage + scaling; ablations needed |

## Open Question

Can 20K-50K random data + 1024×6 architecture close the 10× oracle gap?
Val-loss at 20K (2.29e-4) is within 1.24× of oracle — if benchmark
confirms, Stage 1 is essentially solved by pure scaling.