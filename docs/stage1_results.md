# Stage 1 — Results: Supervised (1A) vs Random Rollout (1B)

> **Date:** 2026-04-11  
> **Dataset:** 2 000 train / 500 test multisine trajectories (`configs/quick.yaml`)  
> **Hardware:** CPU only (CUDA driver incompatible with torch 2.11+cu130)

---

## Training Setup

| | Stage 1A | Stage 1B |
|---|---|---|
| **Data source** | Multisine reference trajectories | Random rollout (mixed: uniform/gaussian/OU/bangbang/multisine) |
| **# train trajectories** | 2 000 | 2 000 |
| **# training pairs** | 1 000 000 | 1 000 000 |
| **Architecture** | 256×3 MLP | 256×3 MLP |
| **# parameters** | 134 402 | 134 402 |
| **Epochs** | 30 | 30 |
| **Batch size** | 65 536 | 65 536 |
| **Optimizer** | Adam, cosine LR 1e-3→0 | Adam, cosine LR 1e-3→0 |
| **Best val loss** | **1.075** | 1.395 |

---

## In-Distribution Evaluation (multisine test set, 200 trajectories)

| Policy | Mean MSE_q | Mean MSE_v | Mean MSE_total | Median MSE_total | Max MSE_total | Mean max err_q |
|---|---|---|---|---|---|---|
| **Oracle** (Newton shooting, n=50) | **1.71e-09** | **1.72e-08** | **3.43e-09** | **6.74e-28** | 8.83e-08 | **1.71e-05** |
| **1A Supervised** | **1.57e-02** | 3.81e-02 | **1.95e-02** | **3.16e-03** | 5.42e-01 | **2.09e-01** |
| **1B Random Rollout** | 2.13e-02 | 6.24e-02 | 2.75e-02 | 9.54e-03 | 1.03e+00 | 2.71e-01 |
| Zero torque | 6.06e-01 | 6.60e+00 | 1.27e+00 | 1.10e+00 | 3.73e+00 | 2.07e+00 |

**On in-distribution multisine data, 1A beats 1B by ~1.4× in mean MSE_total** — expected,
since 1A trains on the same signal family it is evaluated on.

---

## Out-of-Distribution Evaluation (100 trajectories per type)

Mean MSE_total on OOD reference types not seen during training:

| OOD Type | 1A Supervised | 1B Random Rollout | Ratio (1A/1B) | Winner |
|---|---|---|---|---|
| chirp | 1.91e-02 | 2.68e-02 | 0.71× | **1A** |
| **step** | 3.30e-01 | 2.75e-01 | 1.20× | **1B** |
| **random_walk** | 3.27e-01 | 2.05e-01 | 1.59× | **1B** |
| sawtooth | 6.27e-03 | 1.24e-02 | 0.50× | **1A** |
| pulse | 4.06e-03 | 6.01e-03 | 0.68× | **1A** |
| **mixed_ood** | 1.56e-01 | 8.92e-02 | 1.75× | **1B** |

### Key observations

- **1A generalises well to smooth OOD signals** (chirp ≈ sinusoid variant, sawtooth/pulse are short-duration):
  MSE_total stays in the same range as in-distribution (1e-2 to 6e-3).
- **1A struggles with discontinuous references** (step, random_walk):
  MSE_total jumps to ~0.33, 7–17× worse than its in-distribution score.
- **1B is more robust to step and random_walk**: trained on OU/bangbang/uniform
  signals that include abrupt changes, giving it a 1.2–1.6× advantage on those types.
- **On mixed_ood (all OOD types combined) 1B wins by 1.75×** — the clearest summary
  of the generalisation gap.

---

## Gap Analysis

```
                     ID (multisine)    mixed_ood
Oracle              : 3.43e-09         —
1A Supervised       : 1.95e-02         1.56e-01   (+8×  vs ID)
1B Random Rollout   : 2.75e-02         8.92e-02   (+3.2× vs ID)
Zero torque         : 1.27e+00         ~1.3e+00   (flat — does nothing)

1A OOD degradation factor : ×8.0
1B OOD degradation factor : ×3.2   ← 1B degrades much less
```

---

## Rollout Visualizations

The following files are generated in `outputs/rollout_viz/` (trajectory #3, 5 s):

| File | Content |
|---|---|
| `traj3_state_traces.png` | q1, q2, dq1, dq2 time series — all 4 policies vs reference |
| `traj3_oracle.gif` | Double-pendulum animation — oracle (blue) vs reference (red) |
| `traj3_supervised_1a.gif` | Animation — 1A supervised vs reference |
| `traj3_supervised_1b.gif` | Animation — 1B random rollout vs reference |
| `traj3_zero.gif` | Animation — zero torque vs reference |

**To regenerate or pick a different trajectory:**

```bash
# State-trace plot only (fast)
.venv/bin/python scripts/visualize_rollout.py \
    --config configs/quick.yaml \
    --dataset data/test.h5 \
    --trajectory-idx 3 \
    --policies oracle supervised_1a supervised_1b zero \
    --checkpoint-1a outputs/stage1a/best_model.pt \
    --checkpoint-1b outputs/stage1b/best_model.pt \
    --output-dir outputs/rollout_viz

# With GIF animations (add --animate)
.venv/bin/python scripts/visualize_rollout.py \
    --config configs/quick.yaml \
    --dataset data/test.h5 \
    --trajectory-idx 3 \
    --policies oracle supervised_1a supervised_1b zero \
    --checkpoint-1a outputs/stage1a/best_model.pt \
    --checkpoint-1b outputs/stage1b/best_model.pt \
    --animate --animate-skip 5 \
    --output-dir outputs/rollout_viz
```

---

## Conclusions

1. **1A (supervised)** is the better in-distribution model: lower val loss, lower ID MSE.
   It generalises well to smooth signals but degrades heavily on discontinuous ones.

2. **1B (random rollout)** is more broadly robust: 1.75× better on mixed OOD, especially
   on step/random_walk thanks to bangbang and uniform noise in its training distribution.

3. **Both are far from the oracle** (~6–8M× on ID). Closing this gap requires:
   - Full-scale data (80k trajectories) + GPU training
   - Longer training (>30 epochs; loss curves still descending at epoch 30)
   - A hybrid 1A+1B training mix may combine in-distribution accuracy with OOD robustness

---

## Artifacts

| File | Description |
|---|---|
| `outputs/stage1a/best_model.pt` | 1A best checkpoint (epoch 29, val=1.075) |
| `outputs/stage1b/best_model.pt` | 1B best checkpoint (epoch 29, val=1.395) |
| `outputs/stage1a/training_log.json` | Per-epoch loss for 1A |
| `outputs/stage1b/training_log.json` | Per-epoch loss for 1B |
| `outputs/supervised_eval.json` | 1A closed-loop eval (200 traj) |
| `outputs/stage1b_eval.json` | 1B closed-loop eval (200 traj) |
| `outputs/ood_eval/ood_comparison.json` | 1A vs 1B on all OOD types |
| `outputs/rollout_viz/traj3_state_traces.png` | State-trace comparison plot |
| `outputs/rollout_viz/traj3_*.gif` | Per-policy pendulum animations |
