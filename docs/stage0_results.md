# Stage 0 — Comparison Results

> **Date:** 2026-04-11  
> **Dataset:** 2 000 train / 500 test trajectories, 5 s @ 10 ms control dt (500 steps each)  
> **Config:** `configs/quick.yaml`  
> **Hardware:** CPU only (CUDA driver too old for installed PyTorch 2.11+cu130)

---

## Summary Table

| Policy | N | Mean MSE_q | Mean MSE_v | Mean MSE_total | Median MSE_total | Max MSE_total | Mean max err_q | Mean p95 err_q |
|---|---|---|---|---|---|---|---|---|
| **Oracle** (Newton shooting) | 50 | **1.71e-09** | **1.72e-08** | **3.43e-09** | **6.74e-28** | 8.83e-08 | **1.71e-05** | **1.59e-05** |
| **Supervised MLP** (30 epochs) | 200 | 1.57e-02 | 3.81e-02 | 1.95e-02 | 3.16e-03 | 5.42e-01 | 2.09e-01 | 1.71e-01 |
| **Zero torque** (baseline) | 200 | 6.06e-01 | 6.60e+00 | 1.27e+00 | 1.10e+00 | 3.73e+00 | 2.07e+00 | 1.57e+00 |

All metrics are closed-loop (the policy observes the actual state and must track multisine references).  
`MSE_total = MSE_q + 0.1 × MSE_v`

---

## Interpretation

### Oracle (Newton shooting) ✅
- Near machine-epsilon error (mean MSE_total ≈ 3.4 × 10⁻⁹).
- Median MSE_total ≈ 6.7 × 10⁻²⁸ — most trajectories are essentially perfect.
- Confirms the inverse-dynamics oracle is the correct upper bound.
- Verified separately: `Max |open-loop error| = 8.88e-15` (see `verify_oracle` output).

### Supervised MLP (Stage 1A) ✅
- Mean MSE_total ≈ 1.95 × 10⁻², roughly **5.7 × 10⁵× worse** than oracle in mean.
- Median MSE_total ≈ 3.16 × 10⁻³ — typical trajectory is better than the mean (heavy tail).
- Best val loss after 30 epochs: **1.075**; training loss at convergence: **1.018** (both in squared-torque units).
- Gap to oracle is expected: this is a 30-epoch run on 2 k training trajectories without data augmentation, normalisation tricks, or teacher forcing.

### Zero Torque
- Mean MSE_total ≈ 1.27 — sets the "do-nothing" floor.
- Supervised MLP beats zero torque by **65×** in mean MSE_total.

---

## Gap Analysis

```
Oracle          : 3.43e-09  (upper bound)
Supervised MLP  : 1.95e-02  (×5.7M above oracle)
Zero torque     : 1.27e+00  (×3.7e8 above oracle)

MLP improvement over zero : ×65
Remaining gap to oracle   : ×5.7M
```

The MLP clearly learns physics, but large-error tails (max MSE_total 0.54)
indicate instability on some trajectories — consistent with the model having
seen only 2 k trajectories for 30 epochs.

---

## Next Steps to Close the Gap

1. **Scale data** — use full 80 k / 20 k split (`configs/default.yaml`); expected gap to shrink by ~2 orders of magnitude.
2. **More epochs** — 30 epochs is early; cosine LR schedule suggests learning is still improving.
3. **Stage 1B** — add random-rollout data (`scripts/generate_random_rollout.py`) for better state coverage.
4. **Normalisation** — already implemented in `InverseDynamicsMLP`; verify input mean/std are stable.
5. **Architecture search** — try deeper (4–5 hidden) or wider (512) MLP.

---

## Artifacts

| File | Description |
|---|---|
| `outputs/oracle_eval.json` | Oracle closed-loop eval (50 trajectories) |
| `outputs/supervised_eval.json` | Supervised MLP closed-loop eval (200 trajectories) |
| `outputs/zero_eval.json` | Zero-torque eval (200 trajectories) |
| `outputs/oracle_verification.png` | Oracle open-loop error histogram |
| `outputs/stage1a/best_model.pt` | Best MLP checkpoint (epoch 29, val=1.0749) |
| `outputs/stage1a/training_log.json` | Per-epoch train/val loss + elapsed time |
| `data/train.h5` | 2 000-trajectory multisine training set |
| `data/test.h5` | 500-trajectory multisine test set |
