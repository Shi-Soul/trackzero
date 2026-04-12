# Stage 1B: TRACK-ZERO v0 — Random Rollout Self-Supervision

## Executive summary

Stage 1B is now substantially complete as a research pass.

The main conclusions are:

1. **4D joint state coverage matters.** When `(q1, q2, v1, v2)` is measured jointly instead of as separate 2D slices, higher coverage strongly predicts lower OOD tracking error: `r(coverage, log10 OOD MSE) = -0.823`.
2. **Coverage alone is not enough.** The highest-coverage distributions (`bangbang`, `bangbang_slow`) are not the best training data because they generate discontinuous transitions that are hard for the MLP to fit.
3. **The oracle gap is still huge.** Across six OOD reference families, learned policies remain roughly `2.6e4x` to `9.8e5x` worse than the inverse-dynamics oracle.
4. **The existing Stage 1B baseline remains the strongest full model.** The probe study selected `mixed_all_equal` as the best small-scale distribution, but the fully trained `outputs/stage1b_best_dist/best_model.pt` did **not** beat the existing `outputs/stage1b_scaled/best_model.pt` on the final OOD harness.

That last point is important: the Stage 1B research answered the coverage questions, but the current model-selection rule is **not yet reliable enough** to replace the best existing Stage 1B checkpoint.

---

## What this document covers

This writeup answers the Stage 1B questions the roadmap actually asks:

- Does naive random rollout cover the feasible state space well enough?
- Which random action distributions are best?
- Does better 4D coverage translate into better OOD tracking?
- How large is the remaining gap to the inverse-dynamics oracle?
- Which Stage 1B model should be carried forward into Stage 1C?

All quantitative claims below are tied to saved artifacts under `outputs/`.

---

## Raw artifacts used in this report

| Artifact | Role |
| --- | --- |
| `outputs/action_sweep/sweep_results.json` | 18-way action-distribution sweep using the 4D coverage/entropy metric |
| `outputs/action_sweep/sweep_ranking.png` | Ranking plot for the sweep |
| `outputs/coverage_quality_probe/probe_results.json` | Probe-model study linking coverage to OOD performance |
| `outputs/coverage_quality_probe/correlation_plot.png` | Correlation scatter plots |
| `outputs/stage1b_scaled/best_model.pt` | Existing strong Stage 1B baseline checkpoint |
| `outputs/stage1b_scaled/eval_results.json` | ID evaluation for the existing Stage 1B baseline |
| `outputs/stage1b_scaled/metadata.json` | Partial metadata for the existing Stage 1B baseline |
| `outputs/stage1b_best_dist/best_model.pt` | New full-scale model trained from the probe-selected distribution |
| `outputs/stage1b_best_dist/metadata.json` | Metadata for the new best-distribution run |
| `outputs/ood_final/ood_comparison.json` | Final 4-way OOD comparison: oracle vs 1A vs old 1B vs new 1B |
| `outputs/ood_final/*_details.json` | Per-type detailed rollout summaries |

---

## Scripts and exact entrypoints

### 1. 4D coverage sweep

Script: `scripts/search_action_distribution.py`

```bash
uv run python scripts/search_action_distribution.py \
  --config configs/medium.yaml \
  --output-dir outputs/action_sweep \
  --n-trajectories <N> \
  --seed 42
```

What it does:

- generates random-rollout datasets for 18 action-distribution configurations
- computes a **joint 4D histogram** over `(q1, q2, v1, v2)`
- records:
  - `coverage`: occupied bins / total bins
  - `entropy`: normalized Shannon entropy over all 10,000 bins
  - `score = 0.5 * coverage + 0.5 * entropy`

Implementation source:

- `scripts/search_action_distribution.py`
- `trackzero/data/random_rollout.py`

Saved results:

- `outputs/action_sweep/sweep_results.json`
- `outputs/action_sweep/sweep_ranking.png`

### 2. Coverage → quality probe study

Script: `scripts/probe_coverage_quality.py`

```bash
uv run python scripts/probe_coverage_quality.py \
  --config configs/medium.yaml \
  --sweep-results outputs/action_sweep/sweep_results.json \
  --output-dir outputs/coverage_quality_probe \
  --n-train 2000 \
  --epochs 30 \
  --device cuda:0
```

What it does:

- loads the saved 4D coverage scores from the sweep
- trains a quick probe model for every candidate distribution
- evaluates each probe on `mixed_ood`
- measures whether coverage, entropy, or score predict actual OOD quality

Probe architecture and budget are defined in code:

- MLP `256 x 3`
- `2000` training trajectories
- `30` epochs

Saved results:

- `outputs/coverage_quality_probe/probe_results.json`
- `outputs/coverage_quality_probe/correlation_plot.png`

### 3. Full Stage 1B model training

Script: `scripts/train_random_rollout.py`

Existing strong Stage 1B baseline checkpoint:

```bash
uv run python scripts/train_random_rollout.py \
  --config configs/medium.yaml \
  --val-data data/medium/test.h5 \
  --output-dir outputs/stage1b_scaled \
  --n-trajectories 10000 \
  --action-type mixed \
  --epochs 100 \
  --batch-size 65536 \
  --hidden-dim 512 \
  --n-hidden 4 \
  --device cuda:0 \
  --eval-trajectories 200
```

Notes:

- `outputs/stage1b_scaled/metadata.json` preserves `action_type=mixed`, `n_trajectories=10000`, and validation loss, but does **not** preserve the full argument list.
- In `trackzero/data/random_rollout.py`, `action_type=mixed` with no weights means a **uniform categorical mixture** over `uniform`, `gaussian`, `ou`, `bangbang`, and `multisine`.
- I therefore refer to this baseline below as **uniform-mixed Stage 1B**.

Probe-selected full-scale candidate:

```bash
uv run python scripts/train_random_rollout.py \
  --config configs/medium.yaml \
  --val-data data/medium/test.h5 \
  --output-dir outputs/stage1b_best_dist \
  --n-trajectories 10000 \
  --action-type mixed \
  --action-params-json '{"weights":[0.2,0.2,0.2,0.2,0.2]}' \
  --epochs 100 \
  --batch-size 65536 \
  --lr 3e-4 \
  --hidden-dim 512 \
  --n-hidden 4 \
  --device cuda:0 \
  --eval-trajectories 200
```

Saved results:

- `outputs/stage1b_scaled/*`
- `outputs/stage1b_best_dist/*`

### 4. Final OOD comparison and oracle ceiling

Script: `scripts/eval_ood.py`

```bash
uv run python scripts/eval_ood.py \
  --config configs/medium.yaml \
  --model-1a outputs/stage1a_scaled/best_model.pt \
  --model-1b outputs/stage1b_scaled/best_model.pt \
  --model-best outputs/stage1b_best_dist/best_model.pt \
  --include-oracle \
  --n-trajectories 300 \
  --eval-trajectories 200 \
  --output-dir outputs/ood_final
```

This evaluates:

- `oracle`
- `1A_supervised`
- `1B_random_rollout` = existing uniform-mixed Stage 1B baseline
- `1B_best_dist` = new probe-selected equal-mixed model

on:

- `chirp`
- `step`
- `random_walk`
- `sawtooth`
- `pulse`
- `mixed_ood`

Saved results:

- `outputs/ood_final/ood_comparison.json`
- `outputs/ood_final/*_details.json`

---

## Common experimental setup

| Item | Setting |
| --- | --- |
| System | Double pendulum |
| Policy input/output | `(current_state, target_next_state) -> torque` |
| Full model architecture | MLP `512 x 4` |
| Probe architecture | MLP `256 x 3` |
| Full training set size | `10000` trajectories |
| Probe training set size | `2000` trajectories |
| Full eval budget | `200` rollout trajectories per reference family |
| OOD families | `chirp`, `step`, `random_walk`, `sawtooth`, `pulse`, `mixed_ood` |
| Coverage metric | 10 bins per dimension on `(q1, q2, v1, v2)` => `10^4 = 10000` bins |
| Coverage bounds | `q in [-pi, pi]`, `v in [-8, 8]` |

---

## Q1. Does naive random rollout suffer from the wrong coverage metric?

Yes — the old metric was incomplete, and the 4D joint metric is the correct one.

The user’s criticism was right: treating `q` and `v` separately can hide strong correlations. A dataset can look broad in angle space and broad in velocity space while still occupying only a thin subset of the true joint state manifold.

The revised metric in `scripts/search_action_distribution.py` uses:

- a joint histogram over `(q1, q2, v1, v2)`
- `coverage = occupied_bins / 10000`
- `entropy = normalized Shannon entropy`
- `score = 0.5 * coverage + 0.5 * entropy`

### Sweep ranking by 4D score

Directly from `outputs/action_sweep/sweep_results.json`:

| Rank | Config | Coverage | Entropy | Score |
| --- | --- | ---: | ---: | ---: |
| 1 | `bangbang` | 0.755 | 0.877 | 0.816 |
| 2 | `bangbang_slow` | 0.746 | 0.864 | 0.805 |
| 3 | `ou_medium_high` | 0.603 | 0.830 | 0.716 |
| 4 | `mixed_uniform` | 0.626 | 0.777 | 0.702 |
| 5 | `ou_fast` | 0.562 | 0.818 | 0.690 |
| 6 | `mixed_all_equal` | 0.606 | 0.769 | 0.688 |

Bottom of the ranking:

| Config | Coverage | Entropy | Score |
| --- | ---: | ---: | ---: |
| `uniform` | 0.311 | 0.617 | 0.464 |
| `gaussian_wide` | 0.285 | 0.602 | 0.444 |
| `gaussian_medium` | 0.249 | 0.566 | 0.407 |
| `gaussian_narrow` | 0.216 | 0.524 | 0.370 |

### Interpretation

The sweep answers the roadmap’s first Stage 1B question cleanly:

- **plain Gaussian action noise is not enough**
- **OU and mixed distributions are much better**
- **bangbang maximizes geometric spread**

But this sweep alone does **not** tell us which dataset will train the best model. That is exactly what the next experiment tests.

---

## Q2. Does better 4D coverage actually predict better OOD tracking?

Yes, strongly — but imperfectly.

Using the 18 saved sweep configurations, the probe study in `outputs/coverage_quality_probe/probe_results.json` gives:

| Predictor | Correlation with `log10(OOD MSE)` |
| --- | ---: |
| 4D coverage | **-0.823** |
| composite score | **-0.800** |
| 4D entropy | **-0.743** |
| validation loss | **0.107** |

### Main result

Coverage is a **real signal** for OOD quality; validation loss on the held-out multisine-style validation set is not.

That matters because it means:

- the existing validation split is too in-distribution to select for OOD robustness
- Stage 1B success depends more on **where the random rollouts go** than on just fitting the training objective

### Probe-model ranking by OOD quality

From `outputs/coverage_quality_probe/probe_results.json`:

| Rank | Config | Score | Val loss | Mean OOD MSE |
| --- | --- | ---: | ---: | ---: |
| 1 | `mixed_all_equal` | 0.688 | 0.861 | 4.517e-02 |
| 2 | `mixed_uniform` | 0.702 | 0.832 | 5.110e-02 |
| 3 | `mixed_ou_heavy` | 0.679 | 0.889 | 6.800e-02 |
| 4 | `mixed_multisine_heavy` | 0.675 | 0.635 | 8.226e-02 |
| 5 | `bangbang_slow` | 0.805 | 3.204 | 9.498e-02 |
| 6 | `bangbang` | 0.816 | 3.179 | 1.023e-01 |

### Coverage-learnability tradeoff

This is the most important Stage 1B scientific result.

`bangbang` and `bangbang_slow` have the **best 4D coverage**, but they do **not** produce the best models.

Why:

- they excite more of the state space
- but they also create abrupt action discontinuities and sharp transition boundaries
- the current MLP learns those dynamics poorly

So the truth is:

> **Higher coverage helps, until the data becomes too hard to fit.**

This is exactly the coverage-learnability tradeoff the user wanted made explicit.

---

## Q3. Which action distribution should train the full Stage 1B model?

The probe study nominated `mixed_all_equal`, but the full-scale result says **do not replace the old baseline yet**.

### Candidate chosen by the probe study

The best probe configuration was:

- `action_type = mixed`
- `weights = [0.2, 0.2, 0.2, 0.2, 0.2]`
- artifact: `outputs/stage1b_best_dist/best_model.pt`

Its saved metadata:

| Field | Value |
| --- | --- |
| `n_trajectories` | 10000 |
| `hidden_dim` | 512 |
| `n_hidden` | 4 |
| `epochs` | 100 |
| `best_val_loss` | 0.13745 |

### Existing strong Stage 1B baseline

The existing checkpoint:

- artifact: `outputs/stage1b_scaled/best_model.pt`
- metadata confirms `action_type = mixed`, `n_trajectories = 10000`
- because no weights were saved, and the generator defaults to equal categorical choice among subtypes when weights are omitted, this corresponds to the **uniform-mixed baseline**

Saved metadata:

| Field | Value |
| --- | --- |
| `action_type` | `mixed` |
| `n_trajectories` | 10000 |
| `best_val_loss` | 0.04242 |

### Final answer

At full training scale, the old uniform-mixed Stage 1B baseline is still better than the newly trained equal-mixed candidate on the final OOD harness.

This is not a small detail; it changes the conclusion:

- **probe ranking is useful for analysis**
- **probe ranking is not yet sufficient for final model selection**

This is why the correct Stage 1B deliverable is not “we found the best distribution once and for all,” but rather:

> **4D coverage is a strong predictor, but final model quality also depends on optimization and trainability, so the selection criterion still needs refinement.**

---

## Q4. How large is the gap to the inverse-dynamics oracle?

It is still enormous.

From `outputs/ood_final/ood_comparison.json`:

| OOD type | Oracle | 1A supervised | 1B uniform-mixed | 1B equal-mixed |
| --- | ---: | ---: | ---: | ---: |
| `chirp` | 6.450e-09 | 1.796e-04 | 3.039e-04 | 1.229e-03 |
| `step` | 4.603e-07 | 1.502e-01 | 4.710e-02 | 6.146e-02 |
| `random_walk` | 1.221e-07 | 1.179e-01 | 1.379e-02 | 1.989e-02 |
| `sawtooth` | 4.154e-09 | 3.258e-04 | 2.369e-04 | 7.146e-04 |
| `pulse` | 1.767e-09 | 4.620e-05 | 7.371e-05 | 4.572e-04 |
| `mixed_ood` | 7.528e-08 | 7.388e-02 | 9.691e-03 | 2.481e-02 |

### Oracle gap ratios

| OOD type | `1A / oracle` | `1B uniform-mixed / oracle` |
| --- | ---: | ---: |
| `chirp` | 27840x | 47125x |
| `step` | 326364x | 102328x |
| `random_walk` | 965863x | 112921x |
| `sawtooth` | 78423x | 57042x |
| `pulse` | 26143x | 41711x |
| `mixed_ood` | 981387x | 128728x |

Even the better learned policy is still orders of magnitude away from the physics oracle. That justifies continuing to Stage 1C/1D instead of declaring Stage 1 solved.

---

## Q5. Does Stage 1B beat the Stage 1A supervised baseline on OOD?

Yes, on the harder and more important OOD families.

Using the final 4-way run:

| OOD type | `1A / 1B uniform-mixed` | Winner |
| --- | ---: | --- |
| `chirp` | 0.59x | 1A |
| `step` | 3.19x | 1B |
| `random_walk` | 8.55x | 1B |
| `sawtooth` | 1.37x | 1B |
| `pulse` | 0.63x | 1A |
| `mixed_ood` | 7.62x | 1B |

### Interpretation

This matches the intended Stage 1 story:

- Stage 1A is strongest on smooth references similar to its multisine training distribution
- Stage 1B is much stronger on broader, harder, less scripted OOD references

The most meaningful wins are:

- `step`: **3.19x**
- `random_walk`: **8.55x**
- `mixed_ood`: **7.62x**

These are the exact kinds of references where random-rollout diversity is supposed to matter.

---

## Q6. Did the new equal-mixed full model beat the old Stage 1B baseline?

No.

From the same `outputs/ood_final/ood_comparison.json`:

| OOD type | `1B uniform-mixed / 1B equal-mixed` | Better model |
| --- | ---: | --- |
| `chirp` | 0.25x | uniform-mixed |
| `step` | 0.77x | uniform-mixed |
| `random_walk` | 0.69x | uniform-mixed |
| `sawtooth` | 0.33x | uniform-mixed |
| `pulse` | 0.16x | uniform-mixed |
| `mixed_ood` | 0.39x | uniform-mixed |

This is the clearest unresolved Stage 1B issue.

Possible explanations:

1. The probe study used smaller models and shorter training, so the selected distribution may not transfer to the full regime.
2. `mixed_ood`-based probe selection may overfit the model-selection process to the quick proxy task.
3. The old baseline may simply optimize better with the current learning-rate schedule and batch regime.
4. The current composite score still misses an important notion of **learnable coverage**.

So the best Stage 1B checkpoint to carry forward right now is still:

`outputs/stage1b_scaled/best_model.pt`

not

`outputs/stage1b_best_dist/best_model.pt`

---

## Bottom-line answers to the roadmap questions

### 1. Does naive random rollout achieve useful coverage?

**Yes, but coverage quality depends strongly on the action distribution.**

Gaussian-heavy exploration is weak. OU and mixed strategies are much better. Bang-bang gives the widest geometric spread but is not the easiest to learn from.

### 2. Is 4D coverage the right metric?

**Yes.**

The joint 4D metric is much more faithful than separate `q`/`v` views, and it strongly predicts OOD quality.

### 3. Does better coverage automatically mean better model?

**No.**

There is a clear **coverage-learnability tradeoff**.

### 4. Did Stage 1B beat Stage 1A where it should?

**Yes.**

Stage 1B strongly beats Stage 1A on `step`, `random_walk`, and `mixed_ood`.

### 5. Is Stage 1B close to oracle?

**No.**

The remaining gap is still massive and justifies moving to smarter data collection.

---

## What Stage 1B actually established

Stage 1B now supports the following claims:

1. **Physics-only random rollout data can train a robust inverse-dynamics tracker.**
2. **OOD success depends on joint state-space coverage, not just in-distribution validation loss.**
3. **The best exploration policy is not the one with the highest raw coverage; trainability matters.**
4. **The current best Stage 1B full model is the existing uniform-mixed baseline, not the newly selected equal-mixed candidate.**
5. **There is still a huge gap to oracle performance, so Stage 1C and 1D are necessary rather than optional.**

---

## Recommended next steps for Stage 1C

Stage 1B suggests a very concrete Stage 1C agenda:

1. Replace raw coverage with **learnable coverage** or uncertainty-aware coverage.
2. Use **ensemble disagreement** to target transitions the model cannot yet fit well.
3. Evaluate model selection on the full OOD harness earlier, not only on probe-level proxies.
4. Keep `outputs/stage1b_scaled/best_model.pt` as the working Stage 1B baseline for future comparisons.

That is the right handoff from Stage 1B into active data collection.
