# Stage 1C: Active Data Collection via Ensemble Disagreement

## Executive summary

This first Stage 1C pass implemented an **ensemble-disagreement active collector** and compared it against a matched random-rollout baseline.

The result is negative:

1. **The active collector worked technically**: ensemble training, disagreement scoring, trajectory selection, final training, and evaluation all ran end-to-end.
2. **It did not improve model quality**.
3. On held-out multisine test trajectories, the active model was worse than the matched random baseline:
   - random matched: `1.185e-4` mean `MSE_total`
   - active collection: `1.662e-4`
4. On OOD references, the active model also failed to beat the matched random baseline, and usually lost badly on the harder families.

So the current Stage 1C conclusion is:

> **naive trajectory-level ensemble disagreement is not yet a win over strong random-rollout training.**

---

## What was implemented

New code:

| File | Role |
| --- | --- |
| `trackzero/data/active_collection.py` | Bootstrap ensemble training helpers, disagreement scoring, top-trajectory selection |
| `scripts/train_active.py` | End-to-end Stage 1C training entrypoint |

### Active-collection algorithm

The implemented pipeline is:

1. Generate a **seed dataset** from random rollouts.
2. Train an **ensemble** of inverse-dynamics models on bootstrap resamples of that seed data.
3. Generate a larger **candidate pool** of rollout trajectories.
4. Score each candidate trajectory by **ensemble action-prediction variance** over all transitions in that trajectory.
5. Keep the highest-disagreement trajectories.
6. Train the final Stage 1C model on:
   - seed trajectories
   - selected high-disagreement trajectories

The disagreement score is computed from variance of predicted torques for the same `(current_state, next_state)` inputs.

---

## Exact commands and artifacts

### 1. Pilot sanity check

```bash
uv run python scripts/train_active.py \
  --config configs/medium.yaml \
  --val-data data/medium/test.h5 \
  --output-dir outputs/stage1c_pilot \
  --seed-trajectories 200 \
  --candidate-trajectories 400 \
  --select-trajectories 200 \
  --proposal-action-type mixed \
  --ensemble-size 2 \
  --ensemble-hidden-dim 128 \
  --ensemble-n-hidden 2 \
  --ensemble-epochs 2 \
  --final-hidden-dim 128 \
  --final-n-hidden 2 \
  --final-epochs 2 \
  --batch-size 8192 \
  --lr 1e-3 \
  --device cuda:0 \
  --eval-trajectories 20
```

Artifacts:

- `outputs/stage1c_pilot/metadata.json`
- `outputs/stage1c_pilot/eval_results.json`

This run only validated the pipeline.

### 2. Full active run

```bash
uv run python scripts/train_active.py \
  --config configs/medium.yaml \
  --val-data data/medium/test.h5 \
  --output-dir outputs/stage1c_active_full \
  --seed-trajectories 2000 \
  --candidate-trajectories 12000 \
  --select-trajectories 8000 \
  --proposal-action-type mixed \
  --ensemble-size 3 \
  --ensemble-hidden-dim 256 \
  --ensemble-n-hidden 3 \
  --ensemble-epochs 20 \
  --final-hidden-dim 512 \
  --final-n-hidden 4 \
  --final-epochs 100 \
  --batch-size 65536 \
  --lr 1e-3 \
  --device cuda:0 \
  --eval-trajectories 200
```

Artifacts:

- `outputs/stage1c_active_full/metadata.json`
- `outputs/stage1c_active_full/eval_results.json`
- `outputs/stage1c_active_full/ensemble/member_*/best_model.pt`

### 3. Matched random baseline

```bash
uv run python scripts/train_random_rollout.py \
  --config configs/medium.yaml \
  --val-data data/medium/test.h5 \
  --output-dir outputs/stage1c_random_matched \
  --n-trajectories 10000 \
  --action-type mixed \
  --epochs 100 \
  --batch-size 65536 \
  --lr 1e-3 \
  --hidden-dim 512 \
  --n-hidden 4 \
  --seed 42 \
  --device cuda:1 \
  --eval-trajectories 200
```

Artifacts:

- `outputs/stage1c_random_matched/metadata.json`
- `outputs/stage1c_random_matched/eval_results.json`

### 4. OOD comparison

```bash
uv run python scripts/eval_ood.py \
  --config configs/medium.yaml \
  --model-1a outputs/stage1c_random_matched/best_model.pt \
  --model-1b outputs/stage1c_active_full/best_model.pt \
  --model-best outputs/stage1b_scaled/best_model.pt \
  --n-trajectories 300 \
  --eval-trajectories 200 \
  --output-dir outputs/stage1c_ood_compare
```

Label mapping in this run:

| Eval label in JSON | Actual model |
| --- | --- |
| `1A_supervised` | `outputs/stage1c_random_matched/best_model.pt` |
| `1B_random_rollout` | `outputs/stage1c_active_full/best_model.pt` |
| `1B_best_dist` | `outputs/stage1b_scaled/best_model.pt` |

Artifacts:

- `outputs/stage1c_ood_compare/ood_comparison.json`
- `outputs/stage1c_ood_compare/*_details.json`

---

## Active-selection statistics

From `outputs/stage1c_active_full/metadata.json`:

| Field | Value |
| --- | ---: |
| seed trajectories | 2000 |
| candidate trajectories | 12000 |
| selected trajectories | 8000 |
| ensemble size | 3 |
| ensemble architecture | `256 x 3` |
| final architecture | `512 x 4` |
| ensemble best val losses | 1.600, 1.548, 1.633 |
| candidate mean disagreement score | 0.0785 |
| selected mean disagreement score | 0.1047 |
| selected minimum score | 0.0379 |
| selected maximum score | 4.5131 |

So the selector did what it was supposed to do:

- it **found measurably higher-disagreement trajectories**
- it **biased the final dataset toward them**

The failure is therefore not a bug in ranking; it is a failure of the current **selection criterion** to improve the final policy.

---

## ID results: active vs matched random

Held-out multisine evaluation, from:

- `outputs/stage1c_random_matched/eval_results.json`
- `outputs/stage1c_active_full/eval_results.json`

| Model | Mean `MSE_q` | Mean `MSE_v` | Mean `MSE_total` | Max `MSE_total` | Mean max `q` error |
| --- | ---: | ---: | ---: | ---: | ---: |
| random matched | 6.489e-05 | 5.366e-04 | **1.185e-04** | 2.156e-03 | 1.740e-02 |
| active full | 9.858e-05 | 6.765e-04 | **1.662e-04** | 3.807e-03 | 2.051e-02 |

### Interpretation

The active collector is already worse on the easiest evaluation target:

- worse mean error
- worse tail error
- worse peak angle error

So Stage 1C must justify itself through **OOD gains**. It did not.

---

## OOD comparison

Using `outputs/stage1c_ood_compare/ood_comparison.json`:

| OOD type | random matched | active full | old Stage 1B baseline | best |
| --- | ---: | ---: | ---: | --- |
| `chirp` | **2.156e-04** | 3.012e-04 | 3.039e-04 | random matched |
| `step` | 3.341e-02 | **3.338e-02** | 4.710e-02 | active ~= random |
| `random_walk` | 1.744e-02 | 2.332e-02 | **1.379e-02** | old Stage 1B |
| `sawtooth` | **1.557e-04** | 1.996e-04 | 2.369e-04 | random matched |
| `pulse` | 8.263e-05 | 1.229e-04 | **7.371e-05** | old Stage 1B |
| `mixed_ood` | **4.830e-03** | 1.361e-02 | 9.691e-03 | random matched |

### Ratios: active vs matched random

| OOD type | `random / active` | Conclusion |
| --- | ---: | --- |
| `chirp` | 0.72x | active worse |
| `step` | 1.00x | tie |
| `random_walk` | 0.75x | active worse |
| `sawtooth` | 0.78x | active worse |
| `pulse` | 0.67x | active worse |
| `mixed_ood` | 0.35x | active much worse |

### Interpretation

This first Stage 1C attempt does **not** beat the matched random baseline on any OOD family in a meaningful way.

At best:

- it ties on `step`

But on the broader OOD families that matter most:

- it loses on `random_walk`
- it loses on `pulse`
- it loses badly on `mixed_ood`

That means the currently implemented disagreement score is not selecting **useful** training data; it is only selecting **uncertain** data.

---

## Why this likely failed

The most plausible explanations are:

1. **Trajectory-level disagreement is too blunt.** High-disagreement trajectories may contain many chaotic or low-value transitions, not concentrated signal.
2. **Bootstrap ensemble uncertainty is poorly calibrated early.** The ensemble was trained on only 2000 seed trajectories before ranking 12000 candidates.
3. **The selector ignores learnability.** It prefers transitions the current ensemble disagrees on, but some of those may be intrinsically noisy or simply hard to fit with the MLP.
4. **The proposal distribution is unchanged.** The method only reweights trajectories sampled from the same mixed rollout process; it does not truly expand the reachable frontier.
5. **Selection is trajectory-level, not transition-level.** A few high-variance steps can pull in an entire trajectory that is otherwise unhelpful.

---

## What Stage 1C established

This is still a useful research result.

Stage 1C now establishes that:

1. **A basic ensemble-disagreement collector is easy to integrate into the current codebase.**
2. **Disagreement alone is not enough.**
3. **The strong random-rollout baseline from Stage 1B remains hard to beat.**
4. **Future Stage 1C work should target uncertainty-aware but also learnable or coverage-improving transitions, not just raw ensemble variance.**

---

## Recommended next fixes

The next Stage 1C iteration should probably try one or more of:

1. **Transition-level selection** instead of whole-trajectory selection.
2. **Disagreement + coverage hybrid scoring** so selected data is both novel and broad.
3. **Disagreement + loss prediction / learnability penalty** to avoid selecting only pathological transitions.
4. **Iterative active rounds** instead of one-shot seed → rank → retrain.
5. **Hindsight relabeling (Stage 1D)**, which may produce more policy-relevant data than passive rollout filtering.

For now, the best model to carry forward is still:

`outputs/stage1c_random_matched/best_model.pt`

and more broadly, the Stage 1B-style random-rollout recipe remains stronger than this first Stage 1C active variant.
