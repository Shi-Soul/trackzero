# TRACK-ZERO Usage Guide

## 1. Environment Setup

This repo uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# First-time setup — creates .venv and installs all dependencies
cd /path/to/trackzero
uv sync --extra dev

# Run any script inside the managed environment
uv run python scripts/generate_dataset.py --help

# Or activate the venv manually
source .venv/bin/activate
python scripts/generate_dataset.py --help
```

> **Note:** `uv.lock` is committed to the repo so everyone gets the same resolved versions.

---

## 2. Core Scripts (Stage 0 / Stage 1)

### 2.1 Generate the reference dataset

```bash
uv run python scripts/generate_dataset.py \
    --config configs/default.yaml \
    --output-dir data \
    --split both \          # 'train', 'test', or 'both'
    --seed 42 \
    --workers 8             # parallel workers (0 = serial)
```

Produces `data/train.h5` and `data/test.h5`.

---

### 2.2 Verify the inverse dynamics oracle

这个是per step 的 action prediction error, 不是dynamics rollout error

形式: (state, qacc) -> action <-> error

```bash
uv run python scripts/verify_oracle.py \
    --config configs/default.yaml \
    --dataset data/test.h5 \
    --n-trajectories 200 \
    --save-plot outputs/oracle_verification.png
```

Expected output: `Max |error| < 1e-10` (machine-epsilon accuracy via Newton shooting).

---

### 2.3 Evaluate a policy on the test set

```bash
# Oracle policy (performance ceiling)
uv run python scripts/run_eval.py \
    --dataset data/test.h5 \
    --policy oracle \
    --oracle-mode shooting \
    --output outputs/oracle_eval.json

# Zero-torque policy (lower bound)
uv run python scripts/run_eval.py \
    --dataset data/test.h5 \
    --policy zero \
    --output outputs/zero_eval.json

# Trained model (after training)
uv run python scripts/run_eval.py \
    --dataset data/test.h5 \
    --policy supervised \
    --checkpoint outputs/stage1a/best_model.pt \
    --output outputs/stage1a_eval.json
```

---

### 2.4 Train Stage 1A — supervised baseline

```bash
uv run python scripts/train_supervised.py \
    --config configs/default.yaml \
    --train-data data/train.h5 \
    --val-data data/test.h5 \
    --output-dir outputs/stage1a \
    --epochs 20 \
    --hidden-dim 256 \
    --n-hidden 3
```

Saves `outputs/stage1a/best_model.pt` and `outputs/stage1a/eval_results.json`.

---

### 2.5 Train Stage 1B — random rollout (TRACK-ZERO v0)

```bash
uv run python scripts/train_random_rollout.py \
    --config configs/default.yaml \
    --val-data data/test.h5 \
    --output-dir outputs/stage1b \
    --n-trajectories 80000 \
    --action-type mixed \     # uniform | gaussian | ou | bangbang | multisine | mixed
    --epochs 40 \
    --hidden-dim 512 \
    --n-hidden 4
```

---

### 2.6 Evaluate on out-of-distribution references

```bash
uv run python scripts/eval_ood.py \
    --config configs/default.yaml \
    --model-1a outputs/stage1a/best_model.pt \
    --model-1b outputs/stage1b/best_model.pt \
    --n-trajectories 500 \
    --output-dir outputs/ood_eval
```

Compares Stage 1A vs 1B on chirp, step, random-walk, sawtooth, pulse reference types.

---

### 2.7 Ablation study (Stage 1E)

```bash
# Ablation: model capacity (128x2 / 256x3 / 512x4 / 1024x5)
uv run python scripts/ablation_study.py --ablation capacity

# Ablation: dataset size (5k / 10k / 20k / 40k / 80k trajectories)
uv run python scripts/ablation_study.py --ablation datasize

# Ablation: action distribution type
uv run python scripts/ablation_study.py --ablation action_type
```

---

### 2.8 Visualize dataset coverage and evaluation results

```bash
# Coverage heatmaps
uv run python scripts/visualize.py \
    --dataset data/train.h5 \
    --output-dir outputs/viz

# Error distribution from eval JSON
uv run python scripts/visualize.py \
    --eval-results outputs/stage1a/eval_results.json \
    --output-dir outputs/viz
```

---

## 3. Running the Test Suite

```bash
uv run pytest -q                          # all tests
uv run pytest tests/test_simulator.py     # just the simulator tests
uv run pytest tests/test_oracle.py        # oracle round-trip tests
uv run pytest -k "not gpu"               # skip GPU tests
```

Expected baseline (after Stage 0 is implemented):
- `test_simulator.py` — all pass
- `test_oracle.py` — all pass
- `test_harness.py` — all pass
- `test_mlp.py` — all pass
- `test_dataset.py` — all pass (requires `trackzero.data`)
- `test_multisine.py` — all pass (requires `trackzero.data`)
- `test_ood_references.py` — all pass (requires `trackzero.data`)
- `test_gpu_simulator.py` — requires `mujoco_warp` (not in default deps)

---

## 4. GPU Simulation (Optional)

The `GPUSimulator` and `--gpu-sim` flags use [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp).
This is **not** installed by default. To enable it:

```bash
pip install mujoco-warp  # follow MuJoCo Warp install instructions
```

Then pass `--gpu-sim` and `--gpu-device cuda:0` to any training script that supports it.

---

## 5. Stage 0 Completion Checklist

| Check | Script | Pass criterion |
| --- | --- | --- |
| Oracle near-zero error | `verify_oracle.py` | Max \|error\| < 1e-6 |
| Open-loop replay determinism | `run_eval.py --openloop` | MSE_total < 1e-15 |
| Dataset state-space coverage | `visualize.py` | q_coverage ≥ 80% |
| Eval harness reproducible | `run_eval.py` | Same metrics across runs |
