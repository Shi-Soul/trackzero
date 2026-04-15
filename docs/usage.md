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

## 2. Core Scripts

### 2.1 Generate the reference dataset

```bash
uv run python scripts/generate_dataset.py \
    --config configs/default.yaml \
    --output-dir data/medium \
    --split both \
    --seed 42
```

Produces `data/medium/train.h5` and `data/medium/test.h5`.

---

### 2.2 Verify the inverse dynamics oracle

Per-step action prediction error: `(state, next_state) -> torque` via Newton shooting.

```bash
uv run python scripts/verify_oracle.py \
    --config configs/default.yaml \
    --dataset data/medium/test.h5
```

Expected: velocity residual < 1e-10 after Newton convergence.

---

### 2.3 Run the standard benchmark

6-family benchmark (multisine, chirp, step, random_walk, sawtooth, pulse),
100 trajectories each = 600 total tracking tasks.

```bash
uv run python scripts/standard_benchmark.py \
    --model-dir outputs/arch_1024x6_wd0.0001_lr_cosine_10k \
    --device cuda:0
```

Reports per-family MSE and aggregate (AGG) score.

---

### 2.4 Train — supervised baseline (Stage 1A)

```bash
uv run python scripts/train_supervised.py \
    --config configs/default.yaml \
    --train-data data/medium/train.h5 \
    --val-data data/medium/test.h5 \
    --output-dir outputs/stage1a \
    --epochs 200 \
    --hidden-dim 1024 \
    --n-hidden 6
```

Saves `outputs/stage1a/best_model.pt`.

---

### 2.5 Train — random rollout (Stage 1B / 1D best config)

```bash
uv run python scripts/train_random_rollout.py \
    --config configs/default.yaml \
    --val-data data/medium/test.h5 \
    --output-dir outputs/stage1b \
    --n-trajectories 10000 \
    --action-type mixed \
    --epochs 200 \
    --hidden-dim 1024 \
    --n-hidden 6 \
    --weight-decay 1e-4 \
    --lr-schedule cosine
```

`--action-type` choices: `uniform | gaussian | ou | bangbang | multisine | mixed`.
**Best config (Stage 1D)**: 1024×6, cosine LR, WD=1e-4, 10K–50K trajectories.

---

### 2.6 Train — structured architectures (Stage 2A / 3C)

Compares raw_mlp, residual_pd, and factored across DOF:

```bash
# 2-link double pendulum
uv run python -m scripts.run_structured --n-links 2 --arch factored --device cuda:0

# 3-link chain
uv run python -m scripts.run_structured --n-links 3 --arch factored --device cuda:1

# 5-link chain
uv run python -m scripts.run_structured --n-links 5 --arch factored --device cuda:2
```

Available archs: `raw_mlp | residual_pd | factored`.

---

### 2.7 Stage 2 robustness experiments

```bash
# 2A: noise degradation sweep (MLP vs oracle under σ=0..0.5)
uv run python scripts/run_stage2_degradation.py \
    --model-dir outputs/arch_1024x6_wd0.0001_lr_cosine_10k

# 2C: noise-augmented training
uv run python scripts/run_stage2c_robust.py \
    --noise-std 0.05 --noise-frac 0.5 --n-traj 10000

# Residual oracle-input architecture
uv run python scripts/run_residual.py --n-traj 10000 --epochs 200
```

---

### 2.8 N-link chain experiment (Stage 3A)

```bash
uv run python scripts/run_chain_experiment.py \
    --n-links 3 --n-traj 2000 --epochs 200 --device cuda:0
```

---

### 2.9 Ablation and scaling

```bash
# Architecture ablation (raw_mlp vs residual_pd vs factored)
uv run python scripts/run_arch_comparison.py

# Data scaling (10K / 20K / 50K / 100K)
uv run python scripts/train_20k_random.py --device cuda:0
```

---

### 2.10 Humanoid (Stage 4)

```bash
# Baseline: random torque data
CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.run_humanoid

# Stage 1C entropy: diverse torque patterns vs diverse initial states
CUDA_VISIBLE_DEVICES=1 uv run python -m scripts.run_humanoid_entropy

# Finding 24: diversity vs pattern matching ablation
CUDA_VISIBLE_DEVICES=2 uv run python -m scripts.run_humanoid_finding24
```

Results are saved to `outputs/humanoid_*/results.json` and logs to
`outputs/log_humanoid*.txt`.

---

### 2.11 Simulation visualization (GIF output)

Produces closed-loop rollout GIFs for all trained models across all
physical bodies and reference types:

```bash
uv run python scripts/visualize_sim.py
```

Outputs to `outputs/viz_sim/`:

| File | Content |
|------|---------|
| `2dof_step.gif` | All 2-DOF policies vs step reference |
| `2dof_chirp.gif` | All 2-DOF policies vs chirp reference |
| `2dof_random_walk.gif` | All 2-DOF policies vs random-walk reference |
| `chain3_step.gif` | 3-link chain policy vs step reference |
| `chain3_chirp.gif` | 3-link chain policy vs chirp reference |
| `chain5_step.gif` | 5-link chain policy vs step reference |
| `chain5_chirp.gif` | 5-link chain policy vs chirp reference |

Policies visualized (2-DOF): oracle, stage1a, stage1b, stage1d_best,
stage1d_50k, stage2c_noisy, residual (oracle-augmented input).


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

## 5. Key Benchmarks and Pass Criteria

| Check | Script | Pass criterion |
| --- | --- | --- |
| Oracle Newton shooting | `verify_oracle.py` | velocity residual < 1e-10 |
| Standard benchmark (double pendulum) | `standard_benchmark.py` | AGG reported |
| Best Stage 1D (2 DOF) | `standard_benchmark.py` | AGG ≈ 6.8e-4 |
| Factored arch (3-link) | `run_structured.py` | AGG ≈ 3.3e-3 |
| Factored arch (5-link) | `run_structured.py` | AGG ≈ 1.9e-2 |
