# Stage 1C: Reproduction Commands

All commands run from repo root. All use `UV_LINK_MODE=copy uv run python`.

## Matched Random Baseline

```bash
uv run python scripts/train_random_rollout.py \
  --config configs/medium.yaml --val-data data/medium/test.h5 \
  --output-dir outputs/stage1c_random_matched \
  --n-trajectories 10000 --action-type mixed \
  --epochs 100 --batch-size 65536 --lr 1e-3 \
  --hidden-dim 512 --n-hidden 4 --seed 42 \
  --device cuda:1 --eval-trajectories 200
```

## Ensemble Disagreement

```bash
uv run python scripts/train_active.py \
  --config configs/medium.yaml --val-data data/medium/test.h5 \
  --output-dir outputs/stage1c_active_full \
  --seed-trajectories 2000 --candidate-trajectories 12000 \
  --select-trajectories 8000 --proposal-action-type mixed \
  --ensemble-size 3 --ensemble-hidden-dim 256 --ensemble-n-hidden 3 \
  --ensemble-epochs 20 --final-hidden-dim 512 --final-n-hidden 4 \
  --final-epochs 100 --batch-size 65536 --lr 1e-3 \
  --device cuda:0 --eval-trajectories 200
```

## Bin Rebalancing

```bash
uv run python scripts/train_stage1c_selector.py \
  --config configs/medium.yaml --val-data data/medium/test.h5 \
  --output-dir outputs/stage1c_rebalance_full \
  --selector rebalance_bins \
  --seed-trajectories 2000 --candidate-trajectories 12000 \
  --select-trajectories 8000 --proposal-action-type mixed \
  --final-hidden-dim 512 --final-n-hidden 4 --final-epochs 100 \
  --batch-size 65536 --lr 1e-3 --seed 42 \
  --device cuda:0 --eval-trajectories 200
```

## Low-Density / Max-Entropy Proxy

```bash
uv run python scripts/train_stage1c_selector.py \
  --config configs/medium.yaml --val-data data/medium/test.h5 \
  --output-dir outputs/stage1c_density_full \
  --selector low_density \
  --seed-trajectories 2000 --candidate-trajectories 12000 \
  --select-trajectories 8000 --proposal-action-type mixed \
  --final-hidden-dim 512 --final-n-hidden 4 --final-epochs 100 \
  --batch-size 65536 --lr 1e-3 --seed 42 \
  --device cuda:1 --eval-trajectories 200
```

## Hybrid Coverage

```bash
uv run python scripts/train_stage1c_selector.py \
  --config configs/medium.yaml --val-data data/medium/test.h5 \
  --output-dir outputs/stage1c_hybrid_full \
  --selector hybrid_coverage \
  --seed-trajectories 2000 --candidate-trajectories 12000 \
  --select-trajectories 8000 --proposal-action-type mixed \
  --final-hidden-dim 512 --final-n-hidden 4 --final-epochs 100 \
  --batch-size 65536 --lr 1e-3 --seed 42 \
  --device cuda:0 --eval-trajectories 200
```

## Hindsight Relabeling

```bash
uv run python scripts/train_hindsight.py \
  --config configs/medium.yaml --val-data data/medium/test.h5 \
  --base-model outputs/stage1b_scaled/best_model.pt \
  --output-dir outputs/stage1c_hindsight_full \
  --seed-trajectories 2000 --hindsight-trajectories 4000 \
  --reference-source mixed_ood --seed-action-type mixed \
  --hidden-dim 512 --n-hidden 4 --epochs 100 \
  --batch-size 65536 --lr 1e-3 --seed 42 \
  --device cuda:1 --eval-trajectories 200
```

## OOD Comparisons

Label mapping: `1A_supervised` = random matched, `1B_random_rollout` = variant under test, `1B_best_dist` = stage1b_scaled.

```bash
# Disagreement
uv run python scripts/eval_ood.py --config configs/medium.yaml \
  --model-1a outputs/stage1c_random_matched/best_model.pt \
  --model-1b outputs/stage1c_active_full/best_model.pt \
  --model-best outputs/stage1b_scaled/best_model.pt \
  --n-trajectories 300 --eval-trajectories 200 \
  --output-dir outputs/stage1c_ood_compare

# Rebalance
uv run python scripts/eval_ood.py --config configs/medium.yaml \
  --model-1a outputs/stage1c_random_matched/best_model.pt \
  --model-1b outputs/stage1c_rebalance_full/best_model.pt \
  --model-best outputs/stage1b_scaled/best_model.pt \
  --n-trajectories 300 --eval-trajectories 200 \
  --output-dir outputs/stage1c_ood_rebalance

# Density
uv run python scripts/eval_ood.py --config configs/medium.yaml \
  --model-1a outputs/stage1c_random_matched/best_model.pt \
  --model-1b outputs/stage1c_density_full/best_model.pt \
  --model-best outputs/stage1b_scaled/best_model.pt \
  --n-trajectories 300 --eval-trajectories 200 \
  --output-dir outputs/stage1c_ood_density

# Hybrid
uv run python scripts/eval_ood.py --config configs/medium.yaml \
  --model-1a outputs/stage1c_random_matched/best_model.pt \
  --model-1b outputs/stage1c_hybrid_full/best_model.pt \
  --model-best outputs/stage1b_scaled/best_model.pt \
  --n-trajectories 300 --eval-trajectories 200 \
  --output-dir outputs/stage1c_ood_hybrid

# Hindsight
uv run python scripts/eval_ood.py --config configs/medium.yaml \
  --model-1a outputs/stage1c_random_matched/best_model.pt \
  --model-1b outputs/stage1c_hindsight_full/best_model.pt \
  --model-best outputs/stage1b_scaled/best_model.pt \
  --n-trajectories 300 --eval-trajectories 200 \
  --output-dir outputs/stage1c_ood_hindsight
```
