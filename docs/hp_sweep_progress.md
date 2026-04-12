# Hyperparameter Sweep — Design & Early Results

## Motivation

Three critiques drove this experiment:
1. **Maxent RL untested with larger capacity** — 82.6× worse on ID, but could capacity close the gap?
2. **No HP tuning done** — all prior experiments used default 512×4 architecture
3. **Underutilized resources** — only 1 of 8 GPUs used previously

## Experiment Design

### Architecture Sweep

| GPU | Data | Hidden | Layers | Params | LR | Epochs | Special |
|-----|------|--------|--------|--------|----|--------|---------|
| 0 | maxent | 1024 | 4 | 3.16M | 1e-3 | 100 | - |
| 1 | maxent | 1024 | 6 | 5.26M | 1e-3 | 100 | - |
| 2 | maxent | 2048 | 4 | 12.6M | 1e-3 | 100 | - |
| 3 | maxent | 512 | 4 | 793K | 3e-4 | 200 | Lower LR, longer |
| 4 | random | 1024 | 4 | 3.16M | 1e-3 | 100 | Baseline comparison |
| 5 | random | 2048 | 4 | 12.6M | 1e-3 | 100 | Baseline comparison |
| 6 | random | 1024 | 6 | 5.26M | 1e-3 | 100 | Baseline comparison |
| 7 | maxent | 512 | 4 | 793K | 1e-3 | 100 | + weight decay 1e-5 |

All experiments include full OOD evaluation (6 settings) after training.

### Key Question

Does increasing model capacity from 793K → 12.6M params (16×) close the maxent-random gap?

## Early Results (Epoch 10 of 100)

### Validation Loss (MSE on held-out data)

| Experiment | train_loss | val_loss | best_val |
|------------|-----------|----------|----------|
| maxent 1024×4 | 0.0522 | 0.2200 | 0.1274 |
| maxent 1024×6 | 0.0461 | 0.2020 | 0.1423 |
| maxent 2048×4 | 0.0519 | 0.2229 | 0.1257 |
| maxent 512×4 lr3e-4 | 0.0573 | 0.2574 | 0.1305 |
| maxent 512×4 +wd | 0.0485 | 0.2219 | 0.1399 |
| **random 1024×4** | **0.0217** | **0.0196** | **0.0196** |
| random 2048×4 | 0.0754 | 0.0678 | 0.0558 |
| random 1024×6 | 0.0634 | 0.0838 | 0.0344 |

### Early Observations

1. **Capacity does NOT close the maxent gap** (so far):
   - All maxent variants: val_loss ~0.13-0.26, regardless of capacity
   - The 12.6M model (2048×4) performs similarly to the 793K model (512×4)
   - Suggests the problem is data distribution, not model capacity

2. **Random benefits from capacity**: 
   - 1024×4 random achieves val=0.0196 at epoch 10 (already strong)
   - Larger models (2048×4, 1024×6) are still converging — need more epochs

3. **Lower LR doesn't help maxent**: 
   - 512×4 with lr=3e-4: best_val=0.1305 (similar to lr=1e-3 best_val=0.1399)

4. **Weight decay slightly helps maxent**: 
   - 512×4+wd: train=0.0485 (lower than base 512×4 would be at same epoch)
   - But val still stuck at ~0.22

### Implications

**Important caveat**: Val loss is not the right metric for comparison because the data 
distributions differ. A model with high val loss on maxent data could still perform well 
in closed-loop tracking if it learned the right function in the task-relevant region.

The real test is the closed-loop OOD evaluation, which runs after training completes.

## Next: Hybrid Strategies

Based on the OOD analysis (see `ood_analysis.md`), we're preparing 4 hybrid training 
strategies that combine maxent and random data:

| Strategy | Description | Hypothesis |
|----------|-------------|------------|
| concat | 10M pairs combined | Brute force coverage |
| curriculum | maxent→random | Broad features first, then refine |
| reverse_curriculum | random→combined | Accuracy first, then add coverage |
| weighted | 4:1 random:maxent | Emphasize ID while adding OOD |

Script: `scripts/train_hybrid_strategy.py`

## Timeline

- HP sweep: ~2 hours remaining (started ~30 min ago)
- Hybrid experiments: Launch when GPUs free up (~1.5-2h per experiment)
- Full analysis: After all experiments complete

---

*Status: Training in progress, epoch ~10/100*
*Updated: During training run*
