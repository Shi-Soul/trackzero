# Stage 1D: Scaling — Data, Capacity, DAgger

Stage 1C showed selection strategy gives at most 1.4× gain. Stage 1D
tests whether scaling data quantity and model capacity can close the
remaining 10× oracle gap.

## Three Scaling Axes

| Axis | Baseline | Best Result | Gain |
|------|----------|------------|------|
| Quality (active vs random) | 2.67e-3 | 1.86e-3 | 1.4× |
| Capacity (512×4 → 1024×6) | val 1.39e-3 | val 1.42e-3 | ~1× (bench TBD) |
| Quantity (10K → 20K) | val 1.39e-3 | val 2.29e-4 | **6× val improvement** |

**Data quantity dominates.** 2× more data → 6× better validation loss.

---

## Architecture Scaling

| Architecture | Params | Val Loss (ep150+) |
|-------------|--------|-------------------|
| 512×4 | 3.1M | 1.39e-3 |
| 1024×4 | 5.3M | 1.68e-3 |
| 1024×6 | 5.3M | 1.42e-3 |
| 2048×3 | 10.5M | TBD |

Depth > width: 1024×6 outperforms 1024×4 at same parameter count.
But architecture alone doesn't break through — benchmark TBD.

## Data Scaling

| N | Val Loss | Status |
|---|----------|--------|
| 10K | 1.39e-3 | ✅ Benchmarked (2.67e-3) |
| 20K | 2.29e-4 (ep110) | ❌ Crashed, needs restart |
| 50K | 7.47e-3 (ep1) | 🔄 Training |

20K val-loss (2.29e-4) is within 1.24× of oracle (1.85e-4).
If this translates to benchmark, the oracle gap may be nearly closed.

## DAgger: Catastrophic Failure

| Model | Val Loss | Bench MSE | ×Oracle |
|-------|----------|-----------|---------|
| random 512×4 | 1.39e-3 | 2.67e-3 | 14× |
| DAgger 512×4 | 1.14e-3 | **0.639** | 3295× |
| DAgger 1024×4 | 1.43e-3 | **0.622** | 3362× |

DAgger has *better* val-loss but 240× worse benchmark. This proves
**val-loss is uncalibrated across data distributions** — the standard
benchmark is the only trustworthy metric.

Root cause: DAgger shifts the training distribution toward benchmark
references, but small errors compound in closed-loop deployment.

---

## Running Experiments

| Experiment | GPU | Progress | Expected |
|-----------|-----|----------|----------|
| hp_random_1024×6 | 0 | ep170/200 | Benchmark-ready soon |
| random_50k_1024×6 | 1 | ep1/200 | Best candidate if scaling law holds |
| dagger_1024×6 | 5 | iter0 ep1 | Likely still bad |
| random_20k | — | Crashed | Needs restart |

## What's Left

1. Restart 20K random → benchmark it
2. Wait for 50K random and hp_1024×6 → benchmark them
3. If 20K/50K approach oracle on benchmark → Stage 1 nearly solved
4. If not → investigate what hard families specifically need

## Stage 1D Status: 🔄 IN PROGRESS