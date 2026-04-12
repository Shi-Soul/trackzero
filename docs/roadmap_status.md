# TRACK-ZERO Roadmap Status

_Last updated after Stage 1C completion + Stage 1D.1 reachability results._

## Current Status

**Stage 0–1C complete. Stage 1D in progress.**

- **Stage 0 ✅**: Data pipeline, oracle, training loop, evaluation harness.
- **Stage 1A ✅**: Supervised ID MLP (mean_mse=1.22e-4).
- **Stage 1B ✅**: Random-rollout self-supervised; coverage/scaling/OOD analysis.
- **Stage 1C ✅**: 13 experiments exploring coverage strategies. **Goldilocks Principle** discovered — random rollouts are near-optimal. See `docs/stage_1c.md`.
- **Stage 1D 🧪**: Reachability (1D.1) complete — catastrophic failure. DAgger and trajectory optimization next.
- **Stage 1E**: Ablation scripts exist; not complete at full scale.
- **Stage 2–4**: Not yet implemented.

## Progress Estimate

| Scope | Estimate | Notes |
|-------|---------|-------|
| **Overall (Stage 0-4)** | **~45%** | Stage 1 nearly complete |
| **Stage 0** | **100%** | Full pipeline + baseline |
| **Stage 1A** | **100%** | Supervised baseline |
| **Stage 1B** | **100%** | Coverage, ablations, OOD |
| **Stage 1C** | **100%** | 13 experiments, Goldilocks principle |
| **Stage 1D** | **40%** | 1D.1 done (negative), 1D.2/1D.3/DAgger pending |
| **Stage 1E** | **50%** | Ablation scripts exist |
| **Stage 2–4** | **0%** | Not started |

## Stage 1C Key Finding: The Goldilocks Principle

| Category | Method | Ratio vs Random |
|---|---|---:|
| Ergodic (BEST) | random_matched | 1.0× |
| Mild perturbation | hybrid/density/active/rebalance/adversarial | 1.2–2.4× |
| Too broad (RL) | maxent_rl | 82.6× |
| Too narrow | restricted_v | 172× |
| No coherence | reachability | 1,017–4,314× |

Random rollouts produce naturally optimal training distribution.
Both narrowing and broadening catastrophically degrade performance.
