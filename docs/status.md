# Experiment Status

## Critical Bugfix (Apr 13)

All previous scaling experiments had a **tau_max normalization bug**:
training targets were divided by tau_max=5, but MLPPolicy expects
raw torques. Models applied 1/5 the correct torque → ~5000× oracle.
Bug fixed in commit 1188bfa. All experiments relaunched from scratch.

## Active Training (7 GPUs, post-bugfix)

| GPU | Experiment | Arch | Data | Status |
|-----|-----------|------|------|--------|
| 1 | 20K random | 1024x6 | 20K | training |
| 2 | 20K random | 512x4 | 20K | training |
| 3 | 50K random | 1024x6 | 50K | data gen |
| 4 | 50K random | 512x4 | 50K | data gen |
| 5 | 100K random | 1024x6 | 100K | data gen |
| 6 | 100K random | 512x4 | 100K | data gen |
| 7 | bangbang aug | 512x4 | 10K | training |

GPU 0 unavailable (external processes). ETA: ~9-12 hours.

## Experiment Matrix

|  | 512x4 (1.1M) | 1024x6 (5.3M) |
|--|--------------|----------------|
| 10K random | 14× oracle (done) | 3300× oracle (done) |
| 20K random | GPU 2 | GPU 1 |
| 50K random | GPU 4 | GPU 3 |
| 100K random | GPU 6 | GPU 5 |
| 10K bangbang | GPU 7 | — |

## Key Questions This Matrix Answers

1. **Data scaling**: does 10K→100K close the 10× oracle gap?
2. **Architecture recovery**: at what N does 1024×6 stop overfitting?
3. **Targeted coverage**: does bangbang (velocity-biased) beat
   random at equal 10K budget?
