"""Analyze scaling experiment results from standard benchmark.

Loads benchmark JSON (from standard_benchmark.py) and produces:
  - Data scaling curves (MSE vs N_traj) for each architecture
  - Easy/hard family decomposition
  - Architecture comparison
  - Bangbang vs random comparison

Usage: .venv/bin/python scripts/scaling_analysis.py --input outputs/benchmark_v5.json
"""
import argparse, json, sys
import numpy as np
from pathlib import Path

ORACLE = {
    'multisine': 1.01e-4, 'chirp': 3.16e-4, 'sawtooth': 1.93e-4,
    'pulse': 1.29e-4, 'step': 3.25e-4, 'random_walk': 4.49e-5,
    '_aggregate': 1.85e-4,
}
EASY = ['multisine', 'chirp', 'sawtooth', 'pulse']
HARD = ['step', 'random_walk']

# Map model names to metadata
MODEL_META = {
    'random': (10000, '512x4', 'random'),
    'active': (10000, '512x4', 'active'),
    'random_20k_512x4': (20000, '512x4', 'random'),
    'random_50k_512x4': (50000, '512x4', 'random'),
    'random_100k_512x4': (100000, '512x4', 'random'),
    'hp_random_1024x6': (10000, '1024x6', 'random'),
    'random_20k_1024x6': (20000, '1024x6', 'random'),
    'random_50k_1024x6': (50000, '1024x6', 'random'),
    'random_100k_1024x6': (100000, '1024x6', 'random'),
    'bangbang_augmented_512x4': (10000, '512x4', 'bangbang'),
    'bangbang_augmented_1024x6': (10000, '1024x6', 'bangbang'),
    'supervised_1a': (10000, '512x4', 'supervised'),
}

def mse(entry):
    """Extract mean_mse from a benchmark entry."""
    if isinstance(entry, dict):
        return entry.get('mean_mse', 0)
    return entry

def analyze(data):
    """Print scaling analysis tables."""
    oracle_agg = ORACLE['_aggregate']

    # Filter to models we have metadata for
    models = {k: v for k, v in data.items() if k in MODEL_META}
    print(f"Analyzing {len(models)} models (of {len(data)} in file)\n")

    # 1. Scaling curves by architecture
    for arch in ['512x4', '1024x6']:
        print(f"=== Data Scaling: {arch} ===")
        pts = []
        for name, meta in MODEL_META.items():
            ntraj, a, strat = meta
            if a != arch or strat != 'random' or name not in data:
                continue
            agg = mse(data[name]['_aggregate'])
            easy_avg = np.mean([mse(data[name][f]) for f in EASY])
            hard_avg = np.mean([mse(data[name][f]) for f in HARD])
            pts.append((ntraj, name, agg, easy_avg, hard_avg))
        pts.sort()
        if pts:
            print(f"{'N_traj':>8} {'Model':<25} {'Agg':>10} {'xOrc':>6} "
                  f"{'Easy':>10} {'Hard':>10} {'Hard xOrc':>9}")
            print("-" * 85)
            for ntraj, name, agg, easy, hard in pts:
                print(f"{ntraj:>8} {name:<25} {agg:>10.2e} {agg/oracle_agg:>6.1f} "
                      f"{easy:>10.2e} {hard:>10.2e} "
                      f"{np.mean([mse(data[name][f])/ORACLE[f] for f in HARD]):>9.1f}")
            # Improvement ratio
            if len(pts) >= 2:
                first, last = pts[0], pts[-1]
                print(f"\n  {first[1]} → {last[1]}: "
                      f"{first[2]/last[2]:.1f}x improvement "
                      f"({first[0]//1000}K→{last[0]//1000}K data)")
        print()

    # 2. Architecture comparison at each data budget
    print("=== Architecture Comparison (512x4 vs 1024x6) ===")
    for ntraj in [10000, 20000, 50000, 100000]:
        m512 = [n for n, (nt, a, s) in MODEL_META.items()
                if nt == ntraj and a == '512x4' and s == 'random' and n in data]
        m1024 = [n for n, (nt, a, s) in MODEL_META.items()
                 if nt == ntraj and a == '1024x6' and s == 'random' and n in data]
        if m512 and m1024:
            a512 = mse(data[m512[0]]['_aggregate'])
            a1024 = mse(data[m1024[0]]['_aggregate'])
            print(f"  {ntraj//1000:>3}K: 512x4={a512:.2e} ({a512/oracle_agg:.1f}x)  "
                  f"1024x6={a1024:.2e} ({a1024/oracle_agg:.1f}x)  "
                  f"ratio={a1024/a512:.2f}")

    # 3. Bangbang vs random at 10K
    print("\n=== Bangbang vs Random (10K budget) ===")
    for arch in ['512x4', '1024x6']:
        rnd = [n for n, (nt, a, s) in MODEL_META.items()
               if nt == 10000 and a == arch and s == 'random' and n in data]
        bb = [n for n, (nt, a, s) in MODEL_META.items()
              if nt == 10000 and a == arch and s == 'bangbang' and n in data]
        if rnd and bb:
            r_agg = mse(data[rnd[0]]['_aggregate'])
            b_agg = mse(data[bb[0]]['_aggregate'])
            print(f"  {arch}: random={r_agg:.2e} ({r_agg/oracle_agg:.1f}x)  "
                  f"bangbang={b_agg:.2e} ({b_agg/oracle_agg:.1f}x)  "
                  f"ratio={b_agg/r_agg:.2f}")
            # Per-family
            for fam in HARD:
                r_f = mse(data[rnd[0]][fam])
                b_f = mse(data[bb[0]][fam])
                print(f"    {fam}: random={r_f:.2e} ({r_f/ORACLE[fam]:.1f}x)  "
                      f"bangbang={b_f:.2e} ({b_f/ORACLE[fam]:.1f}x)")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='outputs/standard_benchmark_v5.json')
    args = p.parse_args()
    with open(args.input) as f:
        data = json.load(f)
    analyze(data)

if __name__ == '__main__':
    main()
