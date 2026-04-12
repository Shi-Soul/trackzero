#!/usr/bin/env python3
"""Generate unified results table from all benchmark data.

Collects results from:
  - outputs/standard_benchmark.json (original 11 models)
  - outputs/oracle_benchmark.json (FD oracle lower bound)
  - outputs/hp_dagger_benchmark.json (auto-benchmark results)
  - outputs/dagger_benchmark_*/results.json (DAgger iteration results)
  - outputs/random_20k_1024x6/benchmark.json (20K scaling experiment)

Produces a single sorted table and key summary statistics.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
os.chdir(ROOT)

families = ['multisine', 'chirp', 'step', 'random_walk', 'sawtooth', 'pulse']
families_easy = ['multisine', 'chirp', 'sawtooth', 'pulse']
families_hard = ['step', 'random_walk']


def load_oracle():
    with open('outputs/oracle_benchmark.json') as f:
        d = json.load(f)
    agg = d.get('_aggregate', np.mean([d[f] for f in families if f in d]))
    return agg, d


def extract_mse(data, model_name):
    """Extract aggregate, step, rw, easy MSE from various JSON formats."""
    agg_raw = data.get('_aggregate', None)
    if agg_raw is None:
        return None
    if isinstance(agg_raw, dict):
        agg = agg_raw.get('mean_mse', 0)
    else:
        agg = float(agg_raw)

    def get_family(f):
        v = data.get(f, 0)
        if isinstance(v, dict):
            return v.get('mean_mse', 0)
        return float(v) if v else 0

    step = get_family('step')
    rw = get_family('random_walk')
    easy = np.mean([get_family(f) for f in families_easy])
    return {'name': model_name, 'agg': agg, 'step': step, 'rw': rw, 'easy': easy}


def collect_all():
    results = []
    oracle_agg, oracle_data = load_oracle()

    # 1. Standard benchmark (original models)
    if os.path.exists('outputs/standard_benchmark.json'):
        with open('outputs/standard_benchmark.json') as f:
            std = json.load(f)
        for name, data in std.items():
            r = extract_mse(data, name)
            if r:
                r['source'] = 'stage1'
                results.append(r)

    # 2. HP/DAgger auto-benchmark
    if os.path.exists('outputs/hp_dagger_benchmark.json'):
        with open('outputs/hp_dagger_benchmark.json') as f:
            hp = json.load(f)
        for name, data in hp.items():
            if name.startswith('_'):
                continue
            r = extract_mse(data, name)
            if r:
                r['source'] = 'hp_sweep'
                results.append(r)

    # 3. DAgger iteration results
    for dagger_dir in ['dagger_benchmark_512x4', 'dagger_benchmark_1024x4']:
        results_file = f'outputs/{dagger_dir}/results.json'
        if os.path.exists(results_file):
            with open(results_file) as f:
                dres = json.load(f)
            for iter_key, iter_data in dres.items():
                if 'benchmark' in iter_data:
                    bm = iter_data['benchmark']
                    r = extract_mse(bm, f'{dagger_dir}_{iter_key}')
                    if r:
                        r['source'] = 'dagger'
                        results.append(r)

    # 4. 20K random results
    for p in ['outputs/random_20k_1024x6/benchmark.json',
              'outputs/random_20k_1024x6/quick_benchmark.json']:
        if os.path.exists(p):
            with open(p) as f:
                data = json.load(f)
            r = extract_mse(data, 'random_20k_1024x6')
            if r:
                r['source'] = 'scaling'
                results.append(r)
            break

    # Deduplicate (prefer latest source)
    seen = {}
    for r in results:
        seen[r['name']] = r
    results = list(seen.values())

    # Add gap to oracle
    for r in results:
        r['gap'] = r['agg'] / oracle_agg

    results.sort(key=lambda r: r['agg'])
    return results, oracle_agg


def main():
    results, oracle_agg = collect_all()

    print("=" * 95)
    print("TRACK-ZERO UNIFIED BENCHMARK LEADERBOARD")
    print(f"Oracle floor (FD): {oracle_agg:.4e}")
    print("=" * 95)
    print(f"{'#':>3} {'Model':>28} {'Source':>8} {'Agg MSE':>10} {'Gap':>6} "
          f"{'Step':>10} {'RW':>10} {'Easy':>10}")
    print("-" * 95)

    for i, r in enumerate(results):
        marker = " ★" if i == 0 else ""
        print(f"{i+1:3d} {r['name']:>28} {r['source']:>8} {r['agg']:10.4e} "
              f"{r['gap']:5.1f}x {r['step']:10.4e} {r['rw']:10.4e} "
              f"{r['easy']:10.4e}{marker}")

    print("-" * 95)
    print(f"Total: {len(results)} models")

    if len(results) >= 2:
        best = results[0]
        print(f"\nBest: {best['name']} ({best['agg']:.4e}, {best['gap']:.1f}× oracle)")
        print(f"  Step: {best['step']:.4e}, RW: {best['rw']:.4e}, Easy: {best['easy']:.4e}")

    # Save machine-readable
    out = {r['name']: {
        'aggregate_mse': r['agg'], 'gap_to_oracle': r['gap'],
        'step_mse': r['step'], 'random_walk_mse': r['rw'],
        'easy_mean_mse': r['easy'], 'source': r['source']
    } for r in results}
    out['_metadata'] = {
        'oracle_aggregate': oracle_agg,
        'n_models': len(results),
    }

    outpath = 'outputs/unified_leaderboard.json'
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == '__main__':
    main()
