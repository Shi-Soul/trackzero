#!/usr/bin/env python3
"""Aggregate all experiment results into a unified comparison table.

Collects results from:
- Stage 1C experiments (existing models)
- HP sweep experiments (sweep_*)
- Hybrid strategy experiments (hybrid_*)
- Stage 1D experiments (dagger, cem, distill)

Outputs:
- Unified comparison table (ID + OOD)
- Ranking by setting
- Statistical summary
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def collect_results():
    """Gather all available experiment results."""
    results = {}
    base = Path("outputs")

    # 1. OOD benchmark on existing models
    ood_bench = base / "ood_benchmark_existing.json"
    if ood_bench.exists():
        with open(ood_bench) as f:
            existing = json.load(f)
        for name, vals in existing.items():
            results[f"1c_{name}"] = vals

    # 2. HP sweep results
    for d in sorted(base.glob("sweep_*")):
        rfile = d / "results.json"
        if rfile.exists():
            with open(rfile) as f:
                data = json.load(f)
            tag = d.name.replace("sweep_", "hp_")
            if "ood_results" in data:
                results[tag] = data["ood_results"]
            elif "eval" in data:
                results[tag] = data["eval"]

    # 3. Hybrid strategy results
    for d in sorted(base.glob("hybrid_*")):
        rfile = d / "results.json"
        if rfile.exists():
            with open(rfile) as f:
                data = json.load(f)
            tag = d.name
            if "ood_results" in data:
                results[tag] = data["ood_results"]

    # 4. Stage 1D results (check eval_results.json)
    for d in sorted(base.glob("stage1d_*")):
        for rfile in [d / "results.json", d / "eval_results.json"]:
            if rfile.exists():
                with open(rfile) as f:
                    data = json.load(f)
                tag = f"1d_{d.name.replace('stage1d_', '')}"
                # Handle different result formats
                if "ood_results" in data:
                    results[tag] = data["ood_results"]
                elif "id_multisine" in data:
                    results[tag] = data
                elif "mean_mse_total" in data:
                    results[tag] = {"id_multisine": data["mean_mse_total"]}
                break

    return results


def print_table(results):
    """Print unified comparison table."""
    if not results:
        print("No results found.")
        return

    # Find all settings across all results
    all_settings = set()
    for v in results.values():
        all_settings.update(v.keys())
    settings = sorted(all_settings)

    # Find baseline for relative comparison
    baseline_key = None
    for k in ["1c_random_512x4", "random_512x4"]:
        if k in results:
            baseline_key = k
            break

    baseline = results.get(baseline_key, {}) if baseline_key else {}

    # Print header
    print("=" * 140)
    print("UNIFIED EXPERIMENT COMPARISON")
    print("=" * 140)
    header = f"{'Method':<30}"
    for s in settings:
        short = s.replace("id_", "").replace("ood_", "")[:12]
        header += f"  {short:>14}"
    print(header)
    print("-" * len(header))

    # Print rows sorted by ID performance
    def sort_key(item):
        return item[1].get("id_multisine", float("inf"))

    for name, vals in sorted(results.items(), key=sort_key):
        row = f"{name:<30}"
        for s in settings:
            mse = vals.get(s)
            if mse is not None:
                base_val = baseline.get(s, mse)
                ratio = mse / base_val if base_val > 0 else 0
                row += f"  {mse:.2e}({ratio:4.1f}x)"
            else:
                row += f"  {'---':>14}"
        print(row)

    # Print rankings per setting
    print("\n" + "=" * 100)
    print("RANKINGS (best → worst)")
    print("=" * 100)
    for s in settings:
        ranked = [(n, v[s]) for n, v in results.items() if s in v]
        ranked.sort(key=lambda x: x[1])
        short = s.replace("id_", "").replace("ood_", "")
        top3 = ", ".join(f"{n}({v:.2e})" for n, v in ranked[:5])
        print(f"  {short:<16}: {top3}")

    # Summary statistics
    print("\n" + "=" * 100)
    print("DEGRADATION RATIOS (OOD / ID)")
    print("=" * 100)
    for name, vals in sorted(results.items(), key=sort_key):
        id_val = vals.get("id_multisine")
        if id_val is None or id_val == 0:
            continue
        step = vals.get("ood_step")
        rw = vals.get("ood_random_walk")
        if step is not None and rw is not None:
            print(f"  {name:<30}  step/ID={step/id_val:>8.1f}×  rw/ID={rw/id_val:>8.1f}×")


def save_results(results, path="outputs/unified_comparison.json"):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    results = collect_results()
    print(f"Collected {len(results)} experiment results\n")
    print_table(results)
    save_results(results)
