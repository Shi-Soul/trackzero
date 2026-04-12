"""Auto-detect completed training runs and benchmark them immediately."""
import json, time, sys, os, subprocess
from pathlib import Path

MODELS = {
    "hp_random_1024x6": "outputs/hp_random_1024x6_lr3e4/best_model.pt",
    "hp_random_2048x3": "outputs/hp_random_2048x3_lr3e4/best_model.pt",
    "hp_random_512x4_wd": "outputs/hp_random_512x4_lr1e4_wd/best_model.pt",
    "hp_random_1024x4_wd": "outputs/hp_random_1024x4_lr3e4_wd/best_model.pt",
    "hp_maxent_1024x6": "outputs/hp_maxent_1024x6_lr3e4/best_model.pt",
    "hp_maxent_2048x4": "outputs/hp_maxent_2048x4_lr1e4/best_model.pt",
    "dagger_512x4": "outputs/dagger_benchmark_512x4/best_model.pt",
    "dagger_1024x4": "outputs/dagger_benchmark_1024x4/best_model.pt",
}

benchmarked = set()
results_file = Path("outputs/hp_dagger_benchmark.json")

# Load any existing results
if results_file.exists():
    with open(results_file) as f:
        all_results = json.load(f)
    benchmarked = set(all_results.keys())
else:
    all_results = {}

print(f"Auto-benchmarking {len(MODELS) - len(benchmarked)} remaining models...")
print(f"Already benchmarked: {benchmarked}")

while len(benchmarked) < len(MODELS):
    for name, path in MODELS.items():
        if name in benchmarked:
            continue
        if not Path(path).exists():
            continue
        
        print(f"\n{'='*60}")
        print(f"FOUND: {name} → {path}")
        print(f"Running quick benchmark...")
        
        device = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        cmd = [
            sys.executable, "scripts/quick_benchmark.py", path,
            "--device", f"cuda"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        print(result.stdout)
        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            continue
        
        # Load the result
        qb_path = Path(path).parent / "quick_benchmark.json"
        if qb_path.exists():
            with open(qb_path) as f:
                bench = json.load(f)
            all_results[name] = bench
            benchmarked.add(name)
            
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)
            
            print(f"  ★ {name}: {bench['_aggregate']:.4e}")
    
    remaining = len(MODELS) - len(benchmarked)
    if remaining > 0:
        print(f"\n  Waiting for {remaining} models... (checking every 60s)")
        time.sleep(60)

print(f"\n{'='*60}")
print("ALL MODELS BENCHMARKED!")
print(f"{'='*60}")
for name in sorted(all_results.keys(), key=lambda n: all_results[n]["_aggregate"]):
    print(f"  {name:25s}: {all_results[name]['_aggregate']:.4e}")
