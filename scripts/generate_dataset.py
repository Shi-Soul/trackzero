#!/usr/bin/env python3
"""Generate multisine trajectory dataset."""

import argparse
import time
from pathlib import Path

from trackzero.config import load_config
from trackzero.data.generator import generate_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate multisine trajectory dataset")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.workers is not None:
        cfg.dataset.num_workers = args.workers

    output_dir = Path(args.output_dir)

    def progress(done, total):
        print(f"  {done}/{total} trajectories", flush=True)

    splits = []
    if args.split in ("train", "both"):
        splits.append(("train", cfg.dataset.n_train, args.seed))
    if args.split in ("test", "both"):
        splits.append(("test", cfg.dataset.n_test, args.seed + 1_000_000))

    for name, n, seed in splits:
        path = output_dir / f"{name}.h5"
        print(f"Generating {name} split: {n} trajectories -> {path}")
        t0 = time.time()
        generate_dataset(cfg, path, n, seed=seed, progress_callback=progress)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
