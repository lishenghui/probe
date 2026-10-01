#!/usr/bin/env python3
"""Re-download the exact adapter files selected by a completed census."""

import argparse
import csv
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--first-order", type=int, default=75)
    parser.add_argument("--last-order", type=int, default=100)
    args = parser.parse_args()

    with (args.results / "repos.csv").open(newline="") as handle:
        order = {row["repo_id"]: int(row["popularity_order"]) for row in csv.DictReader(handle)}
    with (args.results / "layers.csv").open(newline="") as handle:
        selected = sorted({
            (row["repo_id"], row["filename"])
            for row in csv.DictReader(handle)
            if args.first_order <= order[row["repo_id"]] <= args.last_order
        }, key=lambda item: (order[item[0]], item[1]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, (repo_id, filename) in enumerate(selected, 1):
        source = Path(hf_hub_download(repo_id, filename, cache_dir=args.cache_dir))
        destination = args.output_dir / repo_id / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"[{index}/{len(selected)}] copied {repo_id}/{filename}", flush=True)


if __name__ == "__main__":
    main()
