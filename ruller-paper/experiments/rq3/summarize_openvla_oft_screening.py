#!/usr/bin/env python3
"""Parse official OpenVLA-OFT LIBERO logs into rollout/task/variant tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


START_TASK = re.compile(r"Task \d+: (.+)")
SUCCESS = re.compile(r"Success: (True|False)")
TASK_RATE = re.compile(r"Current task success rate: ([0-9.eE+-]+)")
OVERALL = re.compile(r"Overall success rate: ([0-9.eE+-]+)")


def parse_log(path: Path, suite: str, variant: str) -> tuple[list[dict], list[dict], float]:
    rollouts, tasks = [], []
    task_name = None
    episode = 0
    overall = float("nan")
    for line in path.read_text(errors="replace").splitlines():
        if match := START_TASK.search(line):
            task_name = match.group(1).strip()
            episode = 0
        elif match := SUCCESS.search(line):
            episode += 1
            rollouts.append({"suite": suite, "variant": variant, "task": task_name,
                             "episode": episode, "success": match.group(1) == "True"})
        elif match := TASK_RATE.search(line):
            tasks.append({"suite": suite, "variant": variant, "task": task_name,
                          "success_rate": float(match.group(1))})
        elif match := OVERALL.search(line):
            overall = float(match.group(1))
    return rollouts, tasks, overall


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    all_rollouts, all_tasks, summary = [], [], []
    for suite_dir in sorted(p for p in args.results.iterdir() if p.is_dir()):
        baseline = None
        suite_rows = []
        for variant_dir in sorted(p for p in suite_dir.iterdir() if p.is_dir()):
            logs = sorted(variant_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime)
            if not logs:
                continue
            rollouts, tasks, overall = parse_log(logs[-1], suite_dir.name, variant_dir.name)
            all_rollouts.extend(rollouts)
            all_tasks.extend(tasks)
            local_comp = variant_dir / "compression.json"
            archived_comp = args.variants / suite_dir.name / variant_dir.name / "lora_adapter" / "compression.json"
            comp = local_comp if local_comp.exists() else archived_comp
            metrics = json.loads(comp.read_text()) if comp.exists() else {
                "threshold": 1.0, "mean_retained_rank_ratio": 1.0, "global_L_W": 0.0}
            row = {"suite": suite_dir.name, "variant": variant_dir.name,
                   "threshold": metrics["threshold"], "global_L_W": metrics["global_L_W"],
                   "mean_retained_rank_ratio": metrics["mean_retained_rank_ratio"],
                   "success_rate": overall}
            suite_rows.append(row)
            if variant_dir.name == "e100":
                baseline = overall
        for row in suite_rows:
            row["success_drop_pp"] = 100 * (baseline - row["success_rate"]) if baseline == baseline else float("nan")
            summary.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "rollouts.csv", all_rollouts)
    write_csv(args.output / "tasks.csv", all_tasks)
    write_csv(args.output / "summary.csv", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
