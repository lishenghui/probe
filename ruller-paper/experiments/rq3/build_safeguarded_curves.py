#!/usr/bin/env python3
"""Build a per-budget envelope over independently measured rank allocators."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def load_group(pattern: str, expected_gamma: str) -> dict[str, dict]:
    result = {}
    for filename in glob.glob(pattern):
        doc = json.loads(Path(filename).read_text())
        if ("adapter" in doc and "curve" in doc and
                (expected_gamma == "*" or
                 float(doc.get("gamma", "nan")) == float(expected_gamma))):
            result[doc["adapter"]] = doc
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", action="append", nargs=3,
                    metavar=("LABEL", "GLOB", "GAMMA"), required=True,
                    help="GAMMA is a number or * to disable metadata filtering")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--prefix", default="safeguarded")
    args = ap.parse_args()

    groups = [(label, load_group(pattern, gamma))
              for label, pattern, gamma in args.candidate]
    adapters = set(groups[0][1])
    for label, group in groups[1:]:
        if set(group) != adapters:
            raise SystemExit(f"adapter mismatch for {label}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    totals = {label: 0 for label, _ in groups}
    for adapter in sorted(adapters):
        docs = [(label, group[adapter]) for label, group in groups]
        by_candidate = []
        budgets = None
        for label, doc in docs:
            rows = {int(row["k"]): row for row in doc["curve"]}
            budgets = set(rows) if budgets is None else budgets & set(rows)
            by_candidate.append((label, rows))
        curve = []
        for k in sorted(budgets or ()):
            label, rows = min(by_candidate, key=lambda item: item[1][k]["d_js"])
            row = dict(rows[k])
            row["source"] = label
            totals[label] += 1
            curve.append(row)
        out = dict(docs[0][1])
        out.update(method="safeguarded-envelope",
                   candidates=[label for label, _ in groups], curve=curve)
        (args.output_dir / f"{args.prefix}_{adapter}.json").write_text(
            json.dumps(out, indent=2) + "\n")
    print(f"wrote {len(adapters)} adapter curves; selected points: {totals}")


if __name__ == "__main__":
    main()
