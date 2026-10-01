#!/usr/bin/env python3
"""Assemble the LoRA Land grid, growing the pool as recipes become available.

The pool was seven because five tasks were unusable, and only one of those reasons
was real. CoLA and MRPC scored below chance because their prompts dropped the
trailing space the published model cards specify -- "### Label: " -- which makes
those two adapters emit an empty string on many rows; restoring it takes them to
0.860 and 0.930. CoNLL++, E2E and DBpedia were measured all along but arrived
after the downstream chain had been built on the original seven.

Only MNLI is still excluded: the separator does not rescue it (0.320 to 0.340
against a chance rate of 0.35), so its harness is wrong in some other way.

The divergence grid is rebuilt from per-token dumps rather than copied, because
d_out is the mean per-token JS over the generated positions and only the token
dumps carry it. Sharded runs are merged first: variants are strided across shards,
so a task's twelve variants can arrive in four files.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path("artifacts/rq3/results")
# Added in two rounds. The first five were already measured or needed only the
# trailing separator the model cards specify; the second five are new recipes
# copied verbatim from those cards, each checked for headroom before its sweep ran
# (cnn clears the bar by the least, +0.069, so its retained utility has the
# smallest denominator in the pool).
ROUND1 = ["glue_cola", "glue_mrpc", "conllpp", "e2e_nlg", "dbpedia"]
ROUND2 = ["drop", "glue_stsb", "cnn", "agnews_explained", "hellaswag_processed"]
NEW = ROUND1 + ROUND2


def merge_sharded(task: str):
    """One record per task, with the variant dicts of every shard combined."""
    hits = []
    for stem in ("land12_margin", "land17_margin"):
        hits += sorted(glob.glob(str(ROOT / f"{stem}_{task}.json")))
        hits += sorted(glob.glob(str(ROOT / f"{stem}_{task}_s*.json")))
    if not hits:
        raise FileNotFoundError(f"no sweep output for {task}")
    base = None
    for path in hits:
        for row in json.load(open(path)):
            if base is None:
                base = {k: v for k, v in row.items() if k != "variants"}
                base["variants"] = {}
            base["variants"].update(row["variants"])
    return base


def token_mean(task: str, label: str, dirs: list[str]) -> float:
    for d in dirs:
        for f in glob.glob(str(ROOT / d / f"{task}-{label}.npz")):
            return float(np.load(f)["per_token_js"].mean())
    raise FileNotFoundError((task, label, dirs))


def main() -> None:
    task = json.load(open(ROOT / "land7_task_grid.json"))
    have = {r["adapter"] for r in task}
    for t in NEW:
        if t in have:
            continue
        r = merge_sharded(t)
        r.setdefault("headroom", r["metric_orig"] - r["metric_base"])
        task.append(r)
    (ROOT / "land17_task_grid.json").write_text(json.dumps(task, indent=2) + "\n")

    div = json.load(open(ROOT / "land7_div_grid.json"))
    have = {r["adapter"] for r in div}
    for t in NEW:
        if t in have:
            continue
        r = merge_sharded(t)
        dirs = [f"land12_tokens_{t}", f"land17_tokens_{t}"]
        div.append({"adapter": t, "S": r["S"],
                    "variants": {lab: {"L_W": v["L_W"], "rank_frac": v["rank_frac"],
                                       "d_out": token_mean(t, lab, dirs)}
                                 for lab, v in r["variants"].items()}})
    (ROOT / "land17_div_grid.json").write_text(json.dumps(div, indent=2) + "\n")

    anchor = []
    for path in sorted(glob.glob(str(ROOT / "land_anchor_disjoint_*.json"))):
        for r in json.load(open(path)):
            d = Path(path).with_name(Path(path).stem + "_tokens")
            anchor.append({"adapter": r["adapter"], "S": r["S"],
                           "prompts": r["n"], "example_start": r["example_start"],
                           "variants": {lab: {"L_W": v["L_W"],
                                              "rank_frac": v["rank_frac"],
                                              "d_out": float(np.load(
                                                  d / f"{r['adapter']}-{lab}.npz"
                                              )["per_token_js"].mean())}
                                        for lab, v in r["variants"].items()}})
    (ROOT / "land17_anchor_disjoint.json").write_text(json.dumps(anchor, indent=2) + "\n")

    print(f"task grid   {len(task):3d} adapters")
    print(f"div grid    {len(div):3d} adapters")
    print(f"anchors     {len(anchor):3d} adapters")
    # The allocator consumes the six threshold levels only, which is what the
    # original seven were measured on; the five new tasks also carry six rank
    # levels, which is surplus rather than a mismatch.
    TAUS = [f"e{t}" for t in (99, 95, 90, 80, 70, 50)]
    for name, rows, want in (("task", task, TAUS), ("div", div, TAUS),
                             ("anchor", anchor, ["e99", "e95"])):
        bad = [r["adapter"] for r in rows if not all(l in r["variants"] for l in want)]
        print(f"  {name}: {'all complete' if not bad else 'MISSING levels for ' + str(bad)}")


if __name__ == "__main__":
    main()
