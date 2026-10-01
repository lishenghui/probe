#!/usr/bin/env python3
"""Assemble the 128-prompt anchor file, to test whether A-SCT is calibration-limited.

A-SCT trails unanchored SCT on LoRA Land and beats it on LoRARetriever, and there
are two candidate reasons that more calibration data separates:

  * the per-adapter anchor D_i(tau=.95) is measured on 32 prompts and is noisy,
    and that noise enters the allocation directly; or
  * the model assumes one slope b per pool, and LoRA Land's per-adapter slopes are
    the most dispersed of the three (CV 45% against 28% and 23%), so the shared-b
    assumption is what limits it.

Only the first is fixed by more prompts. Quadrupling to 128 therefore tells the
two apart: if the gap persists, the ceiling is the model and not the budget.

The anchor prompts still start at example 200, past everything used for
evaluation, so the disjointness the method depends on is preserved.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path("artifacts/rq3/results")


def main() -> None:
    out = []
    for path in sorted(glob.glob(str(ROOT / "land_anchor128_*.json"))):
        if path.endswith("_tokens"):
            continue
        for r in json.load(open(path)):
            d = Path(path).with_name(Path(path).stem + "_tokens")
            variants = {}
            for lab, v in r["variants"].items():
                f = d / f"{r['adapter']}-{lab}.npz"
                variants[lab] = {"L_W": v["L_W"], "rank_frac": v["rank_frac"],
                                 "d_out": float(np.load(f)["per_token_js"].mean())}
            out.append({"adapter": r["adapter"], "S": r["S"], "variants": variants,
                        "prompts": r["n"], "example_start": r["example_start"]})
    (ROOT / "land12_anchor128.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"{len(out)} adapters, {out[0]['prompts']} calibration prompts each")

    small = {r["adapter"]: r for r in
             json.load(open(ROOT / "land12_anchor_disjoint.json"))}
    print(f"\n{'adapter':11s}{'D_anc(32)':>12}{'D_anc(128)':>13}{'ratio':>8}")
    for r in sorted(out, key=lambda x: x["adapter"]):
        a = r["variants"]["e95"]["d_out"]
        b = small[r["adapter"]]["variants"]["e95"]["d_out"]
        print(f"{r['adapter']:11s}{b:12.6f}{a:13.6f}{a / b:8.2f}")


if __name__ == "__main__":
    main()
