#!/usr/bin/env python3
"""Build the controlled FRA inner-proposal x fleet-currency allocations."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from two_level_allocation import minimax_allocate


def load(pattern: str) -> dict[str, dict]:
    out = {}
    for filename in glob.glob(pattern):
        doc = json.loads(Path(filename).read_text())
        if "adapter" not in doc or len(doc.get("curve", [])) < 2:
            continue
        if doc["adapter"] in out:
            raise ValueError(f"duplicate adapter {doc['adapter']}")
        out[doc["adapter"]] = doc
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--funcdp", required=True)
    ap.add_argument("--spectral", required=True)
    ap.add_argument("--budgets", type=int, nargs="+", default=[5880, 3191, 2683])
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    func, spec = load(args.funcdp), load(args.spectral)
    proposals_to_solve = ["funcdp"]
    if set(func) == set(spec):
        proposals_to_solve.append("funcdp_spec")
    else:
        print(f"spectral curves incomplete; solving FuncDP row only "
              f"(func-only={len(set(func)-set(spec))}, "
              f"spectral-only={len(set(spec)-set(func))})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for proposals in proposals_to_solve:
        for currency in ("d_js", "sequence_flip"):
            curves = {}
            for adapter in sorted(func):
                candidates = [("funcdp", row) for row in func[adapter]["curve"]]
                if proposals == "funcdp_spec":
                    candidates += [("spectral", row) for row in spec[adapter]["curve"]]
                # At a fixed adapter budget, proposal coverage chooses the
                # candidate with the lower same-currency calibration risk.
                best = {}
                for source, row in candidates:
                    if currency not in row:
                        raise ValueError(f"{adapter}: missing {currency}")
                    enriched = dict(row, source=source, d_js=float(row[currency]),
                                    measured_js=float(row["d_js"]),
                                    measured_sequence_flip=float(row["sequence_flip"]))
                    k = int(row["k"])
                    if k not in best or enriched["d_js"] < best[k]["d_js"]:
                        best[k] = enriched
                curves[adapter] = list(best.values())

            tag = f"{proposals}_{'js' if currency == 'd_js' else 'flip'}"
            for budget in args.budgets:
                chosen, ceiling, spent = minimax_allocate(curves, budget)
                # Restore d_js to its literal meaning for downstream provenance;
                # minimax risk remains explicitly recorded in fleet_risk.
                for row in chosen.values():
                    row["fleet_risk"] = row["d_js"]
                    row["d_js"] = row["measured_js"]
                result = {"pool": "LoRARetriever", "method": tag,
                          "currency": currency, "n": len(chosen),
                          "budget": budget, "spent": spent,
                          "optimal_ceiling": ceiling, "allocation": chosen}
                output = args.output_dir / f"fra2x2_{tag}_b{budget}.json"
                output.write_text(json.dumps(result, indent=2) + "\n")
                sources = {s: sum(r["source"] == s for r in chosen.values())
                           for s in ("funcdp", "spectral")}
                print(f"{tag:18s} B={budget} spent={spent} ceiling={ceiling:.6g} "
                      f"sources={sources}")


if __name__ == "__main__":
    main()
