#!/usr/bin/env python3
"""Compare JS distortion with unlabeled decision-risk currencies.

Consumes the per-token NPZ files emitted by ``predibase_task_metrics.py`` and
matches them to the corresponding task-metric JSON.  The main comparison is
within adapter, so task difficulty and metric scale cannot create a pooled
correlation by themselves.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def finite_rho(x, y) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--tokens", required=True, help="glob for per-variant NPZ files")
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--scales", type=float, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    docs = json.loads(args.metrics.read_text())
    docs = docs if isinstance(docs, list) else [docs]
    by_adapter = {d["adapter"]: d for d in docs}
    rows = []
    token_rows = []
    for filename in sorted(glob.glob(args.tokens)):
        path = Path(filename)
        stem = path.stem
        adapter, label = stem.rsplit("-", 1)
        if adapter not in by_adapter or label not in by_adapter[adapter]["variants"]:
            continue
        task = by_adapter[adapter]
        variant = task["variants"][label]
        z = np.load(path)
        example = z["example"].astype(int)
        margin = np.maximum(z["per_token_orig_margin"].astype(float), 0.0)
        js = np.maximum(z["per_token_js"].astype(float), 0.0)
        n_examples = len(z["per_example"])

        # Equal weight per calibration example, as in E_x[...].  This differs
        # from a token mean when generation lengths vary.
        def example_mean(values):
            sums = np.bincount(example, weights=values, minlength=n_examples)
            count = np.bincount(example, minlength=n_examples)
            return float(np.mean(sums / np.maximum(count, 1)))

        full_score = float(np.mean(z["per_example_orig"]))
        compressed_score = float(np.mean(z["per_example"]))
        row = {
            "adapter": adapter,
            "variant": label,
            "js": example_mean(js),
            "task_loss": full_score - compressed_score,
            "token_flip": float(np.mean(z["per_token_comp_margin"] < 0)),
            "sequence_flip": float(np.mean(z["first_shared_prefix_flip"] >= 0)),
        }
        ratio = np.sqrt(js) / (margin + args.epsilon)
        token_rows.append({"adapter": adapter, "js": js, "ratio": ratio,
                           "flip": z["per_token_comp_margin"] < 0})
        for scale in args.scales:
            row[f"exposure_c{scale:g}"] = example_mean(np.minimum(1.0, scale * ratio))
        # The JSON score is useful for non-binary metrics; retain it separately.
        row["json_task_loss"] = float(variant.get("abs_drop", row["task_loss"]))
        rows.append(row)

    currencies = ["js", "token_flip", "sequence_flip"] + [
        f"exposure_c{x:g}" for x in args.scales
    ]
    summary = {}
    for currency in currencies:
        per_adapter = {}
        for adapter in sorted({r["adapter"] for r in rows}):
            group = [r for r in rows if r["adapter"] == adapter]
            per_adapter[adapter] = finite_rho(
                [r[currency] for r in group], [r["json_task_loss"] for r in group]
            )
        valid = np.asarray([x for x in per_adapter.values() if np.isfinite(x)])
        summary[currency] = {
            "pooled_rho": finite_rho(
                [r[currency] for r in rows], [r["json_task_loss"] for r in rows]
            ),
            "within_adapter": per_adapter,
            "within_mean_rho": float(np.mean(valid)) if valid.size else float("nan"),
            "within_median_rho": float(np.median(valid)) if valid.size else float("nan"),
            "adapters": int(valid.size),
        }

    token_prediction = {}
    for adapter in sorted({r["adapter"] for r in token_rows}):
        group = [r for r in token_rows if r["adapter"] == adapter]
        y = np.concatenate([r["flip"] for r in group])
        if np.unique(y).size < 2:
            continue
        token_prediction[adapter] = {
            "flip_rate": float(y.mean()),
            "js_auc": float(roc_auc_score(y, np.concatenate([r["js"] for r in group]))),
            "exposure_auc": float(roc_auc_score(y, np.concatenate([r["ratio"] for r in group]))),
            "js_average_precision": float(average_precision_score(
                y, np.concatenate([r["js"] for r in group]))),
            "exposure_average_precision": float(average_precision_score(
                y, np.concatenate([r["ratio"] for r in group]))),
        }
    result = {"n_points": len(rows), "epsilon": args.epsilon,
              "rows": rows, "summary": summary, "token_flip_prediction": token_prediction}
    print(f"points={len(rows)} epsilon={args.epsilon:g}")
    print(f"{'currency':18s} {'pooled':>8s} {'within-mean':>12s} {'within-med':>11s} {'N':>3s}")
    for name, stat in summary.items():
        print(f"{name:18s} {stat['pooled_rho']:8.3f} {stat['within_mean_rho']:12.3f} "
              f"{stat['within_median_rho']:11.3f} {stat['adapters']:3d}")
    if token_prediction:
        print("token-flip macro AUC: "
              f"JS={np.mean([x['js_auc'] for x in token_prediction.values()]):.3f} "
              f"exposure={np.mean([x['exposure_auc'] for x in token_prediction.values()]):.3f}")
        print("token-flip macro AP:  "
              f"JS={np.mean([x['js_average_precision'] for x in token_prediction.values()]):.3f} "
              f"exposure={np.mean([x['exposure_average_precision'] for x in token_prediction.values()]):.3f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")


if __name__ == "__main__":
    main()
