#!/usr/bin/env python3
"""Per-module singular spectra and S for a pool, the two inputs the allocator needs.

The allocator in predibase_budget_allocation.py never touches weights: it needs
each adapter's strength and the map from tau to (retained directions, achieved
L_W), and both follow from the spectrum of B A. Extracting spectra once lets the
allocation-transfer test run on every pool instead of only the one whose
evaluation harness happens to be loaded.

CPU only: 64 to 306 modules per adapter, each an r x r SVD after two QRs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wan_task_metrics import to_base as wan_to_base  # noqa: E402


def base_index(base: Path):
    """{tensor name: file} over a sharded or single-file checkpoint."""
    for sub in ("", "transformer"):
        d = base / sub if sub else base
        for idx in d.glob("*.index.json"):
            return d, json.loads(idx.read_text())["weight_map"]
        files = sorted(d.glob("*.safetensors"))
        if files:
            m = {}
            for f in files:
                with safe_open(f, framework="pt") as h:
                    for k in h.keys():
                        m[k] = f.name
            return d, m
    raise SystemExit(f"no weights under {base}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=("cts", "lorare", "land", "wan"), required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--spec", type=Path, required=True,
                    help="strength json (cts/lorare) or pool33.json (wan)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    root, wmap = base_index(args.base)
    norms: dict[str, float] = {}

    def bnorm(key: str):
        if key not in wmap:
            return None
        if key not in norms:
            with safe_open(root / wmap[key], framework="pt") as h:
                norms[key] = float(h.get_tensor(key).float().norm())
        return norms[key]

    spec = json.loads(args.spec.read_text())
    jobs = []
    if args.pool == "cts":
        for e in spec:
            n = e["adapter"]
            # the repository drops the zero padding: task0075 lives at ...-r16-task75
            num = int(n.replace("task", ""))
            jobs.append((n, f"Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task{num}", None))
    elif args.pool == "lorare":
        for e in spec:
            jobs.append((e["task"], f"Styxxxx/llama2_7b_lora-{e['task']}", None))
    elif args.pool == "land":
        for e in spec:
            name = e["adapter"]
            jobs.append((name, f"predibase/{name}", None))
    else:
        for n, e in spec.items():
            jobs.append((n, None, Path(e["path"])))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} adapters", flush=True)

    from huggingface_hub import snapshot_download
    out = {}
    for name, repo, path in jobs:
        try:
            if path is None:
                p = Path(snapshot_download(repo))
                f = next(iter(sorted(p.glob("adapter_model.safetensors"))), None)
                if f is None:
                    print(f"skip {name}: no safetensors", flush=True); continue
                w = load_file(f)
                cfg = json.loads((p / "adapter_config.json").read_text())
                r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
                scale = alpha / (math.sqrt(r) if cfg.get("use_rslora") else r)
            else:
                w = (load_file(path) if path.suffix == ".safetensors"
                     else torch.load(path, map_location="cpu", weights_only=False))
                w = w.get("state_dict", w) if isinstance(w, dict) else w
                scale = 1.0            # verified against the independent screen
            num = den = 0.0
            sig, wnorm = [], []
            for a in sorted(k for k in w if ".lora_A" in k or ".lora_down" in k):
                b = (a.replace(".lora_A", ".lora_B") if ".lora_A" in a
                     else a.replace(".lora_down", ".lora_up"))
                if b not in w:
                    continue
                if args.pool == "wan":
                    key = wan_to_base(a) + ".weight"
                else:
                    key = re.sub(r"\.lora_A(\.default)?\.weight$", ".weight", a)
                    key = key.split("base_model.model.")[-1]
                n = bnorm(key)
                if n is None:
                    continue
                A, B = w[a].float(), w[b].float()
                qb, rb = torch.linalg.qr(B, mode="reduced")
                qa, ra = torch.linalg.qr(A.T, mode="reduced")
                sv = torch.linalg.svdvals(rb @ ra.T) * scale
                sig.append([round(float(x), 8) for x in sv])
                # the per-module base norm is what makes a layer-level strength
                # S_l = ||dW_l||/||W_l|| definable; without it a layer allocation
                # can only rank directions by absolute energy
                wnorm.append(round(n, 6))
                num += float(sv.square().sum())
                den += n * n
            if not sig:
                print(f"skip {name}: no matched modules", flush=True); continue
            out[name] = {"S": math.sqrt(num / den), "sigma": sig, "wnorm": wnorm}
            print(f"  {name[:34]:36s} S={out[name]['S']:.5f} modules={len(sig)}", flush=True)
        except Exception as exc:
            print(f"skip {name}: {type(exc).__name__} {exc}"[:110], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out) + "\n")
    print(f"\nwrote {args.output} ({len(out)} adapters)")


if __name__ == "__main__":
    main()
