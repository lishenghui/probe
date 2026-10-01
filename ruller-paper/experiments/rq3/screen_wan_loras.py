#!/usr/bin/env python3
"""Screen public Wan2.1-T2V-1.3B LoRAs for the video pool.

The pool had four members and only three distinct strengths, which is why its
exponents were not estimable: the cluster bootstrap put a in [-2.65, +0.51].
This widens it. The screen is the cheap step -- download the adapter, map its
keys onto the diffusers base, compute S = ||dW||_F / ||W||_F -- and it is CPU
only, so a candidate is rejected before any GPU time is spent on it.

A candidate is kept only if every LoRA pair maps onto a real base tensor. That
check is what distinguishes a genuine Wan2.1-1.3B T2V adapter from the 14B, I2V,
Wan2.2 and quantised-checkpoint repositories that the same search returns.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wan_task_metrics import b_key, to_base  # noqa: E402

WEIGHT_SUFFIX = (".safetensors", ".ckpt", ".pt", ".bin")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--max-mb", type=float, default=1200,
                    help="skip files too large to be a 1.3B LoRA; a full checkpoint "
                         "is not worth the download to reject")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.99, 0.95, 0.90, 0.80, 0.70, 0.50])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    idx = json.loads((args.base / "transformer"
                      / "diffusion_pytorch_model.safetensors.index.json").read_text())
    wmap = idx["weight_map"]
    norms: dict[str, float] = {}

    def base_norm(key: str) -> float | None:
        if key not in wmap:
            return None
        if key not in norms:
            with safe_open(args.base / "transformer" / wmap[key], framework="pt") as h:
                norms[key] = float(h.get_tensor(key).float().norm())
        return norms[key]

    api = HfApi()
    cands = [c for c in json.loads(args.candidates.read_text())][args.shard::args.shards]
    print(f"shard {args.shard}/{args.shards}: {len(cands)} candidates", flush=True)

    out = []
    for repo in cands:
        try:
            files = [f for f in api.list_repo_files(repo)
                     if f.endswith(WEIGHT_SUFFIX) and "/" not in f.strip("/")]
            info = api.model_info(repo, files_metadata=True)
            sizes = {s.rfilename: (s.size or 0) for s in info.siblings}
        except Exception as exc:
            print(f"skip {repo}: {type(exc).__name__}", flush=True)
            continue
        for fn in files:
            mb = sizes.get(fn, 0) / 1e6
            if mb > args.max_mb or mb < 1:
                continue
            try:
                p = Path(hf_hub_download(repo, fn))
                if p.suffix == ".safetensors":
                    with safe_open(p, framework="pt") as h:
                        w = {k: h.get_tensor(k) for k in h.keys()}
                else:
                    w = torch.load(p, map_location="cpu", weights_only=False)
                    w = w.get("state_dict", w) if isinstance(w, dict) else w
                aks = sorted(k for k in w if ".lora_A" in k or ".lora_down" in k)
                if not aks:
                    continue
                num = den = 0.0
                spectra, matched, missing = [], 0, 0
                for a in aks:
                    b = b_key(a)
                    if b not in w:
                        continue
                    key = to_base(a) + ".weight"
                    n = base_norm(key)
                    if n is None:
                        missing += 1
                        continue
                    A, B = w[a].float(), w[b].float()
                    if A.ndim > 2 or B.ndim > 2 or A.shape[0] != B.shape[1]:
                        missing += 1
                        continue
                    qb, rb = torch.linalg.qr(B, mode="reduced")
                    qa, ra = torch.linalg.qr(A.T, mode="reduced")
                    sv = torch.linalg.svdvals(rb @ ra.T)
                    spectra.append(sv)
                    num += float(sv.square().sum())
                    den += n * n
                    matched += 1
                if not matched or missing:
                    print(f"reject {repo}/{fn}: matched={matched} unmatched={missing}",
                          flush=True)
                    continue
                S = math.sqrt(num / den)
                rec = {"repo": repo, "file": fn, "name": repo.split("/")[-1],
                       "S": S, "modules": matched, "size_mb": round(mb, 1),
                       "rank": int(max(s.numel() for s in spectra)), "L_W": {}, "rank_frac": {}}
                for tau in args.thresholds:
                    keep = drop = 0.0
                    kt = nt = 0
                    for sv in spectra:
                        e = sv.square()
                        c = torch.cumsum(e, 0)
                        k = min(int(torch.searchsorted(c, tau * c[-1]).item()) + 1, e.numel())
                        keep += float(e[:k].sum()); drop += float(e[k:].sum())
                        kt += k; nt += e.numel()
                    lab = f"e{round(tau * 100):02d}"
                    rec["L_W"][lab] = math.sqrt(drop / (keep + drop)) if keep + drop else 0.0
                    rec["rank_frac"][lab] = kt / nt
                out.append(rec)
                print(f"KEEP {repo}/{fn}  S={S:.5f} rank={rec['rank']} modules={matched}",
                      flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(out, indent=2) + "\n")
            except Exception:
                print(f"error {repo}/{fn}:\n{traceback.format_exc(limit=1)}", flush=True)
    print(f"\nwrote {args.output} ({len(out)} adapters kept)")


if __name__ == "__main__":
    main()
