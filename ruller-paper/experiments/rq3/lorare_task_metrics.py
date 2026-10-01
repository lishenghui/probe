#!/usr/bin/env python3
"""Task metrics for the LoraRetriever pool: the low-strength arm of the warm-up.

The point of this population is the contrast. Its adapters share a base
(Llama-2-7B), a configuration (r=8, alpha=16, q and v projections -- the same
configuration as LoRA Land) and a training pipeline, and their strengths sit an
order of magnitude below the controlled pool and two below the LoRA Land adapters
that collapse: S spans 0.0010 to 0.0419 with a strength-tercile ratio of 1.6.

So this is the arm where the answer should be "nothing happens". If the same
retained energy that destroys WikiSQL leaves all 48 of these intact, the
difference has to be attributed to something other than the threshold, and
adapter strength is the quantity that differs.

Prompts, references and the per-task metric all come from the authors' own
evaluation set, so nothing here depends on guessing the training template -- the
failure mode that silently corrupted an earlier pool in this project.

One thing the eval set does not carry is the separator between the input and the
answer. Its `inputs` field ends with the FLAN options block, and handed that string
verbatim the adapter emits EOS immediately: every prediction is the empty string
and the task metric is exactly 0.0000, which reads as "compression destroyed the
adapter" when in fact the adapter never spoke. `calibrate_lorare_prompt.py` sweeps
five candidate cues over five tasks spanning all three metrics; the bare prompt is
the only one with a *negative* adapter-minus-base gain, and "\n\n" gives the
largest mean headroom (0.211) while staying uniform across em, BLEU and ROUGE.
Headroom is the right criterion here because it is the denominator of the retained
utility this pool is being measured for.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compress_adapter import truncated_factors  # noqa: E402

EVAL_SET = "Styxxxx/LoraRetriever_EvalSet"
ADAPTER = "Styxxxx/llama2_7b_lora-{task}"


def normalise(text: str) -> list[str]:
    """Casefold and strip punctuation, keeping every script (the pool has WMT tasks)."""
    text = "".join(" " if unicodedata.category(c).startswith("P")
                   or unicodedata.category(c) in ("Zs", "Cc") else c
                   for c in text.casefold())
    return text.split()


def _lcs_f1(p: list[str], r: list[str]) -> float:
    if not p or not r:
        return 0.0
    prev = [0] * (len(r) + 1)
    for a in p:
        cur = [0]
        for j, b in enumerate(r):
            cur.append(prev[j] + 1 if a == b else max(cur[j], prev[j + 1]))
        prev = cur
    lcs = prev[-1]
    if not lcs:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


def bleu(pred: str, ref: str, n: int = 4) -> float:
    """Sentence BLEU with add-one smoothing, enough for a relative comparison."""
    p, r = normalise(pred), normalise(ref)
    if not p:
        return 0.0
    logs = 0.0
    for k in range(1, n + 1):
        pg = defaultdict(int)
        for i in range(len(p) - k + 1):
            pg[tuple(p[i:i + k])] += 1
        rg = defaultdict(int)
        for i in range(len(r) - k + 1):
            rg[tuple(r[i:i + k])] += 1
        match = sum(min(c, rg[g]) for g, c in pg.items())
        total = max(sum(pg.values()), 1)
        logs += math.log((match + 1) / (total + 1))
    bp = 1.0 if len(p) > len(r) else math.exp(1 - len(r) / max(len(p), 1))
    return bp * math.exp(logs / n)


def score(metric: str, pred: str, target: str) -> float:
    """The authors' own per-task metric: exact match, BLEU or ROUGE-L."""
    if metric == "em":
        p = " ".join(normalise(pred))
        t = " ".join(normalise(target))
        # generation is free-form, so accept the target as the answer prefix
        return float(bool(p) and (p == t or p.startswith(t) or t == p.split("\n")[0]))
    if metric == "bleu":
        return bleu(pred, target)
    return _lcs_f1(normalise(pred), normalise(target))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--thresholds", type=float, nargs="*",
                    default=[0.99, 0.95, 0.90, 0.80, 0.70, 0.50])
    ap.add_argument("--ranks", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--tasks", nargs="*", default=[])
    ap.add_argument("--prompts", type=int, default=50)
    ap.add_argument("--start", type=int, default=0,
                    help="offset within each task, for a fixed disjoint "
                         "calibration/evaluation split")
    ap.add_argument("--new", type=int, default=48)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--cue", default="\n\n",
                    help="separator appended to each prompt; see the module docstring. "
                         "The default was chosen by mean headroom over five tasks, not "
                         "by task-by-task tuning, which would be a confound.")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--allocation", type=Path, nargs="*", default=None,
                    help="dense-allocation JSON(s). One file keeps the label `dense`; "
                         "several are labelled by budget, so one job can score a whole "
                         "allocation curve after paying for base and full once.")
    ap.add_argument("--rule-shard", type=int, default=0,
                    help="split a task's variant list across jobs; only useful when "
                         "that list is long, since base and full are repeated")
    ap.add_argument("--rule-shards", type=int, default=1)
    args = ap.parse_args()
    paths = list(args.allocation or [])
    docs = [json.loads(p.read_text()) for p in paths]
    if len(paths) <= 1:
        labels = ["dense"] * len(paths)
    else:
        labels = [f"dense{int(d['budget'])}" for d in docs]
        if len(set(labels)) != len(labels):
            # several rules can share a budget; fall back to the file stem so an
            # ablation ladder can be scored in a single pass
            labels = [p.stem for p in paths]
        if len(set(labels)) != len(labels):
            raise SystemExit("allocation files are not distinguishable by budget or name")
    dense_allocations = list(zip(labels, [d["allocation"] for d in docs]))

    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_dataset(EVAL_SET, split="test")
    by_task: dict[str, list] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    tasks = args.tasks or sorted(by_task)
    tasks = tasks[args.shard::args.shards]
    print(f"{len(by_task)} tasks in the eval set; this shard takes {len(tasks)}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"
    tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16).to("cuda").eval()
    print("base loaded", flush=True)

    peft_model, results = None, []
    for task in tasks:
        items = by_task[task][args.start:args.start + args.prompts]
        if not items:
            print(f"skip {task}: no rows after offset {args.start}", flush=True)
            continue
        prompts = [r["inputs"] + args.cue for r in items]
        targets = [r["targets"] for r in items]
        metric = items[0]["metric"]
        # the adapter repo name drops the FLAN template suffix
        short = re.sub(r"_\d+templates$", "", task)
        try:
            local = Path(snapshot_download(ADAPTER.format(task=short),
                                           local_dir=args.work / short))
        except Exception as exc:
            print(f"skip {task}: adapter {type(exc).__name__}", flush=True)
            continue

        weights = load_file(local / "adapter_model.safetensors")
        cfg = json.loads((local / "adapter_config.json").read_text())
        r_nom, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
        a_keys = sorted(k for k in weights if ".lora_A" in k)
        energies = {}
        for a in a_keys:
            b = weights[a.replace(".lora_A", ".lora_B")].float()
            qb, rb = torch.linalg.qr(b, mode="reduced")
            qa, ra = torch.linalg.qr(weights[a].float().T, mode="reduced")
            energies[a] = torch.linalg.svdvals(rb @ ra.T).square()

        def run(adapter_dir, name, disable=False):
            nonlocal peft_model
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(model, str(adapter_dir),
                                                       adapter_name=name)
            else:
                peft_model.load_adapter(str(adapter_dir), adapter_name=name)
            m = peft_model
            m.set_adapter(name)
            m.eval()
            hits = []
            ctx = m.disable_adapter() if disable else torch.no_grad()
            with ctx:
                for i in range(0, len(prompts), args.batch):
                    enc = tok(prompts[i:i + args.batch], return_tensors="pt",
                              padding=True, truncation=True,
                              max_length=args.max_length).to("cuda")
                    with torch.no_grad():
                        out = m.generate(**enc, max_new_tokens=args.new,
                                         do_sample=False, pad_token_id=tok.pad_token_id)
                    for j, seq in enumerate(out[:, enc["input_ids"].shape[1]:]):
                        txt = tok.decode(seq, skip_special_tokens=True).strip()
                        hits.append(score(metric, txt, targets[i + j]))
            m.delete_adapter(name)
            torch.cuda.empty_cache()
            # see cts_task_metrics.py: per-example scores are what a split-sample
            # oracle needs, and without them a dense-grid oracle is optimistic
            return sum(hits) / max(len(hits), 1), hits

        m_base, pe_base = run(local, "floor", disable=True)
        m_orig, pe_orig = run(local, "orig")
        rec = {"task": task, "short": short, "metric": metric,
               "domain": items[0]["domain"], "n": len(prompts), "r": r_nom,
               "prompt_start": args.start,
               "cue": args.cue, "metric_base": m_base, "metric_orig": m_orig,
               "headroom": m_orig - m_base, "per_example_base": pe_base,
               "per_example_orig": pe_orig, "variants": {}}
        print(f"{short:26s} [{metric}] n={len(prompts)} base={m_base:.4f} "
              f"full={m_orig:.4f} head={m_orig - m_base:+.4f}", flush=True)

        rules = ([(f"e{round(t * 100):02d}", t, None) for t in args.thresholds]
                 + [(f"k{k:02d}", None, k) for k in args.ranks])
        for label, allocation in dense_allocations:
            key = task if task in allocation else short
            if key not in allocation:
                continue
            pattern = allocation[key]["module_ranks"]
            if len(pattern) != len(a_keys):
                raise ValueError(f"{short}: allocation has {len(pattern)} module ranks, "
                                 f"adapter has {len(a_keys)} modules")
            rules.append((label, None, pattern))
        rules = rules[args.rule_shard::args.rule_shards]
        for label, threshold, fixed in rules:
            out_w, kept, drop, k_tot, n_tot = dict(weights), 0.0, 0.0, 0, 0
            for module_index, a in enumerate(a_keys):
                b = a.replace(".lora_A", ".lora_B")
                fixed_rank = (fixed[module_index] if isinstance(fixed, list) else fixed)
                na, nb, k, _ = truncated_factors(weights[a], weights[b],
                                                 threshold or 0.0, fixed_rank=fixed_rank)
                out_w[a], out_w[b] = na, nb
                e = energies[a]
                kept += float(e[:k].sum())
                drop += float(e[k:].sum())
                k_tot += k
                n_tot += e.numel()
            dest = args.work / f"{short}-{label}"
            dest.mkdir(parents=True, exist_ok=True)
            save_file(out_w, dest / "adapter_model.safetensors")
            (dest / "adapter_config.json").write_text(
                (local / "adapter_config.json").read_text())
            m_c, pe_c = run(dest, label)
            head = m_orig - m_base
            rec["variants"][label] = {
                "L_W": math.sqrt(drop / (kept + drop)) if kept + drop else 0.0,
                "rank_frac": k_tot / max(n_tot, 1), "metric_value": m_c,
                "per_example": pe_c,
                "retained": (m_c - m_base) / head if head > 0 else float("nan")}
            v = rec["variants"][label]
            print(f"  {label}: L_W={v['L_W']:.3f} rank={v['rank_frac']:.2f} "
                  f"m={m_c:.4f} u={v['retained']:+.3f}", flush=True)
        results.append(rec)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {args.output} ({len(results)} tasks)")


if __name__ == "__main__":
    main()
