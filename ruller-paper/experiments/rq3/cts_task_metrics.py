#!/usr/bin/env python3
"""Task metrics for the controlled Lots-of-LoRAs pool.

The pool was introduced as the population where everything except the task is
held fixed -- one base model, one rank, one training pipeline -- and it has been
scored only by divergence, on the grounds that a task metric was not available.
It is: every adapter's dataset repository ships an `output` field of acceptable
answers and a held-out `test` split, for 30 of the 33 pool members.

That makes the strongest version of the warm-up available. LoRA Land shows the
same threshold doing different things to ten adapters that differ in every
respect; this shows it on thirty that differ only in what they were trained to
do.

Scoring follows the Super-NaturalInstructions convention: ROUGE-L against the
best-matching reference, which degrades continuously rather than falling off a
cliff the way exact match does, with exact-match-any reported alongside. The
un-adapted model is evaluated on the same prompts so retention can be measured
against the gain the adapter actually provides.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compress_adapter import truncated_factors  # noqa: E402


def normalise(text: str) -> list[str]:
    """Casefold and strip punctuation, keeping every script.

    The pool is Super-NaturalInstructions and a large share of its tasks are
    translation into non-Latin scripts, so an ASCII-only filter deletes the whole
    answer: the prediction and the reference both become empty, exact match then
    scores a false 1.0 on two empty strings while ROUGE-L scores 0.
    """
    text = "".join(" " if unicodedata.category(c).startswith("P")
                   or unicodedata.category(c) in ("Zs", "Cc") else c
                   for c in text.casefold())
    return text.split()


def rouge_l(pred: str, refs: list[str]) -> float:
    """LCS-based F1 against the best reference, the Super-NI metric."""
    p = normalise(pred)
    best = 0.0
    for ref in refs:
        r = normalise(ref)
        if not p or not r:
            continue
        # longest common subsequence, O(|p||r|) on short strings
        prev = [0] * (len(r) + 1)
        for a in p:
            cur = [0]
            for j, b in enumerate(r):
                cur.append(prev[j] + 1 if a == b else max(cur[j], prev[j + 1]))
            prev = cur
        lcs = prev[-1]
        if lcs:
            prec, rec = lcs / len(p), lcs / len(r)
            best = max(best, 2 * prec * rec / (prec + rec))
    return best


def exact(pred: str, refs: list[str]) -> float:
    p = " ".join(normalise(pred))
    if not p:                       # an empty prediction matches nothing, including
        return 0.0                  # a reference that also normalises to empty
    return float(any(p == " ".join(normalise(r)) for r in refs if r.strip()))


def references(row) -> list[str]:
    out = row["output"]
    if isinstance(out, str):
        try:
            parsed = json.loads(out.replace("'", '"'))
            return parsed if isinstance(parsed, list) else [out]
        except Exception:
            return [out]
    return list(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--strengths", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--thresholds", type=float, nargs="*",
                    default=[0.99, 0.95, 0.90, 0.80, 0.70, 0.50],
                    help="pass with no values to run the rank column only; the two "
                         "columns are independent so they can be separate jobs")
    ap.add_argument("--ranks", type=int, nargs="*", default=[2, 4, 6, 8, 10, 12])
    ap.add_argument("--prompts", type=int, default=100)
    ap.add_argument("--new", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--truncation-side", choices=("left", "right"), default="left",
                    help="SuperNI prompts end with the instance; right truncation "
                         "removes it and collapses the prompts to a shared preamble")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--min-prompts", type=int, default=30,
                    help="skip adapters whose test split is too small to score")
    ap.add_argument("--splits", nargs="+", default=["test"],
                    help="evaluation splits, concatenated in the order given. The "
                         "eight lowest-strength adapters in this pool come from tasks "
                         "with 77-200 instances in total, so an 80/10/10 split leaves "
                         "a test split of 10-25 rows -- below what a ROUGE-L mean can "
                         "resolve. Their `valid` split is the same size and is not "
                         "part of the adapter's training data, so `--splits test valid` "
                         "doubles the evaluation set without touching what was trained "
                         "on. The split actually used is recorded per adapter.")
    ap.add_argument("--adapters", nargs="*", default=[],
                    help="restrict to these adapter names, for topping up a pool "
                         "without re-running what is already measured")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--rule-shard", type=int, default=0,
                    help="split an adapter's variant list across jobs; base and full "
                         "are still measured in each, so use only when the variant "
                         "list is long")
    ap.add_argument("--rule-shards", type=int, default=1)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--allocation", type=Path, nargs="*", default=None,
                    help="dense allocation JSON(s). One file keeps the label `dense`; "
                         "several are labelled by budget so a whole allocation curve "
                         "can be scored in one job, paying for base and full once "
                         "instead of once per curve point.")
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

    from datasets import concatenate_datasets, load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    api = HfApi()
    ds_by_task = {}
    for info in api.list_datasets(author="Lots-of-LoRAs"):
        slug = info.id.split("/")[-1]
        if slug.startswith("task"):
            digits = "".join(c for c in slug[4:] if c.isdigit())
            if digits:
                ds_by_task.setdefault(int(digits), info.id)

    entries = sorted(json.loads(args.strengths.read_text()), key=lambda r: r["mean"])
    if args.adapters:
        want = set(args.adapters)
        entries = [e for e in entries if e["adapter"] in want]
        missing = want - {e["adapter"] for e in entries}
        if missing:
            raise SystemExit(f"not in the strength file: {sorted(missing)}")
    entries = entries[args.shard::args.shards]
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.truncation_side = args.truncation_side
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16).to("cuda").eval()
    print(f"base loaded; shard {args.shard}/{args.shards}, {len(entries)} adapters",
          flush=True)

    peft_model, results = None, []
    for entry in entries:
        task = int(entry["adapter"].replace("task", ""))
        if task not in ds_by_task:
            print(f"skip task{task:04d}: no dataset", flush=True)
            continue
        try:
            local = Path(snapshot_download(
                f"Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task{task}",
                local_dir=args.work / f"task{task:04d}"))
            avail = load_dataset(ds_by_task[task])
        except Exception as exc:
            print(f"skip task{task:04d}: {type(exc).__name__} {exc}"[:140], flush=True)
            continue
        wanted = [sp for sp in args.splits if sp in avail]
        if not wanted:
            wanted = ["test"] if "test" in avail else ["train"]
        split = "+".join(wanted)
        rows = (avail[wanted[0]] if len(wanted) == 1
                else concatenate_datasets([avail[sp] for sp in wanted]))
        if "output" not in rows.column_names:
            print(f"skip task{task:04d}: no output field", flush=True)
            continue
        keep = [r for r in rows.select(range(min(len(rows), args.prompts * 2)))
                if r.get("input", "").strip() and references(r)][: args.prompts]
        if len(keep) < args.min_prompts:
            print(f"skip task{task:04d}: only {len(keep)} scorable rows", flush=True)
            continue
        prompts = [r["input"] for r in keep]
        refs = [references(r) for r in keep]

        weights = load_file(local / "adapter_model.safetensors")
        a_keys = sorted(k for k in weights if ".lora_A." in k)
        energies = {}
        for a_key in a_keys:
            b = weights[a_key.replace(".lora_A.", ".lora_B.")].float()
            a = weights[a_key].float()
            qb, rb = torch.linalg.qr(b, mode="reduced")
            qa, ra = torch.linalg.qr(a.T, mode="reduced")
            energies[a_key] = torch.linalg.svdvals(rb @ ra.T).square()

        def build(label, threshold, fixed):
            out, kept, drop, k_tot, n_tot = dict(weights), 0.0, 0.0, 0, 0
            for module_index, a_key in enumerate(a_keys):
                b_key = a_key.replace(".lora_A.", ".lora_B.")
                fixed_rank = (fixed[module_index] if isinstance(fixed, list) else fixed)
                na, nb, k, _ = truncated_factors(weights[a_key], weights[b_key],
                                                 threshold or 0.0, fixed_rank=fixed_rank)
                out[a_key], out[b_key] = na, nb
                e = energies[a_key]
                kept += float(e[:k].sum())
                drop += float(e[k:].sum())
                k_tot += k
                n_tot += e.numel()
            dest = args.work / f"task{task:04d}-{label}"
            dest.mkdir(parents=True, exist_ok=True)
            save_file(out, dest / "adapter_model.safetensors")
            (dest / "adapter_config.json").write_text(
                (local / "adapter_config.json").read_text())
            L = math.sqrt(drop / (kept + drop)) if kept + drop > 0 else 0.0
            return dest, L, k_tot / max(n_tot, 1)

        def score_with(adapter_dir, name, disable=False):
            nonlocal peft_model
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(model, str(adapter_dir),
                                                       adapter_name=name)
            else:
                peft_model.load_adapter(str(adapter_dir), adapter_name=name)
            m = peft_model
            m.set_adapter(name)
            m.eval()
            rl, em = [], []
            ctx = m.disable_adapter() if disable else torch.no_grad()
            with ctx:
                for i in range(0, len(prompts), args.batch):
                    enc = tok(prompts[i:i + args.batch], return_tensors="pt",
                              padding=True, truncation=True,
                              max_length=args.max_length).to("cuda")
                    with torch.no_grad():
                        got = m.generate(**enc, max_new_tokens=args.new,
                                         do_sample=False, pad_token_id=tok.pad_token_id)
                    for j, seq in enumerate(got[:, enc["input_ids"].shape[1]:]):
                        txt = tok.decode(seq, skip_special_tokens=True)
                        rl.append(rouge_l(txt, refs[i + j]))
                        em.append(exact(txt, refs[i + j]))
            m.delete_adapter(name)
            torch.cuda.empty_cache()
            # per-example scores make a split-sample oracle possible: choosing k on
            # the same examples the oracle is scored on inflates it by the maximum
            # of N noisy draws, which grows with the number of candidate ranks
            return sum(rl) / len(rl), sum(em) / len(em), rl

        m_base, e_base, pe_base = score_with(local, "floor", disable=True)
        m_orig, e_orig, pe_orig = score_with(local, "orig")
        rec = {"adapter": entry["adapter"], "task": task, "S": entry["mean"],
               "split": split, "n": len(prompts), "metric_base": m_base,
               "exact_base": e_base, "metric_orig": m_orig, "exact_orig": e_orig,
               "per_example_base": pe_base, "per_example_orig": pe_orig,
               "variants": {}}
        print(f"task{task:04d} S={entry['mean']:.4f} n={len(prompts)} "
              f"base={m_base:.4f} full={m_orig:.4f} (exact {e_base:.3f}/{e_orig:.3f})",
              flush=True)

        rules = ([(f"e{round(t * 100):02d}", t, None) for t in args.thresholds]
                 + [(f"k{k:02d}", None, k) for k in args.ranks])
        for label, allocation in dense_allocations:
            if entry["adapter"] not in allocation:
                continue
            pattern = allocation[entry["adapter"]]["module_ranks"]
            if len(pattern) != len(a_keys):
                raise ValueError(f"{entry['adapter']}: dense pattern/module mismatch")
            rules.append((label, None, pattern))
        rules = rules[args.rule_shard::args.rule_shards]
        for label, threshold, fixed in rules:
            dest, L, frac = build(label, threshold, fixed)
            m_c, e_c, pe_c = score_with(dest, label)
            head = m_orig - m_base
            rec["variants"][label] = {
                "L_W": L, "rank_frac": frac, "metric": m_c, "exact": e_c,
                "per_example": pe_c,
                "retained": (m_c - m_base) / head if head > 0 else float("nan")}
            print(f"  {label}: L_W={L:.3f} rank={frac:.2f} rougeL={m_c:.4f} "
                  f"u={rec['variants'][label]['retained']:+.3f}", flush=True)
        results.append(rec)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {args.output} ({len(results)} adapters)")


if __name__ == "__main__":
    main()
