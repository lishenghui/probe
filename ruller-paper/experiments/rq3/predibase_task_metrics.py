#!/usr/bin/env python3
"""Task-metric evaluation of energy truncation on the Lora Land population.

Every divergence in this paper so far is a perturbation measure: it establishes
that behaviour changes, not that the change costs anything.  This closes that
gap.  Predibase's Lora Land publishes 27 adapters that all share
mistralai/Mistral-7B-v0.1 with r=8, alpha=16 and {q_proj, v_proj} as the only
target modules, so base model, rank, scaling and adapted sites are held fixed
and the task is the single free variable -- and each adapter's model card names
the dataset and gives a worked prompt, so the real metric is recoverable.

For each adapter we truncate at tau in {0.99, 0.95, 0.90}, then score the
original and every truncated variant on the task's own held-out split.

Task metrics alone turned out not to be enough: ViGGO moves 137/200 outputs for
a net metric change of +0.0008, and WikiSQL absorbs a D_JS of 0.27 without
losing a point, so a task score both hides perturbation and does not follow it.
The mechanism probe below therefore measures, per generated token, how far
compression pushes the decision and how far it had to push.  Both adapters are
teacher-forced on the *original* greedy continuation, so a disagreement at step
t is a decision-boundary crossing at t and not error propagated from step t-1 --
free-running generation is still what scores the task, but it cannot separate
those two.

Prompt templates are transcribed from the model cards.  A wrong template shows
up as a near-chance score for the *uncompressed* adapter, so `metric_orig` is
also the check on the harness: adapters that fail it are reported and excluded
rather than silently contributing a large apparent degradation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compress_adapter import truncated_factors  # noqa: E402
from sweep_cts_compression import js_divergence  # noqa: E402

# ---------------------------------------------------------------- templates --
# Each entry: dataset spec, split, prompt builder, gold builder, decode budget,
# and how many random-guess points the metric floors at (for the sanity check).


_VIGGO_SHOTS = (
    "Here are two examples of meaning representations being translated into plain English:\n\n"
    'Example representation:  "request(release_year[2014], specifier[terrible])"\n'
    'Example output: "Were there even any terrible games in 2014?"\n\n'
    'Example representation: "give_opinion(name[Little Nightmares], rating[good], '
    'genres[adventure, platformer, puzzle], player_perspective[side view])"\n'
    'Example output: "Adventure games that combine platforming and puzzles can be frustrating '
    "to play, but the side view perspective is perfect for them. That's why I enjoyed playing "
    'Little Nightmares."\n\n'
    "Using the previous examples as guidelines, please translate the following representation "
    "into plain English:\nRepresentation: ")

_NER_KEYS = {1: "person", 2: "person", 3: "organization", 4: "organization",
             5: "location", 6: "location", 7: "miscellaneous", 8: "miscellaneous"}


def _ner_gold(row) -> str:
    """Rebuild the card's JSON payload from CoNLL BIO tags."""
    out = {"person": [], "organization": [], "location": [], "miscellaneous": []}
    cur, key = [], None
    for tokrow, tag in zip(row["tokens"], row["ner_tags"]):
        if tag == 0:
            if cur:
                out[key].append(" ".join(cur))
            cur, key = [], None
            continue
        k = _NER_KEYS[tag]
        if tag % 2 == 1 or k != key:          # B- tag, or a switch of category
            if cur:
                out[key].append(" ".join(cur))
            cur, key = [tokrow], k
        else:
            cur.append(tokrow)
    if cur:
        out[key].append(" ".join(cur))
    return json.dumps(out)


def _wikisql_schema(row) -> str:
    t = row["table"]
    return str({"header": t["header"], "types": t["types"], "rows": t["rows"]})


def _viggo_mr(row):
    for k in ("meaning_representation", "mr", "input"):
        if k in row:
            return row[k]
    raise KeyError("no meaning representation field")


def _viggo_target(row):
    for k in ("target", "ref", "output"):
        if k in row:
            return row[k]
    raise KeyError("no target field")


def _e2e_ref(row):
    for k in ("human_reference", "ref", "target"):
        if k in row:
            return row[k]
    raise KeyError("no reference field")


def _hs_endings(row):
    return "[" + ", ".join(repr(e) for e in row["endings"]) + "]"


TASKS = {
    "glue_sst2": dict(
        ds=("nyu-mll/glue", "sst2"), split="validation", new=4, chance=0.5,
        prompt=lambda r: ("Given the following sentence:\n\n" + r["sentence"] +
                          "\n\nRespond with 0 if the sentiment of the sentence is negative "
                          "and 1 if the sentiment of the sentence is positive."),
        gold=lambda r: str(r["label"])),
    # Five tasks added from the published model cards, which give each adapter's
    # exact Sample input and Sample output. Copying the template verbatim -- including
    # the trailing separator, which is what CoLA and MRPC needed -- avoids guessing
    # the training format, the failure mode that produced below-chance scores here
    # before. Tasks whose card points at Kaggle or GitHub (covid, jigsaw, legal,
    # customer_support) are not included: the data is not automatically loadable.
    "drop": dict(
        ds=("ucinlp/drop",), split="validation", new=12, chance=0.0,
        prompt=lambda r: ("Given a passage, you need to accurately identify and extract "
                          "relevant spans of text that answer specific questions. Provide "
                          "concise and coherent responses based on the information present "
                          "in the passage.\n\n### Passage: " + r["passage"] +
                          "\n### Question: " + r["question"] + "\n### Answer:"),
        gold=lambda r: (r["answers_spans"]["spans"] or [""])[0]),
    "glue_stsb": dict(
        ds=("nyu-mll/glue", "stsb"), split="validation", new=6, chance=0.0,
        prompt=lambda r: ("You are given two sentences below, Sentence 1 and Sentence 2. "
                          "Please determine, on a scale from 0 to 5, with 0 being least "
                          "similar and 5 being most similar, how similar the two sentences "
                          "are:\n\n### Sentence 1: " + r["sentence1"] +
                          "\n\n### Sentence 2: " + r["sentence2"] +
                          "\n\n### Similarity Score: "),
        gold=lambda r: f"{r['label']:.1f}"),
    "cnn": dict(
        ds=("abisee/cnn_dailymail", "3.0.0"), split="validation", new=96, chance=0.0,
        prompt=lambda r: ("You are given a news article below. Please summarize the "
                          "article, including only its highlights.\n\n### Article: " +
                          r["article"] + "\n\n### Summary: "),
        gold=lambda r: r["highlights"]),
    "agnews_explained": dict(
        ds=("fancyzhx/ag_news",), split="test", new=64, chance=0.25,
        prompt=lambda r: ("You are given a news article below. Please classify it into one "
                          "of the following categories: World, Sports, Business, Sci/Tech. "
                          "Respond with a JSON object.\n\n### Article: " + r["text"] +
                          "\n\n### JSON Response"),
        gold=lambda r: ["World", "Sports", "Business", "Sci/Tech"][r["label"]]),
    "hellaswag_processed": dict(
        ds=("Rowan/hellaswag",), split="validation", new=32, chance=0.0,
        prompt=lambda r: ("You are provided with an incomplete passage below. Please read "
                          "the passage and then finish it with an appropriate response. For "
                          "example:\n\n### Passage: My friend and I think alike. We\n\n"
                          "### Ending: often finish each other's sentences.\n\nNow please "
                          "finish the following passage:\n\n### Passage: " + r["ctx"] +
                          "\n\n### Ending: "),
        gold=lambda r: r["endings"][int(r["label"])]),
    # CoLA and MRPC keep a trailing space after the label cue because the published
    # model cards do: "### Label: ". Without it both adapters emit an empty string on
    # many rows and score below chance -- 0.380 and 0.370 against chance rates of 0.69
    # and 0.68 -- while with it they reach 0.860 and 0.930. The other eight tasks were
    # measured both ways and are insensitive, so their prompts are left untouched.
    "glue_cola": dict(
        ds=("nyu-mll/glue", "cola"), split="validation", new=4, chance=0.69,
        prompt=lambda r: ("Determine if the sentence below is syntactically and semantically "
                          'correct. If it is syntactically and semantically correct, respond "1". '
                          'Otherwise, respond "0".\n\nSentence: ' + r["sentence"] + "\n\nLabel: "),
        gold=lambda r: str(r["label"])),
    "glue_mnli": dict(
        ds=("nyu-mll/glue", "mnli"), split="validation_matched", new=4, chance=0.35,
        prompt=lambda r: ("You are given a premise and a hypothesis below. If the premise entails "
                          "the hypothesis, return 0. If the premise contradicts the hypothesis, "
                          "return 2. Otherwise, if the premise does neither, return 1.\n\n"
                          "### Premise: " + r["premise"] + "\n\n### Hypothesis: " +
                          r["hypothesis"] + "\n\n### Label:"),
        gold=lambda r: str(r["label"])),
    "glue_mrpc": dict(
        ds=("nyu-mll/glue", "mrpc"), split="validation", new=4, chance=0.68,
        prompt=lambda r: ("You are given two sentences below, Sentence 1 and Sentence 2. If the "
                          "two sentences are semantically equivalent, please return 1. Otherwise, "
                          "please return 0.\n\n### Sentence 1: " + r["sentence1"] +
                          "\n\n### Sentence 2: " + r["sentence2"] + "\n\n### Label: "),
        gold=lambda r: str(r["label"])),
    "glue_qnli": dict(
        ds=("nyu-mll/glue", "qnli"), split="validation", new=4, chance=0.5,
        prompt=lambda r: ("You are provided a question and a corresponding response below. If the "
                          "response properly answers the question, please return 0. Otherwise, "
                          "please return 1.\n\n### Question: " + r["question"] +
                          "\n\n### Response: " + r["sentence"] + "\n\n### Label:"),
        gold=lambda r: str(r["label"])),
    "glue_qqp": dict(
        ds=("nyu-mll/glue", "qqp"), split="validation", new=4, chance=0.63,
        prompt=lambda r: ("You are given two questions below, Question 1 and Question 2. If the "
                          "two questions are semantically equivalent, please return 1. Otherwise, "
                          "please return 0.\n\n### Question 1: " + r["question1"] +
                          "\n\n### Question 2: " + r["question2"] + "\n\n### Label:"),
        gold=lambda r: str(r["label"])),
    "dbpedia": dict(
        ds=("fancyzhx/dbpedia_14",), split="test", new=5, chance=1 / 14,
        prompt=lambda r: ("You are given the title and the body of an article below. Please "
                          "determine the type of the article.\n### Title: " + r["title"] +
                          "\n\n### Body: " + r["content"].strip() + "\n\n### Article Type:"),
        gold=lambda r: str(r["label"])),
    "hellaswag": dict(
        ds=("Rowan/hellaswag",), split="validation", new=5, chance=0.25,
        prompt=lambda r: ("You are provided with an incomplete passage below as well as 4 endings "
                          "in quotes and separated by commas, with only one of them being the "
                          "correct ending. Treat the endings as being labelled 0, 1, 2, 3 in "
                          "order. Please respond with the number corresponding to the correct "
                          "ending for the passage.\n\n### Passage: " + r["ctx"] +
                          "\n\n### Endings: " + _hs_endings(r) + "\n\n### Correct Ending:"),
        gold=lambda r: str(r["label"])),
    "gsm8k": dict(
        ds=("openai/gsm8k", "main"), split="test", new=256, chance=0.0,
        prompt=lambda r: "Please answer the following question: " + r["question"] + "\nAnswer:",
        gold=lambda r: r["answer"].split("####")[-1].strip()),
    "wikisql": dict(
        ds=("Salesforce/wikisql",), alt=[("Salesforce/wikisql", "refs/convert/parquet")],
        split="validation", new=64, chance=0.0,
        prompt=lambda r: ("Considering the provided database schema and associated query, "
                          "produce SQL code to retrieve the answer to the query.\n"
                          "### Database Schema: " + _wikisql_schema(r) +
                          "\n### Query: " + r["question"] + "\n### SQL:"),
        gold=lambda r: r["sql"]["human_readable"]),
    "conllpp": dict(
        ds=("ZihanWangKi/conllpp",), alt=[("ZihanWangKi/conllpp", "refs/convert/parquet")], split="test", new=96, chance=0.0,
        prompt=lambda r: ("Your task is a Named Entity Recognition (NER) task. Predict the "
                          "category of each entity, then place the entity into the list "
                          "associated with the  category in an output JSON payload. Below is an "
                          "example:\nInput: EU rejects German call to boycott British lamb . "
                          'Output: {"person": [], "organization": ["EU"], "location": [], '
                          '"miscellaneous": ["German", "British"]}\nNow, complete the task.\n'
                          "Input: " + " ".join(r["tokens"]) + " Output:"),
        gold=_ner_gold),
    "viggo": dict(
        ds=("GEM/viggo",), alt=[("GEM/viggo", "refs/convert/parquet")], split="validation", new=72, chance=0.0,
        prompt=lambda r: _VIGGO_SHOTS + _viggo_mr(r) + "\nOutput:",
        gold=_viggo_target),
    "e2e_nlg": dict(
        ds=("tuetschek/e2e_nlg",), alt=[("tuetschek/e2e_nlg", "refs/convert/parquet")], split="validation", new=80, chance=0.0,
        prompt=lambda r: ("You are given a meaning representation below. Please translate it "
                          "into plain English. Here is an example:\n\n"
                          "### Meaning Representation: name[Blue Spice], eatType[coffee shop], "
                          "area[city centre]\n\n### Plain English: A coffee shop in the city "
                          "centre area called Blue Spice.\n\nNow please translate the following "
                          "meaning representation:\n\n### Meaning Representation: " +
                          _viggo_mr(r) + "\n\n### Plain English:"),
        gold=_e2e_ref),
}


def score(task: str, pred: str, gold: str) -> float:
    """Exact match, normalised per task family."""
    p, g = pred.strip(), gold.strip()
    if task == "gsm8k":
        nums = re.findall(r"-?[\d,]*\.?\d+", p.split("####")[-1] if "####" in p else p)
        if not nums:
            return 0.0
        try:
            return float(abs(float(nums[-1].replace(",", "")) - float(g.replace(",", ""))) < 1e-4)
        except ValueError:
            return 0.0
    if task == "wikisql":
        norm = lambda x: re.sub(r"\s+", " ", x.strip().rstrip(";").lower())
        return float(norm(p.split("\n")[0]) == norm(g))
    if task == "conllpp":
        try:
            pj = json.loads(p[p.index("{"):p.index("}") + 1])
        except Exception:
            return 0.0
        try:
            gj = json.loads(g)
        except Exception:
            return 0.0
        # entity-level micro F1 over the four categories
        tp = fp = fn = 0
        for k in gj:
            pv, gv = list(pj.get(k, []) or []), list(gj[k])
            for e in pv:
                if e in gv:
                    gv.remove(e); tp += 1
                else:
                    fp += 1
            fn += len(gv)
        if tp == 0:
            return 0.0
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        return 2 * prec * rec / (prec + rec)
    if task == "glue_stsb":
        # a similarity score in [0, 5]: the published protocol scores regression by
        # 1 - mean absolute error, so exact string equality would be far too strict
        try:
            return max(0.0, 1.0 - abs(float(p.split()[0]) - float(g)) / 5.0)
        except (ValueError, IndexError):
            return 0.0
    if task == "agnews_explained":
        # the adapter answers with a JSON object; the label lives in text_label
        try:
            lab = json.loads(p[p.index("{"):p.index("}") + 1]).get("text_label", "")
        except Exception:
            m = re.search(r'"text_label"\s*:\s*"([^"]*)"', p)
            lab = m.group(1) if m else ""
        return float(lab.strip().lower() == g.strip().lower())
    if task in ("cnn", "hellaswag_processed", "drop"):
        # summarisation and free-form completion: the same token-F1 used for the
        # other generation tasks, since there is no single correct string
        pt, gt = p.lower().split(), g.lower().split()
        if not pt or not gt:
            return 0.0
        common = 0
        rest = list(gt)
        for w in pt:
            if w in rest:
                rest.remove(w); common += 1
        if not common:
            return 0.0
        prec, rec = common / len(pt), common / len(gt)
        return 2 * prec * rec / (prec + rec)
    if task in ("viggo", "e2e_nlg"):          # token-F1, generation has no single right string
        pt, gt = p.lower().split(), g.lower().split()
        if not pt or not gt:
            return 0.0
        common = 0
        pool = list(gt)
        for w in pt:
            if w in pool:
                pool.remove(w)
                common += 1
        if common == 0:
            return 0.0
        prec, rec = common / len(pt), common / len(gt)
        return 2 * prec * rec / (prec + rec)
    first = p.split()[0].rstrip(".,:;") if p.split() else ""
    try:                                       # hellaswag's targets are floats ("3.0")
        return float(abs(float(first) - float(g)) < 1e-6)
    except ValueError:
        return float(first == g)


def spectrum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a32, b32 = a.float(), b.float()
    qb, rb = torch.linalg.qr(b32, mode="reduced")
    qa, ra = torch.linalg.qr(a32.T, mode="reduced")
    return torch.linalg.svdvals(rb @ ra.T)


def base_key(a_key: str) -> str:
    name = re.sub(r"\.lora_A\.(default\.)?weight$", ".weight", a_key)
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="mistralai/Mistral-7B-v0.1")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", default=sorted(TASKS))
    ap.add_argument("--thresholds", type=float, nargs="*", default=[0.99, 0.95, 0.90])
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 keeps greedy decoding, which is deterministic and makes "
                         "every difference attributable to the compression. A positive "
                         "value samples instead, so the score estimates E[metric] over "
                         "the model's own output distribution rather than the score of "
                         "one argmax trajectory -- use with --samples.")
    ap.add_argument("--samples", type=int, default=1,
                    help="draws per prompt when sampling; the reported metric is their "
                         "mean and the record also carries the per-draw spread")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1,
                    help="split the variant list across jobs. Variants are "
                         "independent, but every shard re-pays the base-model load "
                         "and the orig/base evaluations, so the speedup saturates "
                         "well below the shard count -- past about four shards the "
                         "fixed cost dominates.")
    ap.add_argument("--ranks", type=int, nargs="*", default=[],
                    help="also emit fixed-rank variants keeping exactly k directions "
                         "per module, the other compression rule in common use")
    ap.add_argument("--examples", type=int, default=200)
    ap.add_argument("--example-start", type=int, default=0,
                    help="offset within the task split, allowing unlabeled calibration "
                         "prompts to be disjoint from the evaluation prefix")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--allocation", type=Path, nargs="*", default=None,
                    help="dense allocation JSON(s). One file keeps the label `dense` "
                         "for backward compatibility; several are labelled by their "
                         "budget, so a whole allocation curve can be scored in one "
                         "job. That matters here: base and full are re-measured per "
                         "job, so evaluating 25 curve points as 25 jobs pays for "
                         "them 25 times, which on this pool is most of the cost.")
    ap.add_argument("--token-output", type=Path, default=None,
                    help="directory for the per-token npz probes "
                         "(default: <output stem>_tokens/); --no-margin-probe to skip")
    ap.add_argument("--margin-batch", type=int, default=4,
                    help="probe holds two full [B, T, V] logit tensors, so keep this small")
    ap.add_argument("--margin-steps", type=int, default=0,
                    help="cap on generated tokens probed per example (0 = all)")
    ap.add_argument("--no-margin-probe", action="store_true")
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
    tok_dir = args.token_output or args.output.with_suffix("").with_name(
        args.output.stem + "_tokens")
    if not args.no_margin_probe:
        tok_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cards = json.loads((Path(__file__).parent / "predibase_cards.json").read_text()) \
        if (Path(__file__).parent / "predibase_cards.json").is_file() else {}

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"                 # decoder-only generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16).to("cuda").eval()
    norms = {n: float(p.detach().float().norm()) for n, p in model.named_parameters()}
    print(f"base loaded ({len(norms)} tensors)", flush=True)

    peft_model, results = None, []
    for task in args.tasks:
        spec = TASKS.get(task)
        if spec is None or spec["prompt"] is None:
            print(f"skip {task}: no template", flush=True)
            continue
        rows = None
        attempts = [(spec["ds"], None)] + [((r,), rev) for r, rev in spec.get("alt", [])]
        for ds_args, rev in attempts:
            try:
                end = args.example_start + args.examples
                kw = {"split": f"{spec['split']}[{args.example_start}:{end}]"}
                if rev:
                    kw["revision"] = rev
                rows = load_dataset(*ds_args, **kw)
                if rev:
                    print(f"  {task}: loaded via {ds_args[0]} @ {rev}", flush=True)
                break
            except Exception as exc:
                last = f"{type(exc).__name__} {str(exc)[:80]}"
        if rows is None:
            print(f"skip {task}: no loadable source ({last})", flush=True)
            continue
        try:
            local = Path(snapshot_download(f"predibase/{task}", local_dir=args.work / task))
        except Exception as exc:
            print(f"skip {task}: adapter {type(exc).__name__}", flush=True)
            continue

        prompts = [spec["prompt"](r) for r in rows]
        golds = [spec["gold"](r) for r in rows]

        weights = load_file(local / "adapter_model.safetensors")
        cfg = json.loads((local / "adapter_config.json").read_text())
        rank, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
        scale = alpha / (math.sqrt(rank) if cfg.get("use_rslora") else rank)

        num = den = 0.0
        spectra = {}
        for a_key in sorted(k for k in weights if ".lora_A." in k):
            b_key = a_key.replace(".lora_A.", ".lora_B.")
            w = norms.get(base_key(a_key))
            if w is None or b_key not in weights:
                continue
            sv = spectrum(weights[a_key], weights[b_key]) * scale
            spectra[a_key] = sv
            num += float(sv.square().sum())
            den += w * w
        if not spectra:
            print(f"skip {task}: no matched modules", flush=True)
            continue
        S = math.sqrt(num / den)

        def run(adapter_dir, name):
            if peft_model_ref[0] is None:
                peft_model_ref[0] = PeftModel.from_pretrained(model, str(adapter_dir),
                                                              adapter_name=name)
            else:
                peft_model_ref[0].load_adapter(str(adapter_dir), adapter_name=name)
            m = peft_model_ref[0]
            m.set_adapter(name)
            m.eval()
            hits, preds, gen_ids = [], [], []
            draws = max(args.samples, 1) if args.temperature > 0 else 1
            per_draw = [[] for _ in range(draws)]
            for i in range(0, len(prompts), args.batch):
                chunk = prompts[i:i + args.batch]
                enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                          max_length=args.max_length).to("cuda")
                gen_kw = dict(max_new_tokens=spec["new"], pad_token_id=tok.pad_token_id)
                if args.temperature > 0:
                    gen_kw.update(do_sample=True, temperature=args.temperature,
                                  top_p=1.0, num_return_sequences=draws)
                else:
                    gen_kw.update(do_sample=False)
                with torch.no_grad():
                    out = m.generate(**enc, **gen_kw)
                if draws > 1:
                    gen_all = out[:, enc["input_ids"].shape[1]:]
                    for j in range(len(chunk)):
                        for d in range(draws):
                            txt = tok.decode(gen_all[j * draws + d], skip_special_tokens=True)
                            per_draw[d].append(score(task, txt, golds[i + j]))
                    out = out[::draws]      # keep the first draw for the prediction trace
                gen = out[:, enc["input_ids"].shape[1]:]
                for j, seq in enumerate(gen):
                    text = tok.decode(seq, skip_special_tokens=True)
                    preds.append(text)
                    hits.append(score(task, text, golds[i + j]))
                    ids = seq.tolist()
                    if tok.eos_token_id in ids:      # keep the EOS: emitting it is a decision
                        ids = ids[:ids.index(tok.eos_token_id) + 1]
                    gen_ids.append(ids)
            m.delete_adapter(name)
            torch.cuda.empty_cache()
            if draws > 1:
                means = [sum(v) / max(len(v), 1) for v in per_draw]
                hits = [sum(d[i] for d in per_draw) / draws for i in range(len(per_draw[0]))]
                return (sum(means) / draws, preds, hits, gen_ids,
                        {"draw_means": means, "draw_spread": max(means) - min(means)})
            return sum(hits) / max(len(hits), 1), preds, hits, gen_ids, None

        def divergence(adapter_dir, ref_dir):
            """Teacher-forced D_JS between two adapters on the scored prompts."""
            m = peft_model_ref[0]
            m.load_adapter(str(ref_dir), adapter_name="djs_ref")
            m.load_adapter(str(adapter_dir), adapter_name="djs_var")
            m.eval()
            tot, seen = 0.0, 0
            for i in range(0, len(prompts), args.batch):
                enc = tok(prompts[i:i + args.batch], return_tensors="pt", padding=True,
                          truncation=True, max_length=args.max_length).to("cuda")
                mask = enc["attention_mask"].bool()
                with torch.no_grad():
                    m.set_adapter("djs_ref")
                    a = m(**enc).logits
                    m.set_adapter("djs_var")
                    b = m(**enc).logits
                js = js_divergence(a, b)[mask]
                tot += float(js.sum())
                seen += int(mask.sum())
                del a, b
            for n_ in ("djs_ref", "djs_var"):
                m.delete_adapter(n_)
            torch.cuda.empty_cache()
            return tot / max(seen, 1)

        def margin_probe(adapter_dir, ref_dir, gen_ids):
            """Per-token decision geometry under a shared original prefix.

            Both adapters are teacher-forced on x, y_<t taken from the *original*
            greedy generation, so at every step they are scored from the same
            decision state.  Returns, per generated token t:

              per_token_orig_margin  M_t  = z_orig[y_t] - max_{v != y_t} z_orig[v]
              per_token_comp_margin  ~M_t = z_comp[y_t] - max_{v != y_t} z_comp[v]
              per_token_js           J_t  = D_JS(softmax z_orig, softmax z_comp)

            M_t > 0 by construction (y_t is the original argmax), so the erosion
            E_t = M_t - ~M_t is how far compression pushed the decision and M_t is
            how far it had to push: ~M_t < 0 iff E_t > M_t iff the token flips.
            first_shared_prefix_flip is the earliest such t, or -1.
            """
            m = peft_model_ref[0]
            m.load_adapter(str(ref_dir), adapter_name="mp_ref")
            m.load_adapter(str(adapter_dir), adapter_name="mp_var")
            m.eval()
            ex, st, m_o, m_c, jsv = [], [], [], [], []
            first = []
            for i in range(0, len(prompts), args.margin_batch):
                chunk = prompts[i:i + args.margin_batch]
                gens = gen_ids[i:i + args.margin_batch]
                enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                          max_length=args.max_length)
                keep = enc["attention_mask"].bool()
                full = [enc["input_ids"][j][keep[j]].tolist() + list(gens[j])
                        for j in range(len(chunk))]
                T = max(len(f) for f in full)
                inp = torch.full((len(full), T), tok.pad_token_id, dtype=torch.long)
                att = torch.zeros((len(full), T), dtype=torch.long)
                for j, f in enumerate(full):          # left-pad: RoPE is relative, so the
                    inp[j, T - len(f):] = torch.tensor(f)   # uniform offset is harmless, and
                    att[j, T - len(f):] = 1           # it leaves no pad hole mid-sequence
                inp, att = inp.to("cuda"), att.to("cuda")
                with torch.no_grad():
                    m.set_adapter("mp_ref")
                    la = m(input_ids=inp, attention_mask=att).logits
                    m.set_adapter("mp_var")
                    lb = m(input_ids=inp, attention_mask=att).logits
                for j, g in enumerate(gens):
                    n = min(len(g), args.margin_steps) if args.margin_steps else len(g)
                    if n == 0:
                        first.append(-1)
                        continue
                    start = T - len(g)                # y_t sits at start + t ...
                    pos = torch.arange(start - 1, start - 1 + n, device=inp.device)
                    y = torch.tensor(g[:n], device=inp.device)   # ... predicted from start+t-1
                    za, zb = la[j, pos].float(), lb[j, pos].float()

                    def margin(z):
                        top2 = z.topk(2, dim=-1)
                        rival = torch.where(top2.indices[:, 0] == y,
                                            top2.values[:, 1], top2.values[:, 0])
                        return z.gather(1, y[:, None]).squeeze(1) - rival

                    ma, mb = margin(za), margin(zb)
                    jj = js_divergence(za, zb)
                    ex.extend([i + j] * n)
                    st.extend(range(n))
                    m_o.extend(ma.tolist())
                    m_c.extend(mb.tolist())
                    jsv.extend(jj.tolist())
                    flipped = (mb < 0).nonzero()
                    first.append(int(flipped[0]) if flipped.numel() else -1)
                del la, lb
            for n_ in ("mp_ref", "mp_var"):
                m.delete_adapter(n_)
            torch.cuda.empty_cache()
            return dict(example=ex, step=st, per_token_orig_margin=m_o,
                        per_token_comp_margin=m_c, per_token_js=jsv), first

        peft_model_ref = [peft_model]
        m_orig, preds_orig, hits_orig, gen_orig, spread_orig = run(local, "orig")
        # the un-adapted model on the same prompts: retention is measured against
        # the gain the adapter actually provides, not against a nominal chance level
        m_base = float("nan")
        if peft_model_ref[0] is not None:
            mm = peft_model_ref[0]
            mm.load_adapter(str(local), adapter_name="basefloor")
            mm.set_adapter("basefloor")
            with mm.disable_adapter():
                hits_b = []
                for i in range(0, len(prompts), args.batch):
                    enc = tok(prompts[i:i + args.batch], return_tensors="pt", padding=True,
                              truncation=True, max_length=args.max_length).to("cuda")
                    with torch.no_grad():
                        out = mm.generate(**enc, max_new_tokens=spec["new"], do_sample=False,
                                          pad_token_id=tok.pad_token_id)
                    for j, seq in enumerate(out[:, enc["input_ids"].shape[1]:]):
                        hits_b.append(score(task, tok.decode(seq, skip_special_tokens=True),
                                            golds[i + j]))
                m_base = sum(hits_b) / max(len(hits_b), 1)
            mm.delete_adapter("basefloor")
            torch.cuda.empty_cache()
        print(f"{task:14s} base(no adapter)={m_base:.4f}", flush=True)
        # Keep a few (prompt tail, prediction, gold) triples: when metric_orig lands
        # at chance these are the only way to tell a wrong template from a real result.
        samples = [{"prompt_tail": prompts[i][-120:], "pred": preds_orig[i][:60], "gold": golds[i]}
                   for i in range(min(6, len(preds_orig)))]
        record = {"adapter": task, "S": S, "modules": len(spectra), "n": len(prompts),
                  "example_start": args.example_start,
                  "chance": spec["chance"], "metric_orig": m_orig, "samples": samples,
                  "per_example_orig": hits_orig, "metric_base": m_base,
                  "temperature": args.temperature, "samples": args.samples,
                  "orig_draw_spread": (spread_orig or {}).get("draw_spread"),
                  "variants": {}}
        print(f"    sample preds: " + " | ".join(f"{x['pred'].strip()[:14]!r}->{x['gold']}"
                                                 for x in samples[:4]), flush=True)
        print(f"{task:14s} S={S:.4f}  orig={m_orig:.4f} (chance {spec['chance']:.2f})", flush=True)

        rules = ([(f"e{round(t * 100):02d}", t, None) for t in args.thresholds]
                 + [(f"k{k:02d}", None, k) for k in args.ranks])
        for label, allocation in dense_allocations:
            if task not in allocation:
                continue
            pattern = allocation[task]["module_ranks"]
            if len(pattern) != len(spectra):
                raise ValueError(f"{task}: dense pattern/module mismatch")
            rules.append((label, None, pattern))
        rules = rules[args.shard::args.shards]      # strided, so a partial result
        if args.shards > 1:                          # still spans the whole sweep
            print(f"  shard {args.shard}/{args.shards}: "
                  f"{[lab for lab, _, _ in rules]}", flush=True)
        for label, tau, fixed in rules:
            out, kept, resid, k_tot, n_tot = dict(weights), 0.0, 0.0, 0, 0
            for module_index, (a_key, sv) in enumerate(spectra.items()):
                b_key = a_key.replace(".lora_A.", ".lora_B.")
                fixed_rank = (fixed[module_index] if isinstance(fixed, list) else fixed)
                na, nb, k, _ = truncated_factors(weights[a_key], weights[b_key],
                                                 tau or 0.0, fixed_rank=fixed_rank)
                out[a_key], out[b_key] = na, nb
                e = sv.square()
                kept += float(e[:k].sum())
                resid += float(e[k:].sum())
                k_tot += k
                n_tot += sv.numel()
            dest = args.work / f"{task}-{label}"
            dest.mkdir(parents=True, exist_ok=True)
            save_file(out, dest / "adapter_model.safetensors")
            (dest / "adapter_config.json").write_text((local / "adapter_config.json").read_text())
            m_c, preds_c, hits_c, _, spread_c = run(dest, label)
            d_js = divergence(dest, local)
            L = math.sqrt(resid / (kept + resid)) if kept + resid > 0 else 0.0
            rel = (m_orig - m_c) / m_orig if m_orig > 0 else float("nan")
            record["variants"][label] = {
                "L_W": L, "rank_frac": k_tot / max(n_tot, 1), "mean_rank": k_tot / len(spectra),
                "metric": m_c, "abs_drop": m_orig - m_c, "rel_drop": rel, "P": S * L,
                "per_example": hits_c, "d_js": d_js,
                "c2w": sum(1 for a, b in zip(hits_orig, hits_c) if a > b),
                "w2c": sum(1 for a, b in zip(hits_orig, hits_c) if a < b),
                "pred_changed": sum(1 for a, b in zip(preds_orig, preds_c)
                                    if a.strip() != b.strip())}
            if spread_c:
                record["variants"][label].update(spread_c)
            if not args.no_margin_probe:
                tokens, first = margin_probe(dest, local, gen_orig)
                np.savez_compressed(tok_dir / f"{task}-{label}.npz",
                                    **{k: np.asarray(v, dtype=np.float32 if "margin" in k
                                                     or k == "per_token_js" else np.int32)
                                       for k, v in tokens.items()},
                                    first_shared_prefix_flip=np.asarray(first, dtype=np.int32),
                                    per_example_orig=np.asarray(hits_orig, dtype=np.float32),
                                    per_example=np.asarray(hits_c, dtype=np.float32))
                mo = np.asarray(tokens["per_token_orig_margin"])
                mc = np.asarray(tokens["per_token_comp_margin"])
                flip = mc < 0
                record["variants"][label].update(
                    first_shared_prefix_flip=first,
                    n_tokens=int(mo.size),
                    token_flip_rate=float(flip.mean()) if mo.size else 0.0,
                    seq_flip_rate=float(np.mean([f >= 0 for f in first])),
                    median_margin=float(np.median(mo)) if mo.size else 0.0,
                    median_erosion=float(np.median(mo - mc)) if mo.size else 0.0,
                    # M_t >= 0 by construction; a nonzero share here means the probe
                    # is not re-creating the state generation was in, not a result
                    neg_orig_margin_frac=float((mo < 0).mean()) if mo.size else 0.0)
                print(f"       probe: tokens={mo.size} token_flip="
                      f"{record['variants'][label]['token_flip_rate']:.1%} seq_flip="
                      f"{record['variants'][label]['seq_flip_rate']:.1%} "
                      f"median M={np.median(mo):.3f} median E={np.median(mo - mc):.3f} "
                      f"neg_M={record['variants'][label]['neg_orig_margin_frac']:.2%}",
                      flush=True)
            print(f"  {label}: L_W={L:.3f} rank={k_tot/len(spectra):.1f}/{rank} "
                  f"metric={m_c:.4f} rel_drop={rel:+.1%} D_JS={d_js:.6f} "
                  f"c2w={record['variants'][label]['c2w']} w2c={record['variants'][label]['w2c']} "
                  f"changed={record['variants'][label]['pred_changed']}/{len(prompts)}", flush=True)
        peft_model = peft_model_ref[0]
        results.append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {args.output} ({len(results)} adapters)")


if __name__ == "__main__":
    main()
