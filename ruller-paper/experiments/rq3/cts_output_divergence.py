#!/usr/bin/env python3
"""Output-position divergence for the primary Lots-of-LoRAs pool.

The primary law is fitted against a divergence averaged over prompt prefixes.
On LoRA Land that quantity turned out to sit between 0.26x and 52x from the
divergence at the tokens the model actually emits, and to manufacture at least
one apparent discontinuity that the output-side measurement does not have.  The
headline fit should not rest on a metric the paper itself shows can be that
distorted, so this recomputes the same sweep against output positions.

For each adapter and threshold we take the uncompressed adapter's greedy
continuation and score *both* adapters at x, y_<t -- shared original prefix, so a
disagreement at t is a decision at t and not error propagated from t-1.  The
prompt-position divergence is recomputed in the same pass at the same precision,
so the two are compared on identical inputs rather than across runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compress_adapter import truncated_factors  # noqa: E402
from sweep_cts_compression import js_divergence  # noqa: E402


def spectrum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Singular values of B @ A via thin QR, without forming Delta W."""
    a32, b32 = a.float(), b.float()
    qb, rb = torch.linalg.qr(b32, mode="reduced")
    qa, ra = torch.linalg.qr(a32.T, mode="reduced")
    return torch.linalg.svdvals(rb @ ra.T)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--strengths", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.99, 0.95, 0.90])
    ap.add_argument("--prompts", type=int, default=48)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--truncation-side", choices=("left", "right"), default="left",
                    help="SuperNI prompts end with the task instance, so right "
                         "truncation removes it and collapses every prompt to the "
                         "shared preamble. At max_length=320 with right truncation "
                         "20 of these 30 adapters are left with a single distinct "
                         "prompt, which silently destroys the effective sample size "
                         "without changing the shape of any curve.")
    ap.add_argument("--new", type=int, default=32, help="greedy decode budget per prompt")
    ap.add_argument("--gen-batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="fp32")
    ap.add_argument("--split", default="test",
                    help="task metrics need held-out data, and the mechanism analysis "
                         "should sit on the same inputs; falls back to train")
    ap.add_argument("--token-output", type=Path, default=None,
                    help="directory for the per-token margin probes")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    tok_dir = args.token_output or args.output.with_suffix("").with_name(
        args.output.stem + "_tokens")
    tok_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
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
    entries = entries[args.shard::args.shards]
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"
    tok.truncation_side = args.truncation_side
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dt = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dt).to("cuda").eval()
    print(f"base loaded; sweeping {len(entries)} adapters", flush=True)

    peft_model, results = None, []
    for entry in entries:
        task = int(entry["adapter"].replace("task", ""))
        dataset_id = ds_by_task.get(task)
        if dataset_id is None:
            print(f"skip task{task:04d}: no dataset", flush=True)
            continue
        try:
            local = Path(snapshot_download(
                f"Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task{task}",
                local_dir=args.work / f"task{task:04d}"))
            avail = load_dataset(dataset_id)
            split = args.split if args.split in avail else "train"
            rows = avail[split]
        except Exception as exc:
            print(f"skip task{task:04d}: {type(exc).__name__} {exc}"[:160], flush=True)
            continue
        field = next((c for c in ("input", "text", "prompt") if c in rows.column_names), None)
        texts = [r for r in rows[field][: args.prompts * 3] if r and r.strip()][: args.prompts]
        if len(texts) < 8:
            print(f"skip task{task:04d}: only {len(texts)} prompts", flush=True)
            continue

        weights = load_file(local / "adapter_model.safetensors")
        a_keys = sorted(k for k in weights if ".lora_A." in k)
        # every module here shares one rank and scaling, so the constant scale
        # cancels in L_W and the raw singular values suffice
        energies = {k: spectrum(weights[k], weights[k.replace(".lora_A.", ".lora_B.")]).square()
                    for k in a_keys}
        variants = {}
        for threshold in args.thresholds:
            label = f"e{round(threshold * 100):02d}"
            out, kept, nominal = dict(weights), 0, 0
            e_keep = e_drop = 0.0
            for a_key in a_keys:
                b_key = a_key.replace(".lora_A.", ".lora_B.")
                na, nb, k, _ = truncated_factors(weights[a_key], weights[b_key], threshold)
                out[a_key], out[b_key] = na, nb
                kept += k
                nominal += weights[a_key].shape[0]
                e = energies[a_key]
                e_keep += float(e[:k].sum())
                e_drop += float(e[k:].sum())
            dest = args.work / f"task{task:04d}-{label}"
            dest.mkdir(parents=True, exist_ok=True)
            save_file(out, dest / "adapter_model.safetensors")
            (dest / "adapter_config.json").write_text(
                (local / "adapter_config.json").read_text())
            variants[label] = dict(dir=dest, rank_frac=kept / max(nominal, 1),
                                   L_W=(e_drop / (e_keep + e_drop)) ** 0.5
                                   if e_keep + e_drop > 0 else 0.0)

        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, str(local), adapter_name="orig")
        else:
            peft_model.load_adapter(str(local), adapter_name="orig")
        m = peft_model
        m.eval()

        # greedy continuations from the uncompressed adapter; these define the
        # positions every variant is then scored at
        m.set_adapter("orig")
        gens = []
        for i in range(0, len(texts), args.gen_batch):
            enc = tok(texts[i:i + args.gen_batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_length).to("cuda")
            with torch.no_grad():
                got = m.generate(**enc, max_new_tokens=args.new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            for seq in got[:, enc["input_ids"].shape[1]:]:
                ids = seq.tolist()
                if tok.eos_token_id in ids:
                    ids = ids[:ids.index(tok.eos_token_id) + 1]
                gens.append(ids)
        print(f"task{task:04d}: {len(texts)} prompts, mean {sum(map(len, gens)) / len(gens):.1f} "
              f"generated tokens", flush=True)

        record = {"adapter": entry["adapter"], "S": entry["mean"], "task": task,
                  "dataset": dataset_id, "split": split, "prompts": len(texts),
                  "gen_tokens": sum(map(len, gens)), "variants": {}}
        for label, var in variants.items():
            m.load_adapter(str(var["dir"]), adapter_name=label)
            p_tot, p_seen = 0.0, 0
            o_tot, o_seen = 0.0, 0
            flips = 0
            ex, st, m_o, m_c, jsv, first = [], [], [], [], [], []
            for i in range(0, len(texts), args.chunk):
                chunk, gs = texts[i:i + args.chunk], gens[i:i + args.chunk]
                enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                          max_length=args.max_length)
                keep = enc["attention_mask"].bool()

                # prompt positions, as the original sweep measures them
                e = {k: v.to("cuda") for k, v in enc.items()}
                with torch.no_grad():
                    m.set_adapter("orig")
                    a = m(**e).logits
                    m.set_adapter(label)
                    b = m(**e).logits
                msk = e["attention_mask"].bool()
                p_tot += float(js_divergence(a, b)[msk].sum())
                p_seen += int(msk.sum())
                del a, b

                # output positions, both adapters on the original continuation
                full = [enc["input_ids"][j][keep[j]].tolist() + list(gs[j])
                        for j in range(len(chunk))]
                T = max(len(f) for f in full)
                inp = torch.full((len(full), T), tok.pad_token_id, dtype=torch.long)
                att = torch.zeros((len(full), T), dtype=torch.long)
                for j, f in enumerate(full):
                    inp[j, T - len(f):] = torch.tensor(f)
                    att[j, T - len(f):] = 1
                inp, att = inp.to("cuda"), att.to("cuda")
                with torch.no_grad():
                    m.set_adapter("orig")
                    la = m(input_ids=inp, attention_mask=att).logits
                    m.set_adapter(label)
                    lb = m(input_ids=inp, attention_mask=att).logits
                for j, g in enumerate(gs):
                    if not g:
                        first.append(-1)
                        continue
                    start = T - len(g)
                    pos = torch.arange(start - 1, start - 1 + len(g), device=inp.device)
                    za, zb = la[j, pos].float(), lb[j, pos].float()
                    jj = js_divergence(za, zb)
                    o_tot += float(jj.sum())
                    o_seen += len(g)
                    flips += int((za.argmax(-1) != zb.argmax(-1)).sum())

                    # M_t and its compressed counterpart, both at the same state
                    y = torch.tensor(g, device=inp.device)

                    def margin(z):
                        top2 = z.topk(2, dim=-1)
                        rival = torch.where(top2.indices[:, 0] == y,
                                            top2.values[:, 1], top2.values[:, 0])
                        return z.gather(1, y[:, None]).squeeze(1) - rival

                    ma, mb = margin(za), margin(zb)
                    ex.extend([i + j] * len(g))
                    st.extend(range(len(g)))
                    m_o.extend(ma.tolist())
                    m_c.extend(mb.tolist())
                    jsv.extend(jj.tolist())
                    fl = (mb < 0).nonzero()
                    first.append(int(fl[0]) if fl.numel() else -1)
                del la, lb
            m.delete_adapter(label)
            torch.cuda.empty_cache()
            mo = np.asarray(m_o, dtype=np.float32)
            mc = np.asarray(m_c, dtype=np.float32)
            np.savez_compressed(
                tok_dir / f"{entry['adapter']}-{label}.npz",
                example=np.asarray(ex, dtype=np.int32), step=np.asarray(st, dtype=np.int32),
                per_token_orig_margin=mo, per_token_comp_margin=mc,
                per_token_js=np.asarray(jsv, dtype=np.float32),
                first_shared_prefix_flip=np.asarray(first, dtype=np.int32))
            record["variants"][label] = {
                "rank_frac": var["rank_frac"], "L_W": var["L_W"],
                "d_prompt": p_tot / max(p_seen, 1),
                "d_out": o_tot / max(o_seen, 1),
                "flip_out": flips / max(o_seen, 1),
                "prompt_positions": p_seen, "out_positions": o_seen,
                "token_flip_rate": float((mc < 0).mean()) if mo.size else 0.0,
                "seq_flip_rate": float(np.mean([f >= 0 for f in first])) if first else 0.0,
                "median_margin": float(np.median(mo)) if mo.size else 0.0,
                "median_erosion": float(np.median(mo - mc)) if mo.size else 0.0,
                # M_t >= 0 by construction; nonzero here means the probe is not
                # re-creating the state generation was in, not a result
                "neg_orig_margin_frac": float((mo < 0).mean()) if mo.size else 0.0}
            v = record["variants"][label]
            print(f"  {label}: d_prompt={v['d_prompt']:.3e} d_out={v['d_out']:.3e} "
                  f"ratio={v['d_prompt'] / max(v['d_out'], 1e-15):6.2f} "
                  f"flip={v['flip_out']:.2%} seqflip={v['seq_flip_rate']:.1%} "
                  f"M={v['median_margin']:.2f} E={v['median_erosion']:.3f} "
                  f"neg_M={v['neg_orig_margin_frac']:.2%}", flush=True)
        m.delete_adapter("orig")
        torch.cuda.empty_cache()
        results.append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {args.output} ({len(results)} adapters)")


if __name__ == "__main__":
    main()
