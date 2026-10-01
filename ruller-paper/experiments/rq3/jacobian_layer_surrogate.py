#!/usr/bin/env python3
"""Downstream-propagated truncation surrogates, matched to the JS oracle.

layer_truncation_probe.py measured what a rank-k truncation of one layer actually
costs, C_l(k) = D_JS(f_full, f_{l->k}), and found that every layer-local
surrogate loses to the plain spectral residual L_W: the incumbent S^g L_W scores
.292, the activation-aware discarded-subspace loss A_l .544, L_W .666. Both
losing surrogates stop at the layer's own output. This measures the two that do
not, by pushing the discarded subspace through the rest of the network:

    Jacobian-L2      || J_l R_l(k) X ||_F^2
    Jacobian-Fisher  sum_t (J_l R_l(k) X)_t^T F(p_t) (J_l R_l(k) X)_t

The Fisher form is the one to believe, because D_JS(p, p+dp) ~ (1/8) dz^T F dz
with F = diag(p) - p p^T, so it is the second-order expansion of the very
quantity the oracle measures. The L2 form is the same machinery under a metric
that ignores the output distribution, and is here to show whether the metric
matters or only the propagation does.

Neither needs an explicit Jacobian. For projection m draw a scalar s_m and take
one backward; the gradient at layer l's output is G_l = ds_m/dH_l.

    L2      s_m = sum_t <r_{t,m}, z_t>,      r Rademacher over the vocabulary
    Fisher  s_m = sum_t xi_{t,m} log p_t(y_t),  y_t ~ p_t, xi Rademacher

For Fisher, grad_z log p(y) = e_y - p and E_y[(e_y-p)(e_y-p)^T] = F(p), so the
estimator is Hutchinson's for the Fisher metric. The Rademacher xi is what makes
the cross-token terms vanish in expectation, so summing the inner product over
tokens before squaring assumes nothing about token independence -- and J_l maps a
perturbation at one position to logits at every later position, so attention
mixing is inside the estimate rather than approximated away.

Nothing per token is stored. With R_l(k) = sum_{j>k} sigma_j u_j v_j^T,

    q_{l,j,m} = sigma_j * sum_t (g_{l,t,m}.u_j)(v_j.x_{l,t})

accumulates online, and a suffix sum over j gives every rank at once:

    Chat_l(k) = (1/(8M)) sum_m ( sum_{j>k} q_{l,j,m} )^2

so the whole calibration costs one forward and M backwards per batch, and keeps
r*M numbers per module.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predibase_task_metrics import TASKS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--examples", type=int, default=16)
    ap.add_argument("--example-start", type=int, default=200)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--projections", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = TASKS[args.task]
    end = args.example_start + args.examples
    rows, last = None, None
    for ds_args, rev in [(spec["ds"], None)] + [((r,), v) for r, v in spec.get("alt", [])]:
        try:
            kw = {"split": f"{spec['split']}[{args.example_start}:{end}]"}
            if rev:
                kw["revision"] = rev
            rows = load_dataset(*ds_args, **kw)
            break
        except Exception as exc:
            last = exc
    if rows is None:
        raise RuntimeError(f"could not load {args.task}: {last}")
    prompts = [spec["prompt"](r) for r in rows]

    args.work.mkdir(parents=True, exist_ok=True)
    local = Path(snapshot_download(f"predibase/{args.task}", local_dir=args.work / args.task))
    cfg = json.loads((local / "adapter_config.json").read_text())
    r_nom = int(cfg["r"])
    scaling = float(cfg["lora_alpha"]) / (math.sqrt(r_nom) if cfg.get("use_rslora") else r_nom)

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).to("cuda").eval()
    model = PeftModel.from_pretrained(base, str(local), adapter_name="full").eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    # autograd needs one node upstream of every module to carry a grad_fn, or the
    # module outputs are leaves and grad() refuses. The embedding weight is the
    # cheapest such node: no parameter gradient is ever used, only activations.
    model.get_input_embeddings().weight.requires_grad_(True)

    mods = [(n, m) for n, m in model.named_modules()
            if hasattr(m, "lora_A") and "full" in m.lora_A]
    mods.sort(key=lambda z: z[0])
    U, V, SG = [], [], []
    for _, m in mods:
        A = m.lora_A["full"].weight.detach().float()
        B = m.lora_B["full"].weight.detach().float()
        qb, rb = torch.linalg.qr(B, mode="reduced")
        qa, ra = torch.linalg.qr(A.T, mode="reduced")
        um, sm, vmh = torch.linalg.svd(rb @ ra.T)
        U.append(qb @ um); V.append(qa @ vmh.T); SG.append(sm * scaling)
    L = len(mods)
    print(f"{args.task}: {L} modules, r={r_nom}, M={args.projections}", flush=True)

    # teacher-forced generated positions, the same protocol the oracle used
    gen = []
    for s in range(0, len(prompts), args.batch):
        enc = tok(prompts[s:s + args.batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_length).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=spec["new"], do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for seq in out[:, enc["input_ids"].shape[1]:]:
            ids = seq.tolist()
            if tok.eos_token_id in ids:
                ids = ids[:ids.index(tok.eos_token_id) + 1]
            gen.append(ids)

    batches = []
    for s in range(0, len(prompts), args.batch):
        chunk = prompts[s:s + args.batch]
        pids = [tok(p, truncation=True, max_length=args.max_length)["input_ids"] for p in chunk]
        gs = gen[s:s + len(chunk)]
        full = [p + g for p, g in zip(pids, gs)]
        w = max(map(len, full))
        ids = torch.full((len(full), w), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(full), w), dtype=torch.long)
        pos = torch.zeros((len(full), w), dtype=torch.bool)
        for row, (seq, g) in enumerate(zip(full, gs)):
            off = w - len(seq)
            ids[row, off:] = torch.tensor(seq)
            att[row, off:] = 1
            first = off + len(seq) - len(g) - 1
            pos[row, first:first + len(g)] = True
        batches.append((ids.to("cuda"), att.to("cuda"), pos.to("cuda")))

    M = args.projections
    q_fis = torch.zeros(L, r_nom, M, dtype=torch.float64)
    q_l2 = torch.zeros(L, r_nom, M, dtype=torch.float64)
    gen_ = torch.Generator(device="cuda").manual_seed(args.seed)

    saved: list = [None] * L
    handles = []
    for i, (_, m) in enumerate(mods):
        def hook(mod, inputs, output, idx=i):
            y = output[0] if isinstance(output, tuple) else output
            saved[idx] = (inputs[0], y)
        handles.append(m.register_forward_hook(hook))

    for bi, (ids, att, pos) in enumerate(batches):
        with torch.enable_grad():
            z = model(input_ids=ids, attention_mask=att).logits
            outs = [saved[i][1] for i in range(L)]
            xs = [saved[i][0].detach().float() for i in range(L)]
            logp = torch.log_softmax(z.float(), dim=-1)
            p = logp.exp()
            for m_i in range(M):
                # Fisher: sample a token from the model's own distribution
                flat = p[pos].detach()
                y = torch.multinomial(flat, 1, generator=gen_).squeeze(-1)
                xi = (torch.randint(0, 2, (flat.shape[0],), generator=gen_,
                                    device=flat.device).float() * 2 - 1)
                s_f = (xi * logp[pos].gather(-1, y.unsqueeze(-1)).squeeze(-1)).sum()
                g_f = torch.autograd.grad(s_f, outs, retain_graph=True)
                # L2: Rademacher over the vocabulary
                r = (torch.randint(0, 2, flat.shape, generator=gen_,
                                   device=flat.device).float() * 2 - 1)
                s_l = (r * z[pos].float()).sum()
                g_l = torch.autograd.grad(s_l, outs, retain_graph=(m_i < M - 1))
                for i in range(L):
                    x = xs[i][pos]                                  # (T, d_in)
                    c = x @ V[i]                                    # (T, r)
                    for tgt, gg in ((q_fis, g_f), (q_l2, g_l)):
                        a = gg[i].detach().float()[pos] @ U[i]      # (T, r)
                        tgt[i, :, m_i] += ((a * c).sum(0) * SG[i]).double().cpu()
        del outs, xs, z, logp, p
        torch.cuda.empty_cache()
        print(f"  batch {bi + 1}/{len(batches)}", flush=True)
    for h in handles:
        h.remove()

    def curves(q):
        # suffix sum over singular directions, then square and average over m
        out = []
        for i in range(L):
            row = []
            for k in range(1, r_nom):
                Q = q[i, k:, :].sum(0)
                row.append(float((Q ** 2).mean()) / 8.0)
            out.append(row)
        return out

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "adapter": args.task, "modules": L, "r": r_nom, "projections": M,
        "examples": len(prompts), "ranks": list(range(1, r_nom)),
        "q_fisher": q_fis.tolist(), "q_l2": q_l2.tolist(),
        "jacobian_fisher": curves(q_fis), "jacobian_l2": curves(q_l2),
    }, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
