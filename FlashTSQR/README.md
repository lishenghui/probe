# FlashTSQR

A batched **tall-skinny QR** operator for GPUs, motivated by the matrix shapes that
LoRA / PEFT aggregation induces: **many** matrices, **very tall**, **very few columns**
(`[M, d, Σr]` with `d` up to 8192 and `Σr` ≤ a few hundred).

**The finding:** cuSOLVER's `geqrf` runs at **0.06–0.5 % of peak** on these shapes.
A shared-memory TSQR (leaf Householder QR + binary tree reduction) is **6–18× faster
at identical numerical quality** — it is the *same* algorithm on a different
execution schedule, not a Gram/Cholesky shortcut.

---

## Verified results (NVIDIA GH200, fp32, CUDA events, no fast-math)

### 1. The gap — cuSOLVER on LoRA shapes (`bench/diag_qr_roofline.py`)

| shape | `geqrf` | CholeskyQR (same flops) | peak fp32 GEMM |
|---|---|---|---|
| `[56, 8192, 64]` | **73.7 GF/s** | 3,785 GF/s | 47,835 GF/s |
| `[56, 3072, 64]` | **50.2 GF/s** | 2,628 GF/s | |
| `[56, 8192, 232]` | **258.6 GF/s** | 12,050 GF/s | |

QR and CholeskyQR have comparable flop counts (~2mn² vs ~3mn²), so the 20–50× gap is
**implementation**, not algorithm or hardware.

### 2. It is not a dispatch artefact (`bench/diag_qr_paths.py`)

On `[56, 8192, 64]`, every alternative path lands within 1.0–1.14× of `torch.linalg.qr`:
per-matrix loop (1.01×), chunked batches (1.0×), `geqrf` alone (1.14×),
`geqrf`+`ormqr` implicit-Q (1.03×). `orgqr` (forming Q) is only **6–14 %** of `qr`;
the cost is `geqrf` itself.

### 3. FlashTSQR — R only (`bench/bench_falsify.py`)

Correctness by the defining property **RᵀR = BᵀB** (not just |R| matching):

| shape | `geqrf` | FlashTSQR | speed-up | our residual | cuSOLVER residual |
|---|---|---|---|---|---|
| `[56, 8192, 64]` | 45.02 ms | **4.32 ms** | **10.4×** | 1.25e-06 | 1.23e-06 |
| `[56, 3072, 64]` | 24.25 ms | **2.20 ms** | **11.0×** | 8.50e-07 | 8.41e-07 |
| `[56, 1024, 64]` | 13.70 ms | **1.22 ms** | **11.2×** | 4.76e-07 | 4.55e-07 |
| `[56, 8192, 32]` | 22.72 ms | **1.31 ms** | **17.3×** | 1.25e-06 | 1.25e-06 |
| `[112, 2048, 64]` | 32.54 ms | **2.29 ms** | **14.2×** | 6.20e-07 | 6.13e-07 |
| `[28, 4096, 128]` | 30.32 ms | 10.08 ms | 3.0× | 8.84e-07 | 8.69e-07 |
| `[8, 16384, 64]` | 8.51 ms | 2.29 ms | 3.7× | 1.88e-06 | 1.88e-06 |

**Residuals match cuSOLVER digit for digit** — same numerical quality.

### 4. Full operator — R + reflectors + Q@v (`bench/bench_full.py`)

The fair comparison: what FraQ actually needs is `geqrf` + `ormqr`.

| shape | `geqrf+ormqr` | FlashTSQR full | speed-up | RᵀR−BᵀB | Q@v error |
|---|---|---|---|---|---|
| `[56, 8192, 64]` | 50.20 ms | **8.14 ms** | **6.2×** | 1.25e-06 | 9.2e-07 |
| `[56, 3072, 64]` | 27.45 ms | **4.22 ms** | **6.5×** | 8.50e-07 | 8.5e-07 |
| `[56, 1024, 64]` | 15.97 ms | **2.39 ms** | **6.7×** | 4.76e-07 | 7.7e-07 |
| `[36, 768, 64]` | 31.51 ms | **1.69 ms** | **18.6×** | 4.67e-07 | 7.5e-07 |

### 5. Where the speed-up comes from — batch parallelism, not a better single QR
(`bench/diag_batch_vs_schedule.py`, fixed `[d=3072, N=64]`, sweeping M)

| M | cuSOLVER per-matrix | ours per-matrix | speed-up |
|---|---|---|---|
| 1 | 0.4923 ms | 1.1889 ms | **0.41×** (we lose!) |
| 4 | 0.4462 ms | 0.2980 ms | 1.50× |
| 16 | 0.4293 ms | 0.0810 ms | 5.30× |
| 112 | 0.4322 ms | **0.0319 ms** | **13.54×** |

cuSOLVER's per-matrix cost is **constant in M** — its "batched" entry point runs the
batch serially. Ours falls 37×. The crossover is M≈4; at M=1 the single-matrix
libraries (cuQR, tsqr-tc, Thies & Röhrig-Zöllner Euro-Par'26) are the right tool and
we are 2.4× worse. Our regime is **many matrices at once** — which is what a LoRA
model is (36–200 modules).

### 6. Cost vs the applied rank k
(`bench/diag_rank_apply.py`; k = the retained rank after FraQ's τ/cap selection —
the τ truncation happens at the eigh, **after** the QR, so the factorisation always
runs at full N=Σr regardless of τ; only the Q@v width varies)

| | k=8 | k=16 | k=32 | k=N (keep all) |
|---|---|---|---|---|
| `[56,8192,64]` vs `geqrf+ormqr` | 6.04× | 5.78× | 5.32× | 3.06× |
| `[28,4096,128]` (dense merges) | 2.13× | 2.03× | 1.56× | **1.05×** |

The apply is **pay-for-what-you-keep**: cost grows smoothly with k and never loses
to the explicit orgqr+GEMM route. Libraries force a Q-less / full-Q choice up front;
FraQ only knows k after the eigh, so deferring that decision is exactly what the
workload needs. The weak corner was large N, not large k — fixed below.

### 7. TT-structured, packed merges (the default kernel)
(`bench/bench_tt.py`) Two structural facts about a merge node, both exploited:

* **compute** — its input is two stacked **upper triangles** [R1; R2], so
  Householder column j has only j+2 active rows: merge flops drop from (10/3)N³
  to ~N³/3, and the merge tree is the dominant factor cost at large N (62 % of
  the flops at `[28,4096,128]`, P=32);
* **storage** — the two triangles live **packed** in shared memory (~N² floats
  instead of the dense 2N²): at N=128 the merge tile drops 131 KB → 67 KB, i.e.
  1 → 3 blocks/SM. Inter-level R blocks and merge reflectors travel packed
  through global memory too (~2× less traffic), and the Q@v application is
  chunked over columns (KC=64) so large k no longer blows the smem budget.

Full operator (factor + Q@v), dense merges → packed-TT merges:

| shape | k | dense | **packed TT** | vs dense | vs cuSOLVER |
|---|---|---|---|---|---|
| `[56,8192,64]` | 8 | 8.23 ms | **5.92 ms** | 1.39× | **8.4×** |
| | 64 | 16.38 | **8.21** | 2.00× | **6.1×** |
| `[56,3072,64]` | 8 | 4.28 | **2.55** | 1.68× | **10.7×** |
| | 64 | 6.80 | **3.52** | 1.93× | **7.8×** |
| `[28,4096,128]` | 8 | 15.92 | **6.90** | 2.31× | **4.9×** |
| | 128 | 31.05 | **10.89** | **2.85×** | **1.05× → 3.01×** |

Factor-only: `[56,8192,64]` 4.98 → 3.11 ms (14.4× over cuSOLVER); `[28,4096,128]`
11.67 → **3.92 ms** (2.98× over dense, **7.8×** over cuSOLVER). Numerics are
untouched (same Householder math, structural zeros skipped): RᵀR−BᵀB residuals
identical to the dense kernel and to cuSOLVER; Q@v at machine precision for
every k. History: dense → TT (compute only) → packed TT; the intermediate
TT-only numbers are in the log of commit f21fd5e.

### 8. Arbitrary leaf counts — the heterogeneous Σr=232 shape, unlocked
(`bench/bench_bigN.py`) The tree now pairs nodes level-by-level with **bye-passing**
(odd node carries through) and the last leaf is **zero-padded**, removing the
`m % rows/leaf == 0` and power-of-two constraints. Zero rows provably do not change
R, and their reflector entries are exactly zero, so Q@v on the valid rows is
untouched — confirmed at machine precision. With packed merges fitting N=232
(218 KB) and the leaf tile at rows/leaf=232 (217 KB), the real heterogeneous
FraQ shape runs for the first time:

| shape (P) | | cuSOLVER | ours | speed-up |
|---|---|---|---|---|
| `[56,8192,232]` (P=36, byes) | factor | 177.0 ms | **49.7 ms** | **3.6×** |
| | full k=8 | 191.2 | **58.6** | 3.3× |
| | full k=105 (τ=0.95 eff. rank) | 187.4 | **68.2** | **2.8×** |
| | full k=232 | 189.1 | **81.4** | **2.3×** |
| `[56,3072,232]` (P=14, byes) | factor | 92.9 | **19.9** | **4.7×** |
| | full k=105 | 100.8 | **28.0** | **3.6×** |

Power-of-two shapes regress-tested through the generalised tree: unchanged
(`[56,8192,64]` factor 13.9×, `[28,4096,128]` factor 7.8×). RᵀR−BᵀB at 1e-06
level throughout.

### 9. WY-blocked apply (compact-WY on batched GEMMs)
For N ≥ 160 the Q@v application dispatches to a **blocked WY** path:
`Q = Π(I − VₚTₚVₚᵀ)` per panel of 32 reflectors, executed as three cuBLAS
batched GEMMs per panel. T factors are built lazily on first apply via the
closed form `Tₚ = (diag(1/τ)+triu(VₚᵀVₚ,1))⁻¹` — one batched triangular solve
per panel, no column loop. At `[56,8192,232]` this cuts the apply from
67 → **19 ms at k=105** (3.6×) and 131 → **32 ms at k=232** (4.1×), at identical
fp32 accuracy. Below N=160 the fused column-wise kernels stay faster (launch
cost dominates GEMM shapes that small), so the dispatch keeps both paths.

**TF32/TensorCore, measured and rejected:** enabling TF32 on the WY GEMMs gains
only ~2 % (the apply is copy/bandwidth-bound after blocking) while degrading the
Q@v error from 1.6e-06 to **2.4e-03**. Not worth it; the certified default
remains plain fp32. Merge reflectors are stored dense again to be GEMM-ready
(packed R blocks and the packed shared-memory merge tiles are unchanged).

The remaining large-N bottleneck is the **factor** (still column-serial rank-1
updates in the leaf/merge tiles): WY-ifying the factorisation itself is the
open roadmap item.

### Where it wins, and where it does not

Speed-up grows as **N shrinks, the batch grows, and k shrinks** — the LoRA regime.
Measured boundaries of the operating region:
* `M=1` → 0.41× (use cuSOLVER/cuQR below M≈4);
* `N=128 ∧ k=N` → 3.01× with packed-TT merges (was parity with dense merges);
* the shared-memory wall: packed merges fit to N≈238 and the leaf tile
  (`rows/leaf × N`, with rows/leaf ≥ N) caps N at ≈235 — the divisibility rules are
  gone (section 8), so Σr=232 is **supported**; beyond N≈235 needs streamed panels;
* at N=232 the win narrows (3.6× factor, 1.05× at k=N): the column-serial rank-1
  updates are the remaining bottleneck — the WY/TensorCore roadmap item.

---

## Falsification log

Everything below was tested and did **not** overturn the result:

| attack | outcome |
|---|---|
| warm-up / synchronisation | 10 warm-ups; CUDA-event timing; same numbers |
| allocation tax | `geqrf` fresh-alloc 44.66 ms vs repeated 44.66 ms — identical; and *we allocate too* (a buffer per tree level) |
| batched vs batched | `torch.geqrf` **is** the batched call; loop vs batched = 1.01× |
| pivoting | `geqrf` is unpivoted Householder (`geqp3` would pivot); ours too |
| Tensor Cores / dtype | both plain fp32, `allow_tf32=False`, no TC in our kernel |
| `--use_fast_math` | removed — still 10.4× |
| lucky launch config | 9 combinations of `tpb` × `rows/leaf` all win (6.1–10.9×) |
| **R only, no Q** (our own objection) | closed: full operator with reflectors + Q@v still **6.2×** |

**Known caveat:** the reported GF/s uses the *plain-QR* flop count. TSQR's tree
reduction is extra work (~2× the flops at `P=64` leaves), so the achieved throughput
is higher than the "useful flops" number suggests — and `P` is not yet tuned.

---

## Layout

```
kernels/tsqr_full.cu         DEFAULT: full operator (R + reflectors + Q@v), TT-structured merges
kernels/tsqr_dense_merge.cu  ablation baseline: same operator with dense merges
kernels/tsqr_r.cu            R-only prototype (dense merges)
bench/diag_qr_roofline.py    cuSOLVER efficiency vs CholeskyQR vs GEMM (the motivation)
bench/diag_qr_paths.py       rules out PyTorch dispatch as the cause
bench/diag_batch_vs_schedule.py  M-sweep: batch parallelism vs schedule (section 5)
bench/diag_rank_apply.py     cost vs applied rank k (section 6)
bench/bench_proto.py         first prototype, R only
bench/bench_falsify.py       adversarial validation (RᵀR=BᵀB, no fast-math, sweeps)
bench/bench_full.py          full operator vs geqrf+ormqr
bench/bench_tt.py            TT merges vs dense merges vs cuSOLVER (section 7)
bench/bench_bigN.py          arbitrary-P trees + padding; hetero Σr=232 (section 8)
slurm/*.sh                   job scripts (they build the toolchain on first run)
```

## Building / running

The cluster has no system CUDA toolkit, so the slurm scripts create a **separate**
conda prefix holding `nvcc` matching the torch build (cu126) — it does not touch the
project env:

```
$W/nvcc_env      # cuda-nvcc 12.6 + cuda-cudart-dev 12.6
CUDA_HOME=$W/nvcc_env
TORCH_EXTENSIONS_DIR=$W/.torch_ext
TORCH_CUDA_ARCH_LIST=9.0        # GH200 / Hopper
```

Then simply:

```bash
sbatch slurm/run_roofline.sh    # the gap
sbatch slurm/run_qr_paths.sh    # rule out dispatch
sbatch slurm/run_falsify.sh     # adversarial validation, R only
sbatch slurm/run_full.sh        # full operator vs geqrf+ormqr
```

Logs land in `/nobackup/.../logs/`.

### Hugging Face LoRA spectral census

The Top-100 rank-32/64 census uses the FlashTSQR R factors to compute each LoRA
update spectrum from an `r x r` core:

```bash
sbatch slurm/run_hf_lora_census.sh
```

Candidate discovery, popularity ordering, SafeTensors download rules, tensor
pair validation, exact rank filtering, cache placement, output schema, and the
important “100 repositories versus 100 adapters” distinction are documented in
[`docs/hf_lora_spectral_census.md`](../docs/hf_lora_spectral_census.md).

## Next

1. ~~Packed-triangle merge storage~~ — done (section 7).
2. **WY / blocked panels — factor side**: the apply is done (section 9); the
   factorisation's trailing updates are still rank-1. Panelising the factor
   (panel QR in smem + GEMM trailing update) is where the next 2–3× at N≥160
   lives. Plain TF32 is ruled out (section 9); fp16+error-correction à la
   tsqr-tc remains the only TC route worth trying.
3. ~~Unlock Σr=232~~ — done via bye-passing trees + zero-padded tail leaf
   (section 8). **Streamed tall leaves** remain useful for shallower trees and
   N beyond ≈235.
4. **Dispatcher**: route M<4 to cuSOLVER (kills the single-matrix weakness by design).
5. **cuQR + CUDA streams comparison** at M=112 — the one experiment that could dent
   the batched-regime claim; must be measured, not argued.
6. Wire into FraQ end-to-end (QR ≈ 88 % of batched FraQ → expect ~3–4×), and
   reproduce on a second GPU (A100).
