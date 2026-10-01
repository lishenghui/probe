# Figure 2 reproduction bundle

Figure 2 of *LoRA on a Budget: Function-Aware Post-Hoc Compression for Adapter
Fleets*. Everything needed to regenerate it from measured data is here; nothing
is fetched and no GPU is used.

## What the figure claims

**(a) The additive surrogate ranks fixed-budget candidates well.** Within-budget
Spearman fidelity averages **0.638 / 0.629 / 0.852** on LoRA Land /
Lots-of-LoRAs / LoRARetriever. LoRARetriever is the *most* rank-faithful, so
"the surrogate cannot rank" does not explain why functional search loses there.

**(b) The calibration shifts across proposal shapes.** Plotting each adapter's
surrogate ratio against its measured ratio, the measured ratio favours spectral
for **1/12, 9/25, 39/41** adapters. Points below y=1 are spectral-favouring; the
LoRARetriever cloud sits mostly there while its surrogate predicts otherwise.
That reversal is why FRA keeps a spectral safeguard instead of trusting the
surrogate as a final risk estimate.

## Run

```bash
python3 plot_frontier_comparison.py \
  --results artifacts/rq3/results \
  --output out/frontier_comparison.pdf
```

Needs `numpy` and `matplotlib` only. Takes about a minute; it reads 234 JSON
files and recomputes every point from the measurements.

On success it prints exactly:

```
fixed-budget means {'LoRA Land': 0.6380116959064327, 'Lots-of-LoRAs': 0.6285914786967419, 'LoRARetriever': 0.8515801699370378}
adapter spectral-favoring {'LoRA Land': 1, 'Lots-of-LoRAs': 9, 'LoRARetriever': 39}
```

and writes `frontier_comparison.{pdf,png,csv}` beside the `--output` path.

## Checking your output

`expected/` holds the versions used in the paper.

- `frontier_comparison.csv` and `frontier_comparison.png` come out **byte-identical**
  (verified: same md5 as the published files).
- `frontier_comparison.pdf` will **not** match byte-for-byte. PDFs embed a
  creation timestamp, so this is expected; the identical PNG shows the rendering
  itself is deterministic.

```bash
cmp out/frontier_comparison.csv expected/frontier_comparison.csv && echo CSV OK
cmp out/frontier_comparison.png expected/frontier_comparison.png && echo PNG OK
```

## The CSV

`frontier_comparison.csv` is written by the script, 4,393 rows plus a header:

| column | meaning |
|---|---|
| `fleet` | LoRA Land / Lots-of-LoRAs / LoRARetriever |
| `adapter` | adapter name within the fleet |
| `pred_ratio` | surrogate ratio, spectral over functional |
| `measured_ratio` | end-to-end measured ratio, same orientation |
| `cost_ratio` | parameter cost of the two proposals, matched to within 2% |

A row is one spectral-versus-functional comparison at matched parameter cost.
`measured_ratio > 1` means functional won; `< 1` means spectral won. Panel (b)
plots the per-adapter median of these.

## Inputs (234 files, `artifacts/rq3/results/`)

| pattern | n | what it holds |
|---|---|---|
| `fixedrho_{land,cts,lorare}_*.json` | 12/25/41 | fixed-budget ranking-fidelity samples, panel (a) |
| `functional_dp0_output_{land12,cts25,lorare}_*.json` | 12/25/41 | functional (FuncDP) proposals and measured risks |
| `rank0_spectral512_land12_*.json`, `rank0_spectral_output_cts25_*.json`, `rank0_spectral_lorare_*.json` | 12/25/41 | spectral proposals and measured risks |

`cts` is the internal name for Lots-of-LoRAs, `lorare` for LoRARetriever.

Note the LoRA Land spectral file is `rank0_spectral512_*`, not `rank0_spectral_*`:
an earlier LoRA Land sweep was capped at rank 256, which pinned two adapters at
the cap. The 512 rerun is the correct one. Other artefact families in the full
repository reuse the same variant labels, so match on the exact patterns above
rather than on file names that merely look right.

---

# Proposed panel (c): where the calibration error lives

Panels (a) and (b) say the surrogate ranks well inside a family but mis-ranks the
families against each other, and the paper stops there -- "consistent with
cross-layer cancellation, but does not identify it". Panel (c) does not identify
the mechanism either, but it localises the error one level further, and it rests
on an identity rather than a fitted hypothesis:

```
log[ (Rt_S/Rt_F) / (R_S/R_F) ]  =  log(Rt_S/R_S) - log(Rt_F/R_F)
```

The cross-family error plotted in (b) *is* the difference between each family's
own surrogate bias. Panel (c) plots those two biases separately.

## What it shows

| fleet | n | functional bias | spectral bias | gap | measured spectral wins |
|---|---|---|---|---|---|
| LoRA Land | 585 | x2.81 (IQR 1.26) | x4.76 (IQR 2.37) | **x1.70** | 1/12 |
| Lots-of-LoRAs | 2038 | x2.15 (IQR 1.38) | x6.27 (IQR 2.26) | **x2.92** | 9/25 |
| LoRARetriever | 1770 | x3.58 (IQR 1.41) | **x17.67** (IQR 2.19) | **x4.94** | 39/41 |

IQRs are in log units. Read together with (a) and (b):

- The surrogate **overstates** risk for both families everywhere (all biases > 1),
  which is the sub-additive tendency of summing isolated per-layer costs.
- Inside a family the bias behaves like a common multiplicative offset, and a
  common factor leaves Spearman untouched -- so (a) can be high while (b) fails.
- Across families the two offsets differ, and their gap is exactly the error in
  (b). The gap orders the fleets 1.70 < 2.92 < 4.94, the same order as how often
  spectral actually wins (1/12 < 9/25 < 39/41). The spectral bias is the single
  quantity that tracks failure severity.

This is why FRA keeps a spectral safeguard: the surrogate is usable for search
inside one family and not usable as a final cross-family risk estimate.

## What it does not show

The within-family IQR is 1.26-2.37 in log units, so the bias is **not** a clean
constant offset -- the claim is about central tendency, not about every
candidate. The spectral bias is also consistently more dispersed (IQR ~2.2-2.4)
than the functional one (~1.3-1.4): it is both larger and less stable. And
nothing here identifies *why* the spectral family is biased more; that remains
open exactly as the paper states.

## A hypothesis that did **not** survive

Before settling on the decomposition I tested the more obvious panel: does the
calibration error grow with how differently the two proposals distribute rank
(total-variation distance between normalised rank profiles)? Spearman over the
same 4,393 comparisons:

```
LoRA Land      -0.260
Lots-of-LoRAs  -0.487
LoRARetriever  +0.118
```

Inconsistent in sign, mostly negative -- the opposite of the hypothesis -- and
absent on the fleet that matters. That panel would have been noise. It is
recorded here so the negative result is not re-run by accident.

## Run

```bash
python3 plot_calibration_decomposition.py \
  --results artifacts/rq3/results --output out/panel_c.pdf     # panel (c) alone

python3 make_figure2_3panel.py \
  --results artifacts/rq3/results --output out/figure2_3panel.pdf   # a|b|c together
```

`make_figure2_3panel.py` is deliberately a separate file: it redraws (a) and (b)
rather than refactoring `plot_frontier_comparison.py`, so the byte-identical
reproduction of the published figure stays intact.

Reference outputs are in `expected_new/`.

## `panel_c.csv`

4,393 rows, one per matched-cost comparison, using the identical acceptance rule
as panel (b) (costs within 2%, distinct allocations, all four quantities > 0):

| column | meaning |
|---|---|
| `functional_bias` | `log(surrogate / measured)` for the functional allocation |
| `spectral_bias` | same for the matched spectral allocation |
| `cross_family_error` | `spectral_bias - functional_bias`, the panel-(b) error |

**Two different summaries, do not mix them.** The arrows in panel (c) show the
*difference of the medians* (x1.70 / x2.92 / x4.94), which is what the eye reads
off the plot. The *median of the per-comparison differences* is slightly
different (x1.47 / x2.86 / x3.91); that is the number to quote if you want a
per-comparison statistic. Both are computable from this CSV.
