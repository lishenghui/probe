# Same-base video LoRA fleets on Hugging Face

Survey date: 2026-10-01. [`tools/survey_base_lora_fleet.py`](../../tools/survey_base_lora_fleet.py)
listed every repo tagged `base_model:adapter:<base>` for 10 video bases and read
each `.safetensors` header with bounded HTTP ranges (no tensor payloads). It ran on
the login node; raw output: `artifacts/video_lora_fleet_survey/login-20261001/`
(`headers.json`, `fleet_candidates.json`). 2,171 files probed, 2,138 parsed; the 33
failures are full checkpoints above the 12 GB adapter limit or headers over 8 MiB.
[`tools/summarize_base_lora_fleet.py`](../../tools/summarize_base_lora_fleet.py)
groups pure A/B adapters by a layer-shape fingerprint.

A shape fingerprint is necessary but not sufficient: Wan2.1-T2V-14B, Wan2.2 A14B
and Wan2.1-I2V-720P adapters share fingerprint `7ce77d9a711d` (40 blocks × 10
linears) yet attach to different base weights. Fleets below are therefore grouped
by base family *and* fingerprint. Many repos hold several epoch checkpoints of one
LoRA, so the "one-per-repo" column keeps only one file per repo.

| Base family | Fingerprint | Files | Repos | One-per-repo BF16 LoRA | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Wan2.1-I2V-14B-480P | `749193c5897d` (480 pairs) | 119 | 103 | 39.5 GB | mostly rank 32, 359 MB each; includes 49 Remade-AI effects |
| Wan2.1-T2V-14B | `7ce77d9a711d` (400 pairs) | 266 | 123 | 45.5 GB | many epoch checkpoints per repo |
| Wan2.2-I2V-A14B | `7ce77d9a711d` | 549 | 283 | — | high/low-noise expert pairs; dominated by NSFW repos |
| Wan2.2-T2V-A14B | `7ce77d9a711d` | 332 | 49 | — | same; few distinct repos |
| "LTX-Video" tag | `e7087099144a` | 84 | 83 | 67.2 GB | mislabelled LTX-2.3 adapters (Muapi), mostly NSFW |
| HunyuanVideo | `08a9e7d1531f` | 79 | 78 | 19.8 GB | rank 32 |

**Recommended pilot fleet: Remade-AI on Wan2.1-I2V-14B-480P.** 49 repos from one
trainer, identical layout (480 A/B pairs: self-attn, cross-attn including image
k/v, FFN), rank 32, BF16, 359 MB each, **17.6 GB** total. Each adapter is a
distinct visible effect (Rotate, Inflate, Squish, Cakeify, Crash-zoom-in, 360-Orbit,
…) with a trigger phrase in its model card, giving a per-task quality check by
comparing original and compressed outputs under fixed seeds. It can be extended to
the other ~54 same-layout I2V-480P repos (39.5 GB total) after content screening.

For scale: the Wan2.1 14B DiT is about 28 GB in BF16, so 49 resident adapters
equal roughly 60% of the DiT weights. Halving retained directions (b50), stored
compactly, would free up to ~8.8 GB of adapter memory. That is an upper bound on
adapter residency, realised only with an unmerged multi-adapter executor and
compact (not zero-padded) factors; it is not a measured end-to-end peak.

## Downloaded fleet: `remade_wan21_i2v_480p`

Fetched 2026-10-01 on the login node by
[`tools/fetch_lora_fleet.py`](../../tools/fetch_lora_fleet.py) into
`/nobackup/proj/disk/bloom/personal/shenghui/data/video_lora_fleets/remade_wan21_i2v_480p/`
(outside Git). Every file matched its Hub LFS sha256. `manifest.json` pins each
repo revision and records sha256, size, rank and the trigger phrase parsed from
the saved model card. Measured: 49 adapters, 17.60 GB on disk, 8.80 B LoRA
parameters; every adapter has rank 32 on all 480 modules, BF16, no `alpha`
tensors (loader default scaling applies), keys `diffusion_model.blocks.N.*`.
The Wan2.1-I2V-14B-480P base model has not been downloaded.
