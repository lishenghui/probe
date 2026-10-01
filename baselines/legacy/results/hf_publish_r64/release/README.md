---
license: mit
base_model: openvla/openvla-7b
base_model_relation: adapter
pipeline_tag: robotics
tags:
- openvla-oft
- lora
- libero
- reconstructed-adapter
---
# OpenVLA-OFT LIBERO reconstructed rank-64 adapters

Four approximate rank-64 adapters reconstructed by LoRAFleet from the published
[OpenVLA-OFT](https://github.com/moojink/openvla-oft) merged checkpoints by moojink:
`spatial`, `object`, `goal`, and `10` (LIBERO-10).
These are reconstructed adapters, not the publisher's original training adapters.

## Shared base and method

All four target the official [openvla/openvla-7b](https://huggingface.co/openvla/openvla-7b)
weights at revision `47a0ec7fc4ec123775a391911046cf33cf9ed83f`.
All three local base weight shards were SHA-256 verified against that revision's
Hub LFS hashes; see `base_verification.json`.

For each of 439 target modules, factorize `W_published - W_base` using randomized
SVD with our custom FlashTSQR GPU implementation, sketch width 192, two power
iterations, seed 0, output rank 64. The projected small core uses SVD.
Factors are BF16, `B = U*S`, `A = Vh`, with scale 1 (`r = lora_alpha = 64`).
The other 543 backbone tensors match the base exactly for every suite.
Each adapter has **221,656,576 A/B parameters**, stored in 443,446,440 bytes.
This release reconstructs a common-base representation; it is not a further
rank-budget compression of those reconstructed adapters and is not lossless.

Each suite contains `lora_adapter/`, its original action head and proprioception
projector, `dataset_statistics.json`, reconstruction diagnostics, and evaluation
records. Task heads and statistics are suite-specific and are required.
Their bytes were verified against the pinned publisher revisions in `manifest.json`.
The 7B base weights are not duplicated here.

## Download and integration

```python
from huggingface_hub import snapshot_download
release = snapshot_download("LoRAFleet/openvla-oft-libero-reconstructed-r64")
base = snapshot_download(
    "openvla/openvla-7b",
    revision="47a0ec7fc4ec123775a391911046cf33cf9ed83f",
)
```

Use the [OpenVLA-OFT LIBERO runtime and setup](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md),
including its continuous action head, proprioception projector, image processing,
and action normalization. A stock OpenVLA action decoder is insufficient.
Publisher repository IDs and fixed revisions for fetching their runtime metadata
are recorded per suite in `manifest.json`.

`apply_adapter.py` provides `apply_adapter(model, base_directory, suite_directory)`:
pass an instantiated OFT-compatible backbone with the published module names.
It explicitly replaces each target weight with the official base tensor plus the
FP32 product of the saved factors, then casts to the model weight dtype. This
matches the reconstruction arithmetic used in our evaluation and also handles
Conv2d factors. It leaves non-target backbone tensors untouched, so those must
already contain the official base weights. Load the suite-specific heads and
statistics separately through the OFT runtime. Do not apply this adapter on top
of the publisher's merged target weights by ordinary additive merging.
The PEFT-format files have portable base ID/revision fields; standalone stock
PEFT/OFT end-to-end loading has not been validated by this release.

## Limited rollout validation

Only task ID 0, eight initial states per suite, was evaluated. This is a smoke
test, not a full LIBERO benchmark or proof of equivalent action distributions.

| Suite | Published successes | Reconstructed successes | Batched policy calls: published / reconstructed |
|---|---:|---:|---:|
| spatial | 8/8 | 8/8 | 11 / 11 |
| object | 8/8 | 8/8 | 19 / 19 |
| goal | 8/8 | 8/8 | 18 / 24 |
| 10 | 8/8 | 8/8 | 46 / 51 |

Evaluation instantiated the published OFT model and replaced all 439 target
weights with `W_base + B@A`; the remaining 543 tensors were audited as exactly
base-equal. A live multi-adapter shared-base serving system was not benchmarked.
Policy-call counts are batched runner calls, not per-episode environment steps.

## Attribution

Base model: OpenVLA authors, https://github.com/openvla/openvla (MIT).
Source fine-tuned models and OFT runtime: OpenVLA-OFT authors / moojink,
https://github.com/moojink/openvla-oft (MIT).
Reconstruction and FlashTSQR implementation: LoRAFleet.
Please also credit the upstream OpenVLA and OpenVLA-OFT work when using these artifacts.
