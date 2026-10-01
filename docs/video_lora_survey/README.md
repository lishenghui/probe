# Large LoRA adapters in video and image generation

Survey date: 2026-10-01. The measurements below come from each public
`safetensors` header, read with bounded HTTP range requests by
[`tools/survey_remote_lora_headers.py`](../../tools/survey_remote_lora_headers.py).
No tensor payload or full checkpoint was downloaded. The raw header summaries
are in `artifacts/video_lora_survey/headers.json` (Slurm job 3224442).
File sizes use decimal GB; parameter counts include every tensor in the file.

| Published adapter | Modality | LoRA rank | Parameters in file | File size | Tensor dtype | Pure A/B LoRA? |
| --- | --- | ---: | ---: | ---: | --- | --- |
| [LTX-2 distilled](https://huggingface.co/Lightricks/LTX-2/blob/main/ltx-2-19b-distilled-lora-384.safetensors) | video/audio | mostly 384 | 3.837B | 7.675 GB | BF16 | Yes, 1,371 pairs |
| [Wan2.1 Pusa](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Pusa/Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors) | video | 512 | 2.454B | 4.907 GB | BF16 | Yes, 400 pairs |
| [Wan2.1 LightX2V](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16.safetensors) | video | mostly 256 | 1.249B | 2.498 GB | BF16 | No, 647 additional tensors |
| [HunyuanVideo AnimeShots](https://huggingface.co/trojblue/HunyuanVideo-lora-AnimeShots/blob/main/v0.1/adapter_model.safetensors) | video | 32 | 0.161B | 0.323 GB | BF16 | Yes, 320 pairs |
| [Wan2.1 identity example](https://huggingface.co/malcolmrey/wan/blob/main/wan2.1/wan_amandapeet_v1.safetensors) | video | 32 | 0.153B | 0.307 GB | BF16 | Yes, 400 pairs |
| [ByteDance Hyper-FLUX 8 steps](https://huggingface.co/ByteDance/Hyper-SD/blob/main/Hyper-FLUX.1-dev-8steps-lora.safetensors) | image | 64 | 0.347B | 1.388 GB | FP32 | Yes, 504 pairs |
| [FLUX Krea BLAZE](https://huggingface.co/MintLab/FLUX-Krea-BLAZE/blob/main/LORA/Flux_Krea_Blaze_Lora-rank128.safetensors) | image | mostly 128 | 0.622B | 1.244 GB | FP16 | No, 314 additional tensors |

These examples answer the size question: some video LoRAs contain billions of
trainable parameters and occupy multiple GB as BF16 weights. But video alone
does not cause large adapters: the two rank-32 video examples are about 0.3 GB.
For a linear layer with shape `d_out × d_in`, a rank-`r` LoRA adds
`r × (d_in + d_out)` parameters. Model width, the set of targeted layers,
rank, dtype, and number of simultaneously resident adapters determine memory.

**Implications for PROBE.** A 4.9–7.7 GB pure BF16 adapter gives much more
absolute memory to remove than the small LLM adapters in the current fleets.
For example, reducing an adapter's stored tensor count by 50% could save
about 2.45 GB for Wan Pusa or 3.84 GB for LTX-2, assuming it remains BF16
and resident independently. This is an upper bound for *adapter-weight* memory,
not a prediction of end-to-end peak GPU memory: the base model, text encoder,
VAE, and video activations also consume memory. Runtime speedup requires an
unmerged LoRA execution path and a measured reduction in computation or memory
traffic. If the adapter is fused into dense base weights before inference,
rank compression by itself need not change steady-state inference cost.

The first clean pilot is a **single pure A/B high-rank adapter** (Wan Pusa or
LTX-2), with the same base model, prompts, video dimensions, denoising steps,
precision, and executor across compressed and original variants. Measure
quality, adapter residency, peak GPU memory, cold-load time, and end-to-end
generation latency. Compare fused and unmerged paths separately. LTX-2's
distilled adapter changes the denoising regime, so comparisons against a
different adapter or step count would conflate LoRA compression with
distillation. LightX2V and FLUX Krea BLAZE require accounting for their extra
tensors before calling total file size a LoRA compression opportunity.

The FP32 Hyper-FLUX file illustrates a second pitfall: its 1.388 GB file would
represent about 0.694 GB of tensor payload if cast to BF16 at loading time.
Checkpoint bytes therefore cannot be substituted for observed GPU residency.
