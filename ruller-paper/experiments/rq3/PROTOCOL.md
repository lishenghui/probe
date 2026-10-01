# RQ3 behavioral evaluation protocol

This protocol is fixed before running behavioral evaluations. Its purpose is to
measure the behavioral cost caused by post-hoc rank reduction, not to compare
the adapter with its base model.

## Conditions shared by all benchmarks

Each adapter is evaluated in four conditions: the original checkpoint and
layer-wise SVD truncations retaining 99%, 95%, or 90% of each nonzero update's
squared spectral energy. For a target threshold `tau`, the smallest retained
rank satisfying the threshold is used independently in every layer. The
refactored factors and LoRA scaling must reconstruct the truncated effective
update; changing rank must not silently change `lora_alpha / r`. Zero updates
remain zero. Inputs, preprocessing, decoding or sampling parameters, and random
seeds are identical across the four conditions.

The primary outcome is the paired quality change from the original adapter.
Report every adapter/threshold result, retained parameters, median change, worst
change, and a paired bootstrap 95% confidence interval over evaluation items.
The base model is an optional context baseline and is not the RQ3 denominator.

## LLM: ReMamba on LongBench-E

- Adapter: `lblankl/ReMamba`, revision `fa33cea292d43034c60eee576afe716cd036fc0a`
- Base: `state-spaces/mamba-2.8b-hf`, revision `96c48e0292b63f5346b6d30061af2551f7101e26`
- Paper: *ReMamba: Equip Mamba with Effective Long-Sequence Modeling*
  (arXiv:2408.15496; Findings of EMNLP 2025)
- Test data: `THUDM/LongBench`, revision `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`
- Scope: the 13 English LongBench-E tasks used in the paper, maximum input
  length 6,144 tokens, and the paper's prompt templates.
- Metric: official metric per task and unweighted macro-average across tasks.

## Additional pure-LLM replication: Qwen2.5-1.5B on MBPP+

This case was added after the initial cases above and is reported as a
replication rather than as preregistered evidence.

- Adapter: `tokhey/qwen2.5-1.5b-mbpp-reasoning-sft`, revision
  `4a58413a1b015fd174b4a98968b98860d7f864c8`.
- Base: `unsloth/Qwen2.5-1.5B-Instruct`, revision
  `b2e27ed8774d78eb2ee474cfe99d2d3b5fae11e5`.
- Test data: all 378 MBPP tasks distributed by EvalPlus 0.3.1.
- Evaluation: merged adapter, vLLM 0.23.0, greedy decoding, one sample per
  task, and the official EvalPlus execution harness.
- Metrics: pass@1 on the original MBPP tests and on MBPP+ (base plus extra
  tests). The paired audit compares generated programs and per-task pass/fail
  changes against the original adapter.

## VLM: EditScore-Qwen3-VL-4B on EditReward-Bench

The adapter is an official release by the EditScore authors, and both the
benchmark and evaluator are official artifacts from the same project. The
Qwen3-VL checkpoint was released later by the project than its initial model
series.

- Adapter: `EditScore/EditScore-Qwen3-VL-4B-Instruct`, revision `87dcfd0eba27335c036ef9616fa0e725141ba571`
- Base: `Qwen/Qwen3-VL-4B-Instruct`, revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`
- Paper: *EditScore: Unlocking Online RL for Image Editing via High-Fidelity
  Reward Modeling* (arXiv:2509.23909; ICLR 2026)
- Test data: complete `EditScore/EditReward-Bench`, revision
  `7dc4a0ff83c143ea0476065b2bb5d35247af0fcf` (1,739 images, 13 subtasks).
- Evaluation: official `VectorSpaceLab/EditScore` evaluator and deterministic
  prompt. No self-ensemble in the primary comparison.
- Metric: overall human-preference alignment/ranking accuracy and per-subtask
  accuracy.

## Video: Wan2.2 LightX2V I2V on VBench++ I2V

- Adapter system: both files below from `lightx2v/Wan2.2-Distill-Loras`,
  revision `570044187a5219776ef30a5c60c6f76428a3a10a`:
  - `wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors`
  - `wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors`
- Base: `Wan-AI/Wan2.2-I2V-A14B`, revision `206a9ee1b7bfaaf8f7e4d81335650533490646a3`
- Project: `ModelTC/LightX2V-Wan2.2-Lightning`; base-model conventions follow
  Wan2.2. The public project specifies steps `[1000, 750, 500, 250]`.
- Test data: VBench++ I2V subject/background suites at VBench commit
  `45e79ec14e69a2187202c675d2dbce1a71843d53`: 355 public image-prompt inputs
  (246 subject and 109 background inputs).
- Generation: 832x480 target area, 81 frames, 16 fps, four denoising steps,
  fixed seed 42, no prompt rewriting. Compress both LoRAs at the same threshold
  and use them together in every condition.
- Metrics: `i2v_subject`, `i2v_background`, `subject_consistency`,
  `background_consistency`, `motion_smoothness`, `dynamic_degree`,
  `aesthetic_quality`, and `imaging_quality`, plus their unweighted average.
  Camera motion is excluded from the primary suite because it uses a separate
  763-prompt pool; it may be a pre-specified secondary experiment.

## Resource estimate

The adapters are already local. Approximate upstream repository sizes are
10.31 GiB for Mamba-2.8B, 8.28 GiB for Qwen3-VL-4B, and 117.54 GiB for the
complete Wan2.2-I2V-A14B runtime. Generated video storage must be budgeted
separately for 355 inputs x 4 conditions.
