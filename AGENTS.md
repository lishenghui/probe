# Repository instructions

These instructions apply throughout this repository, including new agent
conversations. Read them before planning or running compression experiments.

## Default compression backend: our FlashTSQR implementation

- For LoRA/adapter spectral compression, use this repository's custom
  **FlashTSQR GPU implementation** by default. Using our accelerated algorithm
  is part of the experiment requirement, not merely a performance preference.
- The full operator is `FlashTSQR/kernels/tsqr_full.cu`. The existing OpenVLA-OFT
  integration is `vla_fleet/compress_oft_gpu.py`, which calls `tsqr_factor` and
  `tsqr_applyQ` around the small-core SVD. Check its current implementation and
  supported options before constructing a command.
- Do not silently substitute `vla_fleet/compress_oft.py`, `torch.linalg.qr`, or
  another generic QR backend for the custom TSQR path in production compression
  runs. The CPU implementation is a reference for correctness comparisons.
- If the GPU script lacks a workflow feature (for example adapter-only output,
  compression manifests, or validation), add that feature to the accelerated
  path rather than choosing the CPU script for convenience. Avoid copying large
  merged checkpoints when shared metadata and adapter-only output suffice.
- Run the accelerated path on a compatible allocated GPU node. A login node
  without a compatible GPU or Python environment is not a reason to silently
  fall back to CPU compression.
- If the custom backend cannot support the requested operation, explain the
  concrete blocker and proposed fallback before using it. Explicit user
  instructions authorizing a different backend take precedence.

## Experiment provenance and reporting

- Record the compression backend, source adapter, retained-direction budget,
  and output paths in experiment metadata or logs. Verify that the custom
  kernel actually ran before describing results as FlashTSQR results.
- Distinguish the compression method (global spectral allocation and
  truncation) from its implementation (custom TSQR versus reference QR).
  Equivalent mathematical objectives do not establish use of our kernel.
- `b50` means retaining 50% of the original rank directions. Distinguish retained
  directions, compact A/B parameter counts, and actual stored tensor sizes;
  zero-padded factors do not realize storage or inference-compute savings.
- For a new budget comparison, compress from the original adapter unless the
  user explicitly requests iterative recompression. Preserve existing results
  and identify the scope of rollout coverage when reporting quality.

## PROBE iteration rules

- Follow the Arrhenius HPC instructions supplied by the user; use Slurm for compute.
- Preserve `baselines/legacy/` as historical evidence; write new results under `artifacts/`.
- `legacy_launchers/` is archival reference, not an executable submission workflow. Its accounts, environments, and output paths require review before reuse.
- The imported paper calls the method FRA; PROBE is the new project name. Do not relabel old results as new runs.
- Never write into the source LoRAForge checkout or its results.
