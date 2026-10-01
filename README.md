# PROBE

New iteration workspace for multi-adapter fleet allocation, imported from
`lishenghui/LoRAForge` and its `lishenghui/ruller` paper submodule.
The preserved paper calls the method **FRA**; the supplied newer excerpt calls it PROBE.
This import does not claim a new experiment or a numerical reproduction.

## Layout

- `ruller-paper/experiments/rq3/`: allocation, compression, evaluation and analysis code.
- `ruller-paper/ruller-short/`: historical paper and figures for the 12/25/41 fleets.
- `FlashTSQR/`, `tools/`, `loraforge_kernels/`, `configs/`: compression kernels and supporting code.
- `baselines/legacy/results/`: independent copies of historical JSON, CSV and Markdown results (including other legacy experiments).
- `baselines/legacy/import_manifest.json`: source commits, source working-tree status and copied-file inventory.
- `figure2_repro/`: plotting scripts and expected outputs. Its raw inputs are in the baseline snapshot, not duplicated here.
- `legacy_launchers/`: historical shell/Slurm scripts, for reference only; these retain old paths and account settings.
- `artifacts/`: ignored working area for new runs. Never write new output to the baseline snapshot.

## Starting points

`two_level_allocation.py` implements allocation; `solve_two_level_fleet.py`
solves fleet minimax from measured curves. `land_two_level_curve.py` constructs
curves. `build_fra_clean_ablation.py` builds spectral/functional/FRA allocations.
`predibase_task_metrics.py`, `cts_task_metrics.py`, and `lorare_task_metrics.py`
evaluate LoRA Land, Lots-of-LoRAs, and LoRARetriever respectively.
Internal names are `land12`, `cts25`, and `lorare`.

Run scripts from this repository root. Most historical builders use
`artifacts/rq3/results` as a relative input/output directory. If needed, seed it
with independent copies of the preserved baseline (never symlink it):

```bash
cp -n baselines/legacy/results/*.json artifacts/rq3/results/
```

Figure 2 accepts explicit paths:

```bash
python figure2_repro/plot_frontier_comparison.py \
  --results baselines/legacy/results \
  --output artifacts/figure2/frontier_comparison.pdf
```

## HPC and environments

Source `~/.config/hpc-agent/bootstrap.sh` before experiments. Use Slurm for
compute and follow the user-provided Arrhenius rules. Confirm the live account
and pass it via `sbatch -A "$SLURM_ACCOUNT"`, with logs in `$LORAM_LOG_DIR`.
Do not directly submit the archived launchers. They document the old interpreter,
module, model paths and parameters; verify these before building a new launcher.
No environment was created and no GPU experiment was run during import.
Model weights, datasets and environments remain at their original locations;
pass verified input paths explicitly. No large assets were copied to Git.
The source checkout and its original results remain untouched.
