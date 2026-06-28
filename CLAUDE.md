# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:3216161c -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->


## Build & Test

```bash
pip install -e ".[all]"      # core + extras (n4/export/viz/nnunet/dev). pyproject = source of truth.
# torch FIRST from CUDA index (plain PyPI is CPU): pip install torch --index-url https://download.pytorch.org/whl/cu128
conda activate pytorch_training_env          # the working env here
export PYTHONPATH=/d/personal_projects/cardiac-seg   # scripts run as modules from repo root
pytest                                        # testpaths = tests/ (unit + integration)
```

## Architecture Overview

- `core/` — the shared, modality-agnostic kernel: `types`, `config`, `hparams`, `obs`, `labels`,
  `model` (build_unet/load_run), `inference` (predict_volume + TTA), `evaluate`/`measure`/`postprocess`
  (Dice/HD95/EF/largest-CC), `export_onnx`, `preprocessing/` (resample/N4/z-score/fit_square),
  `data/` (store, splits, geo, mri/* adapters + registry). **Dependency rule: core imports NEITHER
  cardioseg NOR cardioview.** (bd cardiac-seg-t8p)
- `cardioseg/` — training + eval *orchestration* on top of core: the pipeline ACDC/M&M-2/M&Ms-1/CMRxMotion
  → adapters (canonical labels 0=bg/1=RV/2=LV-myo/3=LV-cav) → consolidated store → 2D MONAI U-Net → EF
  → evaluate. Holds `training/`, `evaluation/` (validate/distribution/ensemble/calibrate/…),
  `preprocessing/normalization/` (dataset-metadata tooling). Flagship run = `runs/gen`.
- `cardioview/` — browser viewer (TS+Vite+vtk.js): beating 3D hearts + in-browser ONNX segmentation.
  Depends only on `core` (never on cardioseg).
- `mri-sim/` — TS MRI acquisition visualizer (deprioritized).
- `baselines/nnunet/` — SOTA reference, quarantined (own env, never a cardioseg dep).
- `research/` — deep-dives; `learning/` — theory writeups. Public docs = README + ROADMAP. The
  domain-shift / variance-taxonomy canonical doc lives in `cardioseg/preprocessing/normalization/README.md`.
- Numbers are single-sourced: `cardioseg/RESULTS.json` ← `evaluation/results.py`; `cardioseg/evaluation/sync_numbers.py` fills doc marker blocks.

## Conventions & Patterns

- **Tasks → `bd` (beads) only** (see above). **Data lives out of repo** under `<data>/raw/` (gitignored).
- Env quirks: `conda run` swallows stdout + rejects multiline `-c` (write script files); `/tmp` doesn't survive
  Windows-python round-trip (use repo-relative paths). CRLF→LF + beads "auto-export git add failed" warnings are benign.
