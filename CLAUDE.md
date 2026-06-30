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

**uv** is the env+dep manager (one `pyproject.toml` + committed `uv.lock`, both platforms). torch's
CUDA wheel comes from the PyTorch index pinned in `[tool.uv.sources]` — uv resolves it automatically,
no manual index step.

```bash
uv sync --all-extras     # create/refresh .venv from pyproject + uv.lock (torch+cu130 incl.)
uv run pytest            # runs in .venv; editable install -> no PYTHONPATH needed
uv run python -m cardioseg.training.train --out runs/foo   # any entrypoint via `uv run`
```

- **Linux GPU lane:** `uv sync --all-extras` on linux also pulls the `gpu` extra (cupy/cucim, linux-
  only by marker); on Windows those silently skip and the code falls back to CPU. Same command both OSes.
- **WSL:** repo lives on `/mnt/d`, so (a) point uv's env off the shared tree to avoid clobbering the
  Windows `.venv`: `UV_PROJECT_ENVIRONMENT=$HOME/venvs/cardioseg uv sync --all-extras`; and (b) override
  the data root, since `paths.yaml` holds a Windows path (`D:/…`) invalid on linux:
  `CARDIAC_DATA=/mnt/d/<...>/mri`. Full WSL run:
  `cd /mnt/d/personal_projects/cardiac-seg && CARDIAC_DATA=/mnt/d/data/volumetric/mri UV_PROJECT_ENVIRONMENT=$HOME/venvs/cardioseg uv run python -m …`
- Legacy conda env `pytorch_training_env` still works as a fallback (`conda run -n … python …`), but uv
  is the source of truth. NB the old `conda run` stdout-swallow / multiline-`-c` quirks don't apply to `uv run`.

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
