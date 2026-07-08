# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

## Plan
- Substance in `docs/PLAN.md` (+ private `docs/PLAN.local.md`); order/status in `docs/ROADMAP.md`. Read at session start.

## Working directives (this project)
- North star: **domain generalization**. **No external dependencies** for this artifact.
- **All-synthetic training via a COMPOSITE of generation sources** (parametric / SSM / MRXCAT / label-space / learned) — each enters the generation DAG at a different point with a different control degree, unioned to cover the real manifold. See `core/data/dynamic/GENERATION.md`.
- **Two directions**: uncontrolled (diversity → training) + controlled (inverse / parametric **digital-twin**).
- **Physically-constrained diversity** for training (random contrast loses to physics; a single
  fidelity point loses to a swept physical manifold — cf. UltimateSynth > SynthSeg, our sweep >
  best-fit point); **tight fidelity** for the twin.
- **Headline metrics decided up front**; the comparison triad (real / synth-only / synth+DA) IS the result, not a leaderboard number.

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

**Branching:** work on `dev` (long-lived working branch); `main` stays stable. Push `dev`; merge `dev`→`main` at milestones (mirrors mindscape). `git push` below pushes the current branch — stay on `dev`, never commit directly to `main`.

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
- **WSL:** point uv's env off the shared `/mnt`-mounted tree to avoid clobbering the Windows `.venv`:
  `UV_PROJECT_ENVIRONMENT=$HOME/venvs/cardioseg uv sync --all-extras`, then `uv run …`. The data root
  needs NO override — `core.paths.resolve_data_root` auto-translates a Windows drive root to its `/mnt`
  mount under WSL (and raises, never goes relative, on non-WSL POSIX where it can't translate).
- Legacy conda env `pytorch_training_env` still works as a fallback (`conda run -n … python …`), but uv
  is the source of truth. NB the old `conda run` stdout-swallow / multiline-`-c` quirks don't apply to `uv run`.

## Architecture Overview

- `core/` — the shared kernel: `types`, `config`, `hparams`, `obs`,
  `model` (build_unet/load_run), `inference` (predict_volume + TTA), `evaluate`/`measure`/`postprocess`
  (Dice/HD95/EF/largest-CC), `export_onnx`, `preprocessing/` (resample/N4/z-score/fit_square).
  `data/` now splits in two: **`data/static/`** (store, splits, geo, `labels`, `reference`, `mri/*`
  adapters + registry — the read-side of the data cloud) and **`data/dynamic/`** (dataset, augment,
  synth, mri_physics, generator = the training data ENGINE, moved here from `cardioseg/data/`).
  **Dependency rule: core imports NEITHER cardioseg NOR cardioview** (still true) — but note
  `data/dynamic` holds MRI-specific batch generation, so the old "modality-agnostic" phrasing is
  loosened by owner decision: core is the shared kernel, not strictly modality-neutral. (bd cardiac-seg-t8p)
- **Config classes live with the class they configure** (pydantic `BaseModel`s, `validate_assignment`):
  `ModelCfg` in `core/model.py`, `N4Cfg` in `core/preprocessing/n4.py`, `DataCfg` in
  `core/data/static/store.py`, `AugCfg`/`SynthCfg`/`GeneratorCfg` in
  `core/data/dynamic/{augment,synth,generator}.py`, `LossCfg`+variants in `core/losses.py` (co-located
  with the loss classes so a cfg builds its own loss). `TrainCfg` (composition root) stays in
  `core/hparams.py`, which imports the dispersed cfgs to assemble it + re-exports for back-compat.
  Shared `_VALIDATE = ConfigDict(validate_assignment=True)` lives in `core/config.py` (a leaf all import
  without cycling).
- **Config polymorphism**: a cfg that selects a STRATEGY is a pydantic **discriminated union** — each
  variant owns its params + a virtual `build()` that instantiates its object (`cfg.loss.build()`,
  `cfg.acq.build()`, `cfg.bg.build()`). No `kind`/type dispatch (that's the serialization tag only).
  `--set X.kind=`/`X.mode=` rebuilds the variant (an `apply_overrides` shim); a `_lift_legacy` before-
  validator keeps old flat `config.json` loadable so registered models are unaffected. (bd cardiac-seg-bdw4)
- `cardioseg/` — training + eval *orchestration* on top of core: the pipeline ACDC/M&M-2/M&Ms-1/CMRxMotion
  → adapters (canonical labels 0=bg/1=RV/2=LV-myo/3=LV-cav) → consolidated store → 2D MONAI U-Net → EF
  → evaluate. Holds `training/`, `evaluation/` (validate/distribution/ensemble/calibrate/…),
  `preprocessing/normalization/` (dataset-metadata tooling). **Models live in the mlflow registry**
  (`core.registry`, NOT a `runs/` dir): a model = a run's artifacts (model.pth+config+onnx+card) under
  `cardioseg-2dunet` with aliases; flagship = the `production` alias. `resolve(ref)` (alias|version|
  run-id) downloads to a cache dir for the dir-consumers. Train registers via `--alias production`.
- `cardioview/` — browser viewer (TS+Vite+vtk.js): beating 3D hearts + in-browser ONNX segmentation.
  Depends only on `core` (never on cardioseg).
- `mri-sim/` — TS MRI acquisition visualizer (deprioritized).
- `baselines/nnunet/` — SOTA reference, quarantined (own env, never a cardioseg dep).
- `research/` — deep-dives; `learning/` — theory writeups. Public docs = README + `docs/PLAN.md` + `docs/ROADMAP.md`. The
  domain-shift / variance-taxonomy canonical doc lives in `cardioseg/preprocessing/normalization/README.md`.
- Numbers are single-sourced: `cardioseg/RESULTS.json` ← `evaluation/results.py`; `cardioseg/evaluation/sync_numbers.py` fills doc marker blocks.

## Static analysis — the ruff gate (bd cardiac-seg-l79o)

CI requires `tests` + `lint` on dev→main (branch protection). `lint` = **enforced** ruff (pinned
`uvx ruff@0.15.13 check core cardioseg --select F,I,B,S110,BLE001,C901,PLR0912,PLR0913,PLR0915,T201,PLR2004,PLC0415,FBT,RUF100`)
+ an **advisory** full run (`ruff check . --statistics || true`) that surfaces families still being
cleaned. **Ratchet**: a family graduates into the enforced `--select` once it hits 0 — fix it, don't
dodge (no limit-bumps, no noqa spray). The linter is a **bug-finder** (it caught an F821 NameError and
a comment jammed mid-expression that dead-coded a physics term), not cosmetics. What the gate encodes:
logging not prints (T201), named constants not magic numbers (PLR2004), keyword-only bools (FBT),
specific-exception-or-let-it-crash (BLE001/S110 — no blind `except`), low complexity / strategy pattern
(C901/PLR0912/0915), config objects over long arg lists (PLR0913), imports-at-top (PLC0415 — fix
circulars by **extraction**, never lazy imports), **no dead noqa** (RUF100). noqa policy: **bare
`# noqa: RULE`** — no prose reasons. **Minimal comments** — prefer self-documenting code + names.
`getattr`/`hasattr` on a constant attr is a smell (B009 catches the no-default form). E501 stays
advisory (the dense hand-tuned style). Config `pyproject [tool.ruff]`; evaluating vulture + coverage
next (advisory).

## Conventions & Patterns

- **Tasks → `bd` (beads) only** (see above). **Data lives out of repo** under `<data>/raw/` (gitignored).
- **External git-repo deps live gitignored under `external/`** — checked out, never vendored (don't commit third-party code). Commit the URL + pinned commit (in a doc + the fetch step that lives with the consuming lane, not a top-level `scripts/`) so the checkout is reproducible. Only their gated/large DATA stays out; the *how-to-fetch* is committed. (e.g. `external/mrxcat2` ← public ETH MRXCAT2.0, pinned; see `core/data/dynamic/GENERATION.md`.)
- Env quirks: `conda run` swallows stdout + rejects multiline `-c` (write script files); `/tmp` doesn't survive
  Windows-python round-trip (use repo-relative paths). CRLF→LF + beads "auto-export git add failed" warnings are benign.
