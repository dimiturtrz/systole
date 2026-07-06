# Repository structure

An architecture map for the sprawl (74+ py files). For *what* the project is, see the root `README.md`;
for *plan/status*, `docs/PLAN.md` + `docs/ROADMAP.md`. This file is the *where things live* map.

## The three-sibling shape

```
core/  ── the shared kernel (modality-ish-agnostic engine)
  ▲            ▲
  │            │
cardioseg/   cardioview/        (future: ct/, echo/)
 training      browser viewer
 + eval
```

**Dependency rule:** `cardioseg → core`, `cardioview → core`, and **neither imports the other**.
`core` imports **nothing** from `cardioseg`/`cardioview`. The 2nd modality is the forcing function that
keeps this honest.

## Top level

| entry | what |
|---|---|
| `core/` | shared kernel — types, config, model, inference, evaluate, preprocessing, **data/** |
| `cardioseg/` | training + eval **orchestration** on core (`training/`, `evaluation/`, `preprocessing/normalization/`) |
| `cardioview/` | browser viewer (TS + Vite + vtk.js) — beating hearts + in-browser ONNX seg |
| `baselines/` | nnU-Net SOTA reference (quarantined, own env, never a cardioseg dep) |
| `mri-sim/` | TS MRI-acquisition visualizer (deprioritized) |
| `docs/` | `PLAN.md`, `ROADMAP.md`, this file |
| `research/`, `learning/` | deep-dives; theory writeups |
| `tests/` | `unit/`, `integration/` |
| `scripts/` | one-off helpers |
| `.beads/` | issue tracker (`issues.jsonl` tracked; dolt engine gitignored) |
| `pyproject.toml` + `uv.lock` | uv env + deps (one lock, both platforms) |
| `paths.yaml` (gitignored) | the machine's data-root path; copy from `paths.example.yaml` |

**Data lives OUT of the repo** at the configured root (`paths.yaml` → e.g. `D:/data/volumetric/mri`):
`raw/<dataset>/` (pristine downloads), `processed/<dataset>/<paramkey>/` (preprocess cache, auto-built),
`reference/` (nyul/acquisition), and meshes beside it (`meshes/`). None of these are in git.

## `core/` — the kernel

```
core/
  types config hparams obs paths registry            # primitives + mlflow model registry
  model inference evaluate measure postprocess mesh export_onnx   # model + eval kernel
  preprocessing/  { preprocess, n4, nyul }
  data/
    ingest/     ← the unifying ingestion layer (real + synth are peer SOURCES)
      source.py            Source protocol + StaticSource
      split.py             SplitDef + resolve (train=complement) + Resolution
      splits/              coded split FAMILIES: static_main, synth_main (+ registry)
      testsets.py          coded TestSet (predicate + hash lock) + --freeze/--check CLI
      testsets.lock.json   the frozen locks (committed; --freeze writes, --check drift-guards)
    static/     ← the REAL-data source (read side of the cloud)
      store.py             DataCfg + consolidate + load() = the data cloud
      splits.py            make_split / model_val / split_from_cfg  (criteria evaluator, legacy)
      labels geo reference
      mri/  { acdc mnm2 mnms1 cmrxmotion  scd(DICOM) kaggle_dsb(EF)  dicom base registry pathology }
    dynamic/    ← the SYNTH source + the batch ENGINE
      source.py            DynamicSource (synth train_gen)
      generator.py         the batch engine (runs the pipeline)
      pipeline.py          composable transforms: SynthReplace | Augment | Soften
      synth augment mri_physics anatomy dataset inverse mrxcat
      GENERATION.md        the composite-generation-DAG design
    analysis/   ← diagnostics (not the train path)
      attribution render synth_fidelity shape_coverage sim2real static_compare eda viz
```

### The ingestion spine (the heart of the data layer)

```
static/ (real cloud)  ┐
                      ├──►  ingest/ : Source · Split · TestSet  ──►  cardioseg training/eval
dynamic/ (synth gen)  ┘         (coded, versioned, hash-frozen)
```

- A **Split** family (`ingest/splits/`) is CODE — coded polars filters, versioned, `train`=complement.
  Its `test` is a **TestSet** (`ingest/testsets.py`): a coded predicate + a content-hash `lock` so the
  test set is frozen + comparable across store growth (no string-named manifests).
- A **Source** (`StaticSource`/`DynamicSource`) owns its own batch engine (`train_gen`); real and synth
  flow through one seam — no `if real/synth` in training. `resident()` gives raw tensors for val/test.
- The **generalization matrix** (`cardioseg/evaluation/matrix.py`) scores any registered model × any
  TestSet, flagging OOD vs leak from the model's reconstructed SEEN set (`train ∪ val`).

**cardioseg** never re-implements any of this — it *orchestrates* it (train loop, registry, reporting).

## Conventions

- **Config classes live with the class they configure** (before-init pydantic): `ModelCfg` in `model.py`,
  `DataCfg` in `data/static/store.py`, `AugCfg`/`SynthCfg`/`GeneratorCfg` in `data/dynamic/*`. `TrainCfg`
  (composition root) in `core/hparams.py` re-exports them.
- **Models live in the mlflow registry** (`core.registry`), not a `runs/` dir. Flagship = `production` alias.
- **Numbers single-sourced**: `cardioseg/RESULTS.json` ← `evaluation/results.py` → doc markers via `sync_numbers.py`.
- **Tasks → beads** (`bd`), not markdown TODOs. **Data out of repo**, gitignored.
