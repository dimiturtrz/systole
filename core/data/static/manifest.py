"""Frozen test manifests — the comparability foundation.

A split is criteria over a GROWING store, so its resolved test set drifts as data lands (splits.py:
"a split isn't a named thing"). For TEST that silently breaks comparability: an old model and a new
one get scored on different sets that happen to share a name. A manifest freezes a test set's
IDENTITY — a pinned `(dataset, subject_id)` list — versioned in-repo, immutable. Provenance
(criteria, snapshot, date) travels with it; a `task` tag says which metrics are valid:
`seg4` (4-class), `seg_lv` (SCD: no RV, classes {0,2,3}), `ef` (Kaggle: no masks, EF-only).

Identity is store-param-independent: the manifest pins (dataset, subject_id), and `resolve_paths`
binds those ids to the CURRENT store's npz `path` column — so the SAME frozen test is scoreable
under any preprocessing (n4/nyul/norm/inplane). Test identity is fixed; preprocessing is a model-side
choice. This is what lets the generalization matrix back-fill: score any registered model on any
manifest by pure inference (evaluation.validate), old models included, forever comparable.

Freezing is TEST-only. Train/val stay criteria over the current store (splits.make_split) — the
training side keeps its flexibility; only the honest immutable report is pinned.

    from core.data.static import store, manifest
    meta = store.load()                                  # current cloud
    manifest.freeze("vendor_canon", meta, test_vendors=["Canon"], task="seg4",
                    note="unseen-vendor stress; Canon only in mnms1")
    m = manifest.load("vendor_canon")
    paths, missing = manifest.resolve_paths(m, store.load(norm="zscore"))   # any preprocessing
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

# In-repo (versioned, committed) — the manifests ARE the comparability contract, they ride with the
# code. They hold only public (dataset, subject_id) identifiers: no machine paths, no images, no PII.
MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"

TASKS = ("seg4", "seg_lv", "ef")


def _key(df: pl.DataFrame) -> pl.DataFrame:
    """Add the identity key = 'dataset\\tsubject_id' (subject_id cast to str: it's int for some sets)."""
    return df.with_columns(
        (pl.col("dataset") + "\t" + pl.col("subject_id").cast(pl.Utf8)).alias("_key"))


def freeze(name: str, meta: pl.DataFrame, *, test_datasets=(), test_vendors=(),
           task: str = "seg4", note: str = "", labelled_only: bool = True,
           created: str | None = None) -> Path:
    """Resolve criteria over `meta` NOW and pin the matching subjects to manifests/<name>.json.
    IMMUTABLE by contract: to change a test set, freeze a new name — never edit an existing file.
    `created` = provenance date (pass explicitly for reproducible commits; else today)."""
    if task not in TASKS:
        raise ValueError(f"task {task!r} not in {TASKS}")
    if MANIFEST_DIR.joinpath(f"{name}.json").exists():
        raise FileExistsError(f"manifest {name!r} exists — manifests are immutable; pick a new name")
    if created is None:
        from datetime import date
        created = date.today().isoformat()
    expr = (pl.col("dataset").is_in(list(test_datasets))
            | pl.col("vendor").is_in(list(test_vendors)))
    if labelled_only:
        expr = expr & pl.col("labelled")
    sel = meta.filter(expr).sort(["dataset", "subject_id"])
    subjects = [[d, str(s)] for d, s in zip(sel["dataset"], sel["subject_id"])]
    if not subjects:
        raise ValueError(f"criteria matched 0 subjects (datasets={test_datasets} vendors={test_vendors})")
    doc = {
        "name": name,
        "task": task,
        "created": created,
        "criteria": {"test_datasets": list(test_datasets), "test_vendors": list(test_vendors)},
        "note": note,
        "snapshot": {"store_total": len(meta), "n": len(subjects),
                     "by_dataset": dict(sel.group_by("dataset").len().sort("dataset").iter_rows())},
        "subjects": subjects,
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    out = MANIFEST_DIR / f"{name}.json"
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return out


def load(name: str) -> dict:
    """Read manifests/<name>.json."""
    p = MANIFEST_DIR / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"no manifest {name!r} in {MANIFEST_DIR}")
    return json.loads(p.read_text(encoding="utf-8"))


def list_manifests() -> list[str]:
    """Names of all frozen manifests."""
    return sorted(p.stem for p in MANIFEST_DIR.glob("*.json"))


def resolve_paths(m: dict, meta: pl.DataFrame) -> tuple[list[str], list[list]]:
    """Bind a manifest's frozen (dataset, subject_id) ids to the CURRENT store's npz paths.
    Returns (paths, missing) — `missing` = frozen ids absent from `meta` (store drift: a dataset not
    loaded, or a subject dropped). Non-empty `missing` means the score is over a SUBSET — the caller
    must surface it, never silently score fewer cases."""
    want = {f"{d}\t{s}" for d, s in m["subjects"]}
    keyed = _key(meta).filter(pl.col("_key").is_in(want))
    have = set(keyed["_key"])
    missing = [k.split("\t") for k in sorted(want - have)]
    return keyed["path"].to_list(), missing
