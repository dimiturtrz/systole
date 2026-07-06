"""Cross-domain generalization matrix — score registered models on frozen coded TestSets.

Pure inference, no retrain: this is what the frozen TestSets buy. Any model, any test set, forever
comparable (an old model and a new one score on the SAME lock-frozen subjects). For each (model,
testset) cell: resolve the model (registry) with its OWN preprocessing, resolve the TestSet to current
npz paths (its lock guards drift), validate. A cell is OOD (the honest generalization number) when NONE
of the test subjects were in the model's SEEN set (train ∪ val — a val subject IS seen) — reconstructed
as a coded filter from the model's saved config: a coded split family (new) or DataCfg criteria (old).
A synth-trained model's SEEN = its real val only. Otherwise the cell is in-domain (a leak) and flagged.

Task per TestSet: seg4 -> all three classes; seg_lv (SCD, no RV in GT) -> myo+cav only.

    python -m cardioseg.evaluation.matrix --models production 60 61 --testsets canon ge scd_lv
    python -m cardioseg.evaluation.matrix --models 60 61            # the default granular battery
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from core.data.ingest.testsets import MATRIX_TESTSETS, TESTSETS, EVAL_SOURCES
from core.data.ingest.source import subject_keys

_LV_ONLY = (2, 3)                                        # seg_lv reports myo + cav (no RV=1)


def _seen_keys(cfg, meta: pl.DataFrame) -> set[str] | None:
    """The real subjects the model SAW (train ∪ val) — the honest basis for the OOD/leak flag (a val
    subject IS seen; early-stopping touched it). Restricted to the model's OWN `sources`. None if
    unknowable. Reconstruction is a coded filter either way, no split_from_cfg:
      - new coded split (cfg.data.split): resolve -> train (if static) ∪ val Sources. A DYNAMIC train
        contributes no real subjects, so a synth-trained model's SEEN = its real val only.
      - old criteria: SEEN = labelled ∩ sources − test (test_datasets/test_vendors)."""
    if cfg is None:
        return None
    d = cfg.generator.data
    in_sources = meta.filter(pl.col("dataset").is_in(list(d.sources)))
    if getattr(d, "split", ""):
        from core.data.ingest.splits import resolve_cfg
        r = resolve_cfg(d, in_sources)
        seen = set(r.val.subjects())                            # val is always a real StaticSource
        if getattr(r.train, "kind", "static") == "static":
            seen |= set(r.train.subjects())
        return {f"{a}\t{b}" for a, b in seen}
    test = pl.col("dataset").is_in(list(d.test_datasets)) | pl.col("vendor").is_in(list(d.test_vendors))
    return subject_keys(in_sources.filter(pl.col("labelled") & ~test))


def score_matrix(model_refs: list[str], testset_names: list[str] | None = None,
                 tta: bool = True) -> list[dict]:
    """Score each model on each TestSet -> flat rows (model, testset, ood, dice/class, ef_mae, n). Each
    model is loaded once with its own preprocessing; the eval cloud is loaded per (model-preprocessing)
    so npz match the weights."""
    from core.registry import resolve as resolve_model
    from core.model import load_run
    from core.data.static import store
    from cardioseg.evaluation.validate import validate

    tsets = [TESTSETS[n] for n in testset_names] if testset_names else list(MATRIX_TESTSETS)
    rows: list[dict] = []
    for ref in model_refs:
        model, cfg, device = load_run(resolve_model(ref))
        d = cfg.generator.data if cfg else None
        size = d.size if d else 256
        meta = store.load_cfg(d, sources=EVAL_SOURCES) if d else store.load(EVAL_SOURCES)
        seen = _seen_keys(cfg, meta)
        for ts in tsets:
            src = ts.source(meta)                        # lock-checked; raises on drift
            paths, subj = src.paths(), set(src.subjects())
            ood = None if seen is None else {f"{a}\t{b}" for a, b in subj}.isdisjoint(seen)
            dice, ef_rows, _ = validate(model, paths, size, device, tta=tta)
            classes = _LV_ONLY if ts.task == "seg_lv" else tuple(dice)
            dmean = float(np.nanmean([dice[c] for c in classes]))
            ef_mae = (float(np.mean([abs(r["ef_gt"] - r["ef_pred"]) for r in ef_rows]))
                      if ef_rows else float("nan"))
            rows.append(dict(model=ref, testset=ts.name, task=ts.task, n=len(paths), ood=ood,
                             dice_mean=round(dmean, 4),
                             **{f"dice_{c}": round(dice[c], 4) for c in classes},
                             ef_mae=round(ef_mae, 2)))
    return rows


def _print(rows: list[dict]):
    print(f"\n=== generalization matrix ({len(rows)} cells) — OOD=honest, in-domain=leak ===")
    for r in rows:
        flag = "OOD " if r.get("ood") else ("LEAK" if r.get("ood") is False else "?   ")
        print(f"  [{flag}] {r['model']:>12} x {r['testset']:<18} n={r['n']:>3} "
              f"Dice {r['dice_mean']:.3f}  EF {r['ef_mae']:>5}%")


if __name__ == "__main__":
    import argparse
    from core.obs import setup
    setup()
    ap = argparse.ArgumentParser(description="cross-domain generalization matrix over frozen TestSets")
    ap.add_argument("--models", nargs="+", required=True, help="registry refs (alias|version|run-id)")
    ap.add_argument("--testsets", nargs="*", default=None,
                    help=f"TestSet names (default: the granular battery). known: {sorted(TESTSETS)}")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--out", default=None, help="write rows to this json")
    a = ap.parse_args()
    rows = score_matrix(a.models, a.testsets, tta=not a.no_tta)
    _print(rows)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {a.out}")
