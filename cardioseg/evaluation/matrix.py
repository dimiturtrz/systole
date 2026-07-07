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

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from cardioseg.evaluation.validate import EvalCfg, Evaluator
from core import registry as registry_mod
from core import run as run_mod
from core.data.ingest.testsets import EVAL_SOURCES, MATRIX_TESTSETS, TESTSETS
from core.data.static import splits, store
from core.obs import setup

log = logging.getLogger("cardioseg.matrix")

_LV_ONLY = (2, 3)                                        # seg_lv reports myo + cav (no RV=1)


def score_matrix(model_refs: list[str], testset_names: list[str] | None = None,
                 *, tta: bool = True) -> list[dict]:
    """Score each model on each TestSet -> flat rows (model, testset, ood, dice/class, ef_mae, n). Each
    model is loaded once with its own preprocessing; the eval cloud is loaded per (model-preprocessing)
    so npz match the weights."""
    tsets = [TESTSETS[n] for n in testset_names] if testset_names else list(MATRIX_TESTSETS)
    rows: list[dict] = []
    for ref in model_refs:
        model, cfg, device = run_mod.load_run(registry_mod.resolve(ref))
        d = cfg.generator.data if cfg else None
        size = d.size if d else 256
        meta = store.load_cfg(d, sources=EVAL_SOURCES) if d else store.load(EVAL_SOURCES)
        seen = splits.seen_keys(d, meta) if d else None      # subjects the model saw (train∪val); None = unknowable
        for ts in tsets:
            src = ts.source(meta)                        # lock-checked; raises on drift
            paths, subj = src.paths(), set(src.subjects())
            ood = None if seen is None else {f"{a}\t{b}" for a, b in subj}.isdisjoint(seen)
            dice, ef_rows, _ = Evaluator(model, device, EvalCfg(size=size, tta=tta)).validate(paths)
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
    log.info(f"\n=== generalization matrix ({len(rows)} cells) — OOD=honest, in-domain=leak ===")
    for r in rows:
        flag = "OOD " if r.get("ood") else ("LEAK" if r.get("ood") is False else "?   ")
        log.info(f"  [{flag}] {r['model']:>12} x {r['testset']:<18} n={r['n']:>3} "
              f"Dice {r['dice_mean']:.3f}  EF {r['ef_mae']:>5}%")


if __name__ == "__main__":
    setup()
    ap = argparse.ArgumentParser(description="cross-domain generalization matrix over frozen TestSets")
    ap.add_argument("--models", nargs="+", required=True, help="registry refs (alias|version|run-id)")
    ap.add_argument("--testsets", nargs="*", default=None,
                    help=f"TestSet names (default: the granular battery). known: {sorted(TESTSETS)}")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--out", default=None, help="write rows to this json")
    args = ap.parse_args()
    rows = score_matrix(args.models, args.testsets, tta=not args.no_tta)
    _print(rows)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        log.info(f"\nwrote {args.out}")
