"""Cross-domain generalization matrix — score registered models on frozen test manifests.

Pure inference, no retrain: this is what the frozen manifests buy. Any model, any test set, forever
comparable (an old model and a new one score on the SAME pinned subjects). For each (model, manifest)
cell: resolve the model (registry) with its OWN preprocessing, resolve the manifest to current npz
paths, validate. A cell is OOD (the honest generalization number) when NONE of the manifest's
subjects were in the model's TRAIN — computed exactly by reconstructing the model's train subjects
from its saved DataCfg (split_from_cfg over the current store); otherwise it's in-domain (a leak) and
flagged, never silently mixed with OOD cells.

Task handling by manifest tag: seg4 -> all three classes; seg_lv (SCD, no RV in GT) -> myo+cav only;
ef (Kaggle, no seg npz) -> out of scope here (separate EF-at-scale path, load_sax -> seg->EF).

    python -m cardioseg.evaluation.matrix --models production 55 56 --manifests vendor_canon vendor_ge
    python -m cardioseg.evaluation.matrix --models production            # all seg manifests
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

# every dataset any seg manifest can reference (incl. SCD, which is NOT in SOURCE_DATASETS)
_ALL_SOURCES = ["acdc", "mnm2", "mnms1", "cmrxmotion", "scd"]
_SEG_TASKS = ("seg4", "seg_lv")                          # ef manifests are scored elsewhere
_LV_ONLY = (2, 3)                                        # seg_lv reports myo + cav (no RV=1)


def _key(df: pl.DataFrame) -> set[str]:
    return set((df["dataset"] + "\t" + df["subject_id"].cast(pl.Utf8)).to_list())


def _train_keys(cfg, meta: pl.DataFrame) -> set[str] | None:
    """The model's TRAIN subject keys (for the leak/OOD flag), reconstructed from its DataCfg over the
    CURRENT store. None if the model carried no config (can't know its split). Restricted to the
    model's OWN `sources` — else a dataset the model never loaded (e.g. SCD, absent from its sources)
    would wrongly appear in the reconstructed train and mis-flag an honest OOD cell as a leak."""
    if cfg is None:
        return None
    from core.data.static.splits import split_from_cfg
    d = cfg.generator.data
    in_sources = meta.filter(pl.col("dataset").is_in(list(d.sources)))
    train, _, _, _ = split_from_cfg(d, in_sources, seed=cfg.seed)
    return _key(train)


def score_matrix(model_refs: list[str], manifest_names: list[str] | None = None,
                 tta: bool = True) -> list[dict]:
    """Score each model on each seg manifest -> flat rows (model, manifest, ood, dice/class, ef_mae, n).
    Each model is loaded once with its own preprocessing; the store is loaded per (model-preprocessing)
    so npz match the weights."""
    from core.registry import resolve
    from core.model import load_run
    from core.data.static import store, manifest as mf
    from cardioseg.evaluation.validate import validate

    names = manifest_names or [n for n in mf.list_manifests()
                               if mf.load(n)["task"] in _SEG_TASKS]
    manifests = [(n, mf.load(n)) for n in names]
    rows: list[dict] = []
    for ref in model_refs:
        model, cfg, device = load_run(resolve(ref))
        d = cfg.generator.data if cfg else None
        size = d.size if d else 256
        # store with THIS model's preprocessing so npz paths match the weights it trained under
        meta = store.load(_ALL_SOURCES, inplane=(d.inplane if d else store.TARGET_INPLANE),
                          n4=(d.n4 if d else False), nyul=(d.nyul if d else False),
                          norm=(d.norm if d else "zscore"))
        train_keys = _train_keys(cfg, meta)
        for name, m in manifests:
            paths, missing = mf.resolve_paths(m, meta)
            if not paths:
                rows.append(dict(model=ref, manifest=name, task=m["task"], n=0, missing=len(missing),
                                 ood=None, note="unresolved (dataset not in store?)"))
                continue
            msubj = {f"{a}\t{b}" for a, b in m["subjects"]}
            ood = None if train_keys is None else msubj.isdisjoint(train_keys)
            dice, ef_rows, _ = validate(model, paths, size, device, tta=tta)
            classes = _LV_ONLY if m["task"] == "seg_lv" else tuple(dice)
            dmean = float(np.nanmean([dice[c] for c in classes]))
            ef_mae = (float(np.mean([abs(r["ef_gt"] - r["ef_pred"]) for r in ef_rows]))
                      if ef_rows else float("nan"))
            rows.append(dict(model=ref, manifest=name, task=m["task"], n=len(paths),
                             missing=len(missing), ood=ood, dice_mean=round(dmean, 4),
                             **{f"dice_{c}": round(dice[c], 4) for c in classes},
                             ef_mae=round(ef_mae, 2)))
    return rows


def _print(rows: list[dict]):
    print(f"\n=== generalization matrix ({len(rows)} cells) — OOD=honest, in-domain=leak ===")
    for r in rows:
        flag = "OOD " if r.get("ood") else ("LEAK" if r.get("ood") is False else "?   ")
        drift = f" drift={r['missing']}" if r.get("missing") else ""
        if r.get("n"):
            print(f"  [{flag}] {r['model']:>12} x {r['manifest']:<20} n={r['n']:>3} "
                  f"Dice {r['dice_mean']:.3f}  EF {r['ef_mae']:>5}%{drift}")
        else:
            print(f"  [----] {r['model']:>12} x {r['manifest']:<20} {r.get('note','')}{drift}")


if __name__ == "__main__":
    import argparse
    from core.obs import setup
    setup()
    ap = argparse.ArgumentParser(description="cross-domain generalization matrix over frozen manifests")
    ap.add_argument("--models", nargs="+", required=True, help="registry refs (alias|version|run-id)")
    ap.add_argument("--manifests", nargs="*", default=None, help="manifest names (default: all seg manifests)")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--out", default=None, help="write rows to this json")
    a = ap.parse_args()
    rows = score_matrix(a.models, a.manifests, tta=not a.no_tta)
    _print(rows)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {a.out}")
