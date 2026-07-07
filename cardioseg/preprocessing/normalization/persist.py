"""Persist per-dataset acquisition meta -> <data>/raw/<ds>/meta/<ds>.yaml (source-only; the yaml is a
regenerable cache, out-of-repo). Merges the AUTO tier (adapter.meta() parsing the shipped sidecars)
with the paper-cited overlay (sources.yaml), tagging each field {value, source, by, verified}.

Reproducible by design: re-running regenerates the bulk from the sidecars deterministically; only the
small paper-cited layer is hand-curated (and visibly flagged verified / unverified).

    python -m cardioseg.preprocessing.normalization.persist --dataset acdc
    python -m cardioseg.preprocessing.normalization.persist            # all
"""
import argparse
import logging
from pathlib import Path

from omegaconf import OmegaConf

from core.config import KNOWN_DATASETS, data_root
from core.data.static.mri.registry import get_adapter
from core.obs import setup

log = logging.getLogger("cardioseg.persist")

_DATASETS = KNOWN_DATASETS
_SOURCES = Path(__file__).parent / "sources.yaml"


def _overlay() -> dict:
    return OmegaConf.to_container(OmegaConf.load(_SOURCES)) if _SOURCES.exists() else {}


def _prov(field: str, auto_src: dict, paper: dict) -> dict:
    """Provenance for one field: the paper overlay wins (cited + verified flag), else the adapter's
    _source map (parsed = deterministic = verified)."""
    p = paper.get(field)
    if isinstance(p, dict):
        return {"source": p.get("source", "paper"), "by": "paper", "verified": bool(p.get("verified"))}
    src = auto_src.get(field) or auto_src.get("rest") or auto_src.get("all") or "unknown"
    return {"source": src, "by": "auto", "verified": True}


def persist_meta(dataset: str) -> Path:
    a = get_adapter(dataset)
    paper = _overlay().get(dataset) or {}
    subjects = {}
    for case in a.cases():
        m = dict(a.meta(case))
        auto_src = m.pop("_source", {}) or {}
        subjects[case.name] = {k: {"value": v, **_prov(k, auto_src, paper)} for k, v in m.items()}
    out = Path(data_root("raw")) / dataset / "meta" / f"{dataset}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create({"dataset": dataset, "n": len(subjects),
                                     "paper_layer": paper, "subjects": subjects}), out)
    log.info(f"{dataset}: {len(subjects)} subjects -> {out}")
    return out


def load_meta(dataset: str):
    """Read the persisted meta yaml; None if absent (caller falls back to per-scan parsing)."""
    p = Path(data_root("raw")) / dataset / "meta" / f"{dataset}.yaml"
    return OmegaConf.to_container(OmegaConf.load(p)) if p.exists() else None


def main():
    setup()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="all", choices=("all",) + _DATASETS)
    a = ap.parse_args()
    for ds in (_DATASETS if a.dataset == "all" else [a.dataset]):
        persist_meta(ds)


if __name__ == "__main__":
    main()
