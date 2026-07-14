"""Persist per-dataset acquisition meta -> <data>/raw/<ds>/meta/<ds>.yaml (source-only; the yaml is a
regenerable cache, out-of-repo). Merges the AUTO tier (adapter.meta() parsing the shipped sidecars)
with the paper-cited overlay (sources.yaml), tagging each field {value, source, by, verified}.

Reproducible by design: re-running regenerates the bulk from the sidecars deterministically; only the
small paper-cited layer is hand-curated (and visibly flagged verified / unverified).

    python -m cardioseg.preprocessing.normalization.persist --dataset acdc
    python -m cardioseg.preprocessing.normalization.persist            # all
"""
import logging
from pathlib import Path

from omegaconf import OmegaConf

from core.config import Config
from core.data.static.mri.registry import SEG_DATASETS, AdapterRegistry

log = logging.getLogger("cardioseg.persist")

_DATASETS = SEG_DATASETS
_SOURCES = Path(__file__).parent / "sources.yaml"


class Persist:
    """Dataset acquisition-meta persistence: merge the adapter AUTO tier with the paper overlay and
    serialize provenance-tagged fields (the free helpers folded in as staticmethods)."""

    @staticmethod
    def _overlay() -> dict:
        return OmegaConf.to_container(OmegaConf.load(_SOURCES)) if _SOURCES.exists() else {}

    @staticmethod
    def _prov(field: str, auto_src: dict, paper: dict) -> dict:
        """Provenance for one field: the paper overlay wins (cited + verified flag), else the adapter's
        _source map (parsed = deterministic = verified)."""
        p = paper.get(field)
        if isinstance(p, dict):
            return {"source": p.get("source", "paper"), "by": "paper", "verified": bool(p.get("verified"))}
        src = auto_src.get(field) or auto_src.get("rest") or auto_src.get("all") or "unknown"
        return {"source": src, "by": "auto", "verified": True}

    @staticmethod
    def persist_meta(dataset: str) -> Path:
        a = AdapterRegistry.get_adapter(dataset)
        paper = Persist._overlay().get(dataset) or {}
        subjects = {}
        for case in a.cases():
            m = dict(a.meta(case))
            auto_src = m.pop("_source", {}) or {}
            subjects[case.name] = {k: {"value": v, **Persist._prov(k, auto_src, paper)} for k, v in m.items()}
        out = Path(Config.data_root("raw")) / dataset / "meta" / f"{dataset}.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create({"dataset": dataset, "n": len(subjects),
                                         "paper_layer": paper, "subjects": subjects}), out)
        log.info(f"{dataset}: {len(subjects)} subjects -> {out}")
        return out

    @staticmethod
    def add_args(ap):
        ap.add_argument("--dataset", default="all", choices=("all", *_DATASETS))

    @staticmethod
    def run(args):
        for ds in (_DATASETS if args.dataset == "all" else [args.dataset]):
            Persist.persist_meta(ds)
