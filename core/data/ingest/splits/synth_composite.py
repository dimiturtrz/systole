"""SynthComposite — the composite-source arm: train on a UNION of synth SOURCES, each a clean single-
origin generation node with its OWN painter, composed via CompositeSource (NOT a frankenstein mega-pool).

Ingredients: the Rodero SSM pool (healthy anatomy manifold) + the label-space pathology pool (DCM/HCM/
abnormal-RV deforms). Data-space justification (bd uy4d, no retrain): SSM-alone shape coverage 0.78 ->
SSM+pathology 0.94 — the pathology source fills the 22% tail SSM undercovers. Same real val/test as
synth_main (ACDC val held out; SYNTH_MAIN_TEST locked). (bd cumw/uch6)
"""
from __future__ import annotations

from typing import ClassVar

import polars as pl

from core.data.dynamic.source import CompositeSource, DynamicSource
from core.data.dynamic.synth import ProceduralBgCfg
from core.data.ingest.source import StaticSource
from core.data.ingest.split import SplitDef
from core.data.ingest.splits.synth_main import SynthMain
from core.data.ingest.testsets import SYNTH_MAIN_TEST
from core.data.static.mri.base import Dataset

V = pl.col

# The composite's ingredient SOURCES — each a distinct generation node unioned behind the batch() seam
# (CompositeGenerator), NOT concatenated into one pool. SSM = healthy manifold; PATHOLOGY = the DCM/HCM/RV
# deforms that fill the tail SSM misses. Both painted zero-real with the procedural bg (each keeps its own).
_SSM = "pool_1000_bal"
_PATHOLOGY = "pool_pathology"

# GPU-resident VRAM budget: synth_main preloads ~10k slices (pool_1000_bal) on the 32 GB card and fits;
# the FULL 42k composite union does NOT (synth_main.py: "the 42k-slice composite pool does not"). Cap each
# source so the union stays ~synth_main's resident size — which also SIZE-MATCHES synth_main for a clean
# A/B (isolates the effect of pathology SHAPES, not 'more data'). Budget split evenly across the sources.
_RESIDENT_BUDGET = 10000
_CAP = _RESIDENT_BUDGET // 2            # per-source slice cap (SSM + pathology)


class SynthComposite:
    name = "synth_composite"
    sources = ()                        # synth train adds no real store sources

    @staticmethod
    def _synth_source(pool: str, note: str) -> DynamicSource:
        return DynamicSource(pool=SynthMain.pool(pool), bg=ProceduralBgCfg(), note=note, cap=_CAP)

    versions: ClassVar[dict[str, SplitDef]] = {
        "1.0.0": SplitDef(
            train=lambda c: CompositeSource(
                [SynthComposite._synth_source(_SSM, "Rodero SSM (healthy manifold)"),
                 SynthComposite._synth_source(_PATHOLOGY, "label-space pathology (DCM/HCM/RV tail)")],
                note="SSM + pathology (source union, per-source painter)"),
            val=lambda c: StaticSource(c.filter(V("labelled") & (V("dataset") == Dataset.ACDC)),
                                       "ACDC real val (held from test)"),
            # all seg real minus ACDC val (locked); lambda defers the global lookup (testset swap)
            test=lambda c: SYNTH_MAIN_TEST.source(c),  # noqa: PLW0108
        ),
    }
