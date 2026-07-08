"""SynthComposite — the composite-source arm: train on a UNION of synth SOURCES, each a clean single-
origin generation node with its OWN painter, composed via CompositeSource (NOT a frankenstein mega-pool).

Ingredients: the Rodero SSM pool (healthy anatomy manifold) + the label-space pathology pool (DCM/HCM/
abnormal-RV deforms). Data-space justification (bd uy4d, no retrain): SSM-alone shape coverage 0.78 ->
SSM+pathology 0.94 — the pathology source fills the 22% tail SSM undercovers. Same real val/test as
synth_main (ACDC val held out; SYNTH_MAIN_TEST locked). (bd cumw/uch6)
"""
from __future__ import annotations

import polars as pl

from core.data.dynamic.source import CompositeSource, DynamicSource
from core.data.dynamic.synth import ProceduralBgCfg
from core.data.ingest.source import StaticSource
from core.data.ingest.split import SplitDef
from core.data.ingest.splits.synth_main import _pool
from core.data.ingest.testsets import SYNTH_MAIN_TEST

V = pl.col

# The composite's ingredient SOURCES — each a distinct generation node unioned behind the batch() seam
# (CompositeGenerator), NOT concatenated into one pool. SSM = healthy manifold; PATHOLOGY = the DCM/HCM/RV
# deforms that fill the tail SSM misses. Both painted zero-real with the procedural bg (each keeps its own).
_SSM = "pool_1000_bal"
_PATHOLOGY = "pool_pathology"


def _synth_source(pool: str, note: str) -> DynamicSource:
    return DynamicSource(pool=_pool(pool), bg=ProceduralBgCfg(), note=note)


class SynthComposite:
    name = "synth_composite"
    sources = ()                        # synth train adds no real store sources
    versions = {
        "1.0.0": SplitDef(
            train=lambda c: CompositeSource(
                [_synth_source(_SSM, "Rodero SSM (healthy manifold)"),
                 _synth_source(_PATHOLOGY, "label-space pathology (DCM/HCM/RV tail)")],
                note="SSM + pathology (source union, per-source painter)"),
            val=lambda c: StaticSource(c.filter(V("labelled") & (V("dataset") == "acdc")),
                                       "ACDC real val (held from test)"),
            test=lambda c: SYNTH_MAIN_TEST.source(c),            # all seg real minus ACDC val (locked)
        ),
    }
