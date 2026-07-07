"""SynthMain — the magnum-opus arm: train on synthetic data, test on real.

train = DynamicSource (Rodero anatomy pool, zero real pixels — painted by the Generator from physical
bounds). test = (near) all real, held-out entirely. val = ACDC real, held from test — so the honest
claim here is "zero real in TRAIN". Making val synthetic too (the full "zero real anywhere in training"
claim) is cardiac-seg-6rd7; until then val is real and excluded from test.

test_lock pins the real test set by content hash (labelled real minus the ACDC val).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from core.config import data_root
from core.data.dynamic.source import DynamicSource
from core.data.ingest.source import StaticSource
from core.data.ingest.split import SplitDef
from core.data.ingest.testsets import SYNTH_MAIN_TEST

V = pl.col


# This split VERSION's synth ingredients — argued, not magic (bump the version to change either):
#   POOL: the balanced 1000-slice Rodero SSM pool. BALANCED = cleaner label distribution than the raw
#         cohort (bd generation-ceiling); 1000 slices fit the VRAM-resident preload — the 42k-slice
#         composite pool does not.
#   BG:   'procedural' = whole-FOV synthetic organ field. The zero-real-input painter: it clears the
#         flat-background 0.07 Dice wall (bd bwp) when there is NO real image to seed the background.
POOL = "pool_1000_bal"
ZERO_REAL_BG = "procedural"


def _pool(name: str) -> str:
    """Resolve a built anatomy pool by name under the meshes root (paths.yaml `meshes:` key; the
    Rodero SSM anatomy lives beside the MRI root, not under it). No machine path in code."""
    return str(Path(data_root("meshes")) / "processed" / "rodero_anatomy" / f"{name}.npz")


class SynthMain:
    name = "synth_main"
    sources = ()                        # synth train adds no real store sources
    versions = {
        "1.0.0": SplitDef(
            train=lambda c: DynamicSource(pool=_pool(POOL), bg_mode=ZERO_REAL_BG,
                                          note=f"Rodero {POOL}, zero-real {ZERO_REAL_BG} bg"),
            val=lambda c: StaticSource(c.filter(V("labelled") & (V("dataset") == "acdc")),
                                       "ACDC real val (held from test; synth val = 6rd7)"),
            test=lambda c: SYNTH_MAIN_TEST.source(c),            # all seg real minus ACDC val (642, locked)
        ),
    }
