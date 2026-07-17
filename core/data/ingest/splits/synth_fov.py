"""SynthFov — the whole-FOV MRXCAT arm (bd hpy.3): same zero-real goalpost as synth_main, but the
anatomy source is the SSM x MRXCAT composite pool (our Rodero hearts placed in XCAT whole-FOV torso
context) painted with the FovBg strategy (each surrounding organ its own physical bSSFP tissue), not
a procedural organ field. The one untested MRXCAT lever: does STRUCTURED whole-FOV context beat the
statistically-matched procedural bg (whose TORSO_BG fractions were already derived from this phantom)?

val + test are identical to synth_main so the number is directly comparable to the 0.61 zero-real arm.
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import polars as pl

from core.config import Config
from core.data.dynamic.source import DynamicSource
from core.data.dynamic.synth import MrxcatBgCfg
from core.data.ingest.source import StaticSource
from core.data.ingest.split import SplitDef
from core.data.ingest.testsets import SYNTH_MAIN_TEST
from core.data.static.mri.base import Dataset

V = pl.col

POOL = "pool_ssmfov"       # SSM x MRXCAT composite (our hearts in XCAT whole-FOV context), 7-class FOV maps


class SynthFov:
    name = "synth_fov"
    sources = ()

    @staticmethod
    def pool(name: str) -> str:
        """Resolve a built MRXCAT pool by name under the processed MRI root (built offline by
        `python -m core.data mrxcat build-ssm-fov-pool`)."""
        return str(Path(Config.data_root("processed")) / "mrxcat" / f"{name}.npz")

    versions: ClassVar[dict[str, SplitDef]] = {
        "1.0.0": SplitDef(
            train=lambda c: DynamicSource(pool=SynthFov.pool(POOL), bg=MrxcatBgCfg(),
                                          note=f"SSMxMRXCAT {POOL}, zero-real whole-FOV bg"),
            val=lambda c: StaticSource(c.filter(V("labelled") & (V("dataset") == Dataset.ACDC)),
                                       "ACDC real val (held from test)"),
            test=lambda c: SYNTH_MAIN_TEST.source(c),  # noqa: PLW0108
        ),
    }
