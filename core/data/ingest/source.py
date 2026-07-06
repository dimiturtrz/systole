"""A data Source — the ingestion contract between a split and the training pipe.

A split component (train / val / test) is `(cloud) -> Source`. The Source is what the pipe consumes,
blind to origin: STATIC (real npz selected by a coded filter over the processed cloud) or DYNAMIC
(synth generated from parameter bounds, optionally seeded by real data — lives in data/dynamic). This
module holds the Protocol + the static impl; DynamicSource lives with the generator (data/dynamic).

The Source's identity for lineage/comparability is a CONTENT HASH of its subjects (sorted
`dataset\\tsubject_id`), NOT a human name — so a frozen test set is anchored by checksum, and drift
(the store grew, the filter now matches more) is a hash mismatch, never a silent change.
"""
from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import polars as pl


def ids_hash(subjects: list[tuple[str, str]]) -> str:
    """sha256 of the sorted (dataset, subject_id) set — the freeze anchor / drift guard."""
    blob = "\n".join(f"{d}\t{s}" for d, s in sorted(subjects))
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


@runtime_checkable
class Source(Protocol):
    """Yields samples to the pipe. Static exposes a fixed subject set (paths); dynamic streams from a
    generator. Both carry `provenance()` (the lineage link) and, where finite, an `ids_hash()`."""

    def provenance(self) -> dict: ...


class StaticSource:
    """Real npz selected by a coded filter over the processed cloud (a polars frame with `path`,
    `dataset`, `subject_id`). Finite: exposes `paths()` for the training/eval consumers and an
    `ids_hash()` that pins exactly which subjects it is."""

    kind = "static"

    def __init__(self, frame: pl.DataFrame, note: str = ""):
        self._f = frame
        self._note = note

    @property
    def frame(self) -> pl.DataFrame:
        """The selected rows — for consumers that still work on the polars frame (preload, counts)."""
        return self._f

    def paths(self) -> list[str]:
        return self._f.get_column("path").to_list()

    def subjects(self) -> list[tuple[str, str]]:
        return list(zip(self._f.get_column("dataset").to_list(),
                        (str(s) for s in self._f.get_column("subject_id").to_list())))

    def ids_hash(self) -> str:
        return ids_hash(self.subjects())

    def resident(self, size: int, device: str):
        """Raw resident (X [N,1,H,W], Y [N,H,W]) — real slices, no transforms. Used for val/test scoring
        and as a train_gen's seed."""
        from core.data.dynamic.dataset import load_to_gpu
        return load_to_gpu(self.paths(), size, device)

    def train_gen(self, size: int, device: str, gen_cfg, n_classes: int):
        """The source's own batch engine (owns its resident tensors + transform chain). Static = real
        pixels + the configured aug/DR-synth (gen_cfg.synth as-is: synth_p>0 -> physics-recontrast DR on
        real labels; the flagship recipe). No force_synth (real rows are never fully replaced)."""
        from core.data.dynamic.generator import Generator
        X, Y = self.resident(size, device)
        return Generator(gen_cfg, X, Y, n_classes, device, force_synth=None)

    def __len__(self) -> int:
        return self._f.height

    def provenance(self) -> dict:
        return {"kind": self.kind, "n": len(self), "note": self._note, "ids_hash": self.ids_hash()}
