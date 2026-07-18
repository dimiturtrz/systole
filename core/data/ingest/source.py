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
from typing import Any, Protocol, runtime_checkable

import polars as pl
import torch
from jaxtyping import Integer

from core.data.dynamic import dataset as _dataset
from core.data.dynamic.generator import Generator
from core.data.static.labels import Labels
from core.types import shapecheck


class SubjectIds:
    """The subject-identity primitives (free funcs folded in as staticmethods): the content-hash freeze
    anchor and the '{dataset}\\t{subject_id}' key set — the lineage/leak-check surface shared by the
    Source impls and the split builders."""

    @staticmethod
    def ids_hash(subjects: list[tuple[str, str]]) -> str:
        """sha256 of the sorted (dataset, subject_id) set — the freeze anchor / drift guard."""
        blob = "\n".join(f"{d}\t{s}" for d, s in sorted(subjects))
        return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()

    @staticmethod
    def subject_keys(df: pl.DataFrame) -> set[str]:
        """The '{dataset}\\t{subject_id}' identity-key set for a frame — for join/leak/dedup checks."""
        return set((df["dataset"] + "\t" + df["subject_id"].cast(pl.Utf8)).to_list())


@runtime_checkable
class Source(Protocol):
    """Yields samples to the pipe. Static exposes a fixed subject set (paths); dynamic streams from a
    generator. Both carry `provenance()` (the lineage link) and, where finite, an `ids_hash()`."""

    def provenance(self) -> dict[str, Any]: ...


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
                        (str(s) for s in self._f.get_column("subject_id").to_list()), strict=True))

    def ids_hash(self) -> str:
        return SubjectIds.ids_hash(self.subjects())

    def resident(self, size: int, device: str):
        """Raw resident (X [N,1,H,W], Y [N,H,W]) — real slices, no transforms. Used for val/test scoring
        and as a train_gen's seed."""
        return _dataset.ACDCSliceDataset.load_to_gpu(self.paths(), size, device)

    def train_gen(self, size: int, device: str, gen_cfg: Any, n_classes: int) -> Generator:
        """The source's own batch engine (owns its resident tensors + transform chain). Static = real
        pixels + the configured aug/DR-synth (gen_cfg.synth as-is: synth_p>0 -> physics-recontrast DR on
        real labels; the flagship recipe). No force_synth. Carries a per-slice partial-label mask when
        the source mixes datasets that annotate different classes (e.g. SCD = LV-only)."""
        X, Y, owners = _dataset.ACDCSliceDataset.load_to_gpu(self.paths(), size, device, return_owners=True)
        return Generator(gen_cfg, X, Y, n_classes, device, force_synth=None,
                         valid=self._valid_mask(owners, n_classes, device))

    @shapecheck
    def _valid_mask(self, owners: Integer[torch.Tensor, "*n"], n_classes: int, device: str):
        """Per-slice class-validity [N,C] bool (labels.valid_row of each slice's dataset). None if every
        slice is full-label — keeps the full-label path mask-free (loss stays on the standard recipe)."""
        ds = self._f.get_column("dataset").to_list()            # per-path dataset, aligned to paths()
        rows = [Labels.valid_row(ds[o], n_classes) for o in owners.tolist()]
        if all(all(r) for r in rows):
            return None
        return torch.tensor(rows, dtype=torch.bool, device=device)

    def __len__(self) -> int:
        return self._f.height

    def provenance(self) -> dict[str, Any]:
        return {"kind": self.kind, "n": len(self), "note": self._note, "ids_hash": self.ids_hash()}
