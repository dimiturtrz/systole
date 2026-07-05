"""Split families — coded filters, versioned, content-hash-frozen test.

A split is CODE, not data: each component is a `(cloud) -> Source` callable (a polars predicate for
static, a generator spec for dynamic). No named-manifest strings, no enum: the filter IS the
definition. A family (StaticMain, SynthMain, …) owns the LOGIC once; its `versions` dict holds only
the per-version INGREDIENTS (the callables + the test lock). Same logic, new lists -> new version
entry; different LOGIC -> a new family class (never a branch inside).

TEST is frozen for comparability by a CONTENT HASH (`test_lock`), not a name: resolve the test filter,
hash the resulting ids, assert it equals the lock. Data grew and the set changed -> hash mismatch ->
hard error (bump the version), never a silent drift. train defaults to the labelled COMPLEMENT of
test+val (you never enumerate train).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import polars as pl

from core.data.source import Source, StaticSource

CloudFn = Callable[[pl.DataFrame], Source]


@dataclass(frozen=True)
class SplitDef:
    """One version's ingredients. test/val/train = coded (cloud)->Source. train=None -> complement.
    test_lock = sha256 of the resolved test ids (drift guard); "" until first freeze."""
    test: CloudFn
    val: CloudFn
    train: CloudFn | None = None
    test_lock: str = ""


class Split(Protocol):
    name: str
    versions: dict[str, SplitDef]


@dataclass(frozen=True)
class Resolution:
    train: Source
    val: Source
    test: Source
    version: str
    test_hash: str


def _latest(versions: dict[str, SplitDef]) -> str:
    return max(versions, key=lambda v: tuple(int(x) for x in v.split(".")))


def _complement(cloud: pl.DataFrame, exclude: list[Source]) -> StaticSource:
    """Labelled rows not in any excluded source (the implicit train). Static only."""
    used: set[str] = set()
    for s in exclude:
        used |= {f"{d}\t{sid}" for d, sid in s.subjects()}      # type: ignore[attr-defined]
    k = pl.col("dataset") + "\t" + pl.col("subject_id").cast(pl.Utf8)
    keep = cloud.filter(pl.col("labelled") & ~k.is_in(list(used)))
    return StaticSource(keep, "complement (labelled rest)")


def resolve(family: Split, cloud: pl.DataFrame, version: str | None = None) -> Resolution:
    """Build (train, val, test) Sources for a family@version over `cloud`. Enforces the test lock:
    a drifted test set raises rather than silently rescoring on different subjects."""
    v = version or _latest(family.versions)
    d = family.versions[v]
    test, val = d.test(cloud), d.val(cloud)
    train = d.train(cloud) if d.train is not None else _complement(cloud, [test, val])
    h = test.ids_hash()                                          # type: ignore[attr-defined]
    if d.test_lock and d.test_lock != h:
        raise ValueError(f"{family.name}@{v}: test drifted (lock {d.test_lock[:19]}… != {h[:19]}…) — "
                         f"the store changed the test set; freeze a new version, don't mutate this one")
    return Resolution(train, val, test, v, h)
