"""Reference store loader — clinical/derived reference values (normal ranges, BSA→volume,
conventions) kept LOCAL at `<data>/reference/` (lives with the data, outside the repo), with the
loader + schema committed here (source lives with source).

Two honesty rules, matching the acquisition tier (normalization/sources.yaml):
  - every value carries provenance: {value, source, based_on, extracted_by, verified}
      source       human cite (paper/DOI) or "computed"
      based_on     lineage — which processed run/cohort it was derived from, or which paper table
      extracted_by computed | paper | llm
      verified     bool
  - STRICT: a `verified: false` value is never used — the loader skips it, so an unverified
    (e.g. LLM-extracted) number can sit in the file visibly without silently affecting anything.

Graceful fallback: reference/ present → use the known constants; absent / value missing / unverified
→ `get()` returns the caller's default, i.e. the general per-scan path. Absence IS "unknown" — no
special row, the default is the fallback. So a fresh clone with no reference/ just works.

Data is scarce here, so reference values are ideally DERIVED from `processed/` aggregates (with
`based_on` recording the cohort) rather than hand-typed; papers fill what can't be derived.
"""
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from core.config import data_root

_PROV_KEYS = {"value", "source", "based_on", "extracted_by", "verified"}


def reference_dir() -> Path:
    """`<data>/reference/` — sibling of raw/ and processed/ (data_root('reference'))."""
    return Path(data_root("reference"))


def _is_prov(node) -> bool:
    """A leaf provenance entry = a mapping with a 'value' key (plus the provenance fields)."""
    return isinstance(node, dict) and "value" in node


class Reference:
    """Loaded reference store. `get('a', 'b')` walks nested keys and returns the value only if the
    leaf is present AND (strict) verified; otherwise `default`. Missing store → every get() is the
    default, so callers fall back to the per-scan path with no branching."""

    def __init__(self, strict: bool = True, root: str | Path | None = None):
        self.strict = strict
        self._d: dict = {}
        base = Path(root) if root is not None else reference_dir()
        if base.is_dir():
            # Merge every reference/*.yaml at the top level — files are just organization
            # (reference.yaml / acquisition.yaml / conventions.yaml); content keys are the namespace,
            # so consumers query by category (get('normal_ranges', ...)), not by filename.
            for f in sorted(base.glob("*.yaml")):
                self._d.update(OmegaConf.to_container(OmegaConf.load(f), resolve=True) or {})

    def present(self) -> bool:
        """Whether any reference file was loaded (else everything falls back)."""
        return bool(self._d)

    def get(self, *keys, default=None):
        """Value at nested `keys` if present + (strict) verified, else `default`.
        e.g. ref.get('normal_ranges', 'ef_normal')."""
        node = self._d
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        if not _is_prov(node):
            return default                                   # not a leaf value
        if self.strict and not node.get("verified", False):
            return default                                   # unverified -> never used
        return node.get("value", default)

    def provenance(self, *keys) -> dict | None:
        """The full {value, source, based_on, ...} entry at `keys` (None if absent) — for the
        model card / audit, regardless of verified."""
        node = self._d
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return None
            node = node[k]
        return node if _is_prov(node) else None
