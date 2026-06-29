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


# --- generator: derive reference ranges from the consolidated store (the "data is scarce -> trace
#     every number to a cohort" path). Computed from GROUND TRUTH, so model-independent + reproducible. ---

def _range_entry(vals, based_on: str) -> dict:
    """A provenance leaf for a derived [p5, p95] range. verified=true: it's reproducible from the
    cohort (not an unchecked external fact)."""
    import numpy as np
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    return {
        "value": [round(float(np.percentile(a, 5)), 1), round(float(np.percentile(a, 95)), 1)],
        "mean": round(float(a.mean()), 1), "n": int(a.size),
        "source": "computed", "based_on": based_on,
        "extracted_by": "computed", "verified": True,
    }


def build_from_store(sources=None, inplane: float | None = None, out_dir: str | Path | None = None) -> Path:
    """Derive EF / EDV / ESV reference ranges per pathology from the store's GROUND-TRUTH masks and
    write `<data>/reference/derived.yaml`. Each entry's `based_on` records the exact cohort (paramkey,
    pathology, n, datasets) so a scarce-data reference traces back to what produced it.

    Healthy cohort (pathology contains 'normal') is also surfaced as normal_ranges.* for consumers."""
    import polars as pl
    from core.config import DEFAULT_INPLANE
    from core.data import store
    from core.measure import ejection_fraction

    inplane = DEFAULT_INPLANE if inplane is None else inplane
    df = store.load(sources, inplane=inplane).filter(pl.col("labelled"))
    paramkey = store.param_key(inplane)
    # per-case EF/EDV/ESV from GT, grouped by pathology
    by_path: dict[str, dict[str, list]] = {}
    datasets: dict[str, set] = {}
    for r in df.iter_rows(named=True):
        c = store.load_arrays(r["path"])
        if "ed_gt" not in c or "es_gt" not in c:
            continue
        sp = tuple(float(s) for s in c["spacing"])
        ef, edv, esv = ejection_fraction(c["ed_gt"], c["es_gt"], sp)
        path = (r.get("pathology") or "unknown")
        g = by_path.setdefault(path, {"ef": [], "edv": [], "esv": []})
        g["ef"].append(ef); g["edv"].append(edv); g["esv"].append(esv)
        datasets.setdefault(path, set()).add(r.get("dataset"))

    out: dict = {"ef_by_pathology": {}, "volumes": {}}
    for path, g in sorted(by_path.items()):
        ds = sorted(d for d in datasets[path] if d)
        base = f"processed/{paramkey} GT, pathology={path}, n={len(g['ef'])}, datasets={ds}"
        out["ef_by_pathology"][path] = _range_entry(g["ef"], base)
        out["volumes"][f"{path}_edv_ml"] = _range_entry(g["edv"], base)
        out["volumes"][f"{path}_esv_ml"] = _range_entry(g["esv"], base)

    # healthy cohort -> normal_ranges.* (consumer convenience)
    healthy = [p for p in by_path if "normal" in p.lower() or p.lower() == "nor"]
    if healthy:
        p = healthy[0]
        base = f"processed/{paramkey} GT, healthy cohort='{p}', n={len(by_path[p]['ef'])}, datasets={sorted(d for d in datasets[p] if d)}"
        out["normal_ranges"] = {"ef_normal": _range_entry(by_path[p]["ef"], base),
                                "edv_normal_ml": _range_entry(by_path[p]["edv"], base),
                                "esv_normal_ml": _range_entry(by_path[p]["esv"], base)}

    base_dir = Path(out_dir) if out_dir is not None else reference_dir()
    base_dir.mkdir(parents=True, exist_ok=True)
    path_out = base_dir / "derived.yaml"
    header = ("# DERIVED reference ranges — generated by `python -m core.reference --build` from the\n"
              "# ground-truth masks in processed/. LOCAL (not committed). Regenerate after the store\n"
              "# changes. Each value: computed [p5,p95], verified=true, based_on = its cohort.\n")
    path_out.write_text(header + OmegaConf.to_yaml(OmegaConf.create(out)))
    return path_out


def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Derive reference ranges from processed/ ground truth.")
    ap.add_argument("--build", action="store_true", help="compute + write <data>/reference/derived.yaml")
    ap.add_argument("--sources", nargs="*", default=None)
    a = ap.parse_args()
    if a.build:
        from core.obs import setup
        setup()
        p = build_from_store(sources=a.sources)
        print(f"wrote {p}")
        ref = Reference()
        ef = ref.provenance("normal_ranges", "ef_normal")
        if ef:
            print(f"  normal EF (p5-p95): {ef['value']}%  (n={ef['n']}, {ef['based_on']})")
    else:
        ap.print_help()


if __name__ == "__main__":
    _main()
