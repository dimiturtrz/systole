"""Tensor-shape enforcement decorators (bd cardiac-seg-zwno).

`@shapecheck` on a tensor-boundary function makes its jaxtyping annotations (`Float[Tensor, "b c h w"]`)
LIVE: a wrong-shape / wrong-dtype array raises at the call instead of silently broadcasting. The runtime
checker is beartype, a dev/test-only dependency — so in a production install (beartype absent) the
decorator degrades to `jaxtyped(typechecker=None)`, which is inert (no per-call cost, annotations are
just documentation). Scoped by explicit decoration, not an import hook, so exactly the annotated
boundaries are checked and nothing leaks to unrelated modules.

The cost is O(1) PER CALL (reads .shape/.dtype), independent of tensor size — negligible at coarse
seams (per-volume/per-batch, ~10^3-10^5 calls), but it adds up on element-granular hot loops. So there
are three tiers (bd cardiac-seg-ai0q follow-up):
  - warm seam        -> `@shapecheck`     : checked in tests, inert in prod. The default for boundaries.
  - hot, want docs   -> `@shapecheck_off` : checker pinned None, so NEVER checked (even in tests); the
                        jaxtyping annotation stays as living documentation. Use where the call is hot
                        enough that a per-call check would show up but the shape doc still earns its place.
  - element-loop hot -> no decorator      : a bare jaxtyping annotation is already inert (zero wrapper).
"""
from jaxtyping import jaxtyped

try:
    from beartype import beartype as _checker
except ImportError:                       # pragma: no cover  (prod w/o dev extra -> checks inert)
    _checker = None

shapecheck = jaxtyped(typechecker=_checker)     # test-enforced, prod-inert
shapecheck_off = jaxtyped(typechecker=None)     # deliberately never checked (hot path) — annotation = docs
