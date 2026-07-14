"""Tensor-shape enforcement decorators + the shape-contract convention (bd cardiac-seg-zwno / wk8m).

`@shapecheck` on a boundary function makes its jaxtyping annotations (`Float[Tensor, "b c h w"]`) LIVE: a
wrong-shape / wrong-dtype array raises at the call instead of silently broadcasting. The runtime checker
is beartype, a dev/test-only dependency — so in a production install (beartype absent) the decorator
degrades to `jaxtyped(typechecker=None)`, which is inert (no per-call cost, annotations are just docs).
Scoped by explicit decoration, not an import hook, so exactly the annotated boundaries are checked and
nothing leaks to unrelated modules.

Cost is O(1) PER CALL (reads .shape/.dtype), independent of tensor size — negligible at coarse seams,
and this codebase avoids tiny/element-granular functions, so checks never land on hot inner loops. Three
tiers:
  - warm seam        -> `@shapecheck`     : checked in tests, inert in prod. The default for boundaries.
  - hot, want docs   -> `@shapecheck_off` : checker pinned None, NEVER checked (even in tests); the
                        jaxtyping annotation stays as living documentation.
  - element-loop hot -> no decorator      : a bare jaxtyping annotation is already inert (zero wrapper).

## The shape-contract convention (bd cardiac-seg-dnx6) — how to annotate at scale

beartype checks EVERY annotated param of a decorated fn (not just the array ones), and it is strict, so
loose scientific-Python types need a shared convention or the rollout drowns in false positives:

  - **arrays / tensors** -> jaxtyping, TAIL-DIM / VARIADIC so batch stays optional and broadcasting is
    free: `Float[Tensor, "*b c h w"]`, `Integer[np.ndarray, "*grid"]`. A pure reduction (a count/sum
    that ignores shape) is DTYPE-ONLY: `Integer[np.ndarray, "..."]`. Name a dim to TIE it across args
    (ed/es same grid, mask/img same batch).
  - **python scalars** -> the checker enables the PEP 484 numeric tower (`is_pep484_tower=True`), so an
    `int` satisfies a `float` annotation and `np.float64` (a real `float` subclass) is accepted. But
    `np.float32` / `np.int64` are NOT python-scalar subclasses, so a param that receives a numpy scalar
    (e.g. a spacing off a stored header) uses the numpy-tolerant aliases `Real` / `Integral`
    (`core.types`), not bare `float` / `int`.
  - **complex non-array types** (polars DataFrame, torch Module, omegaconf, pydantic models) -> annotate
    SHALLOW (the bare class, whose isinstance beartype handles) or leave UNANNOTATED — beartype only
    checks annotated params, so an unannotated param is simply skipped. No deep generics at enforced
    boundaries (their stubs trip beartype).
"""
from jaxtyping import jaxtyped

try:
    from beartype import BeartypeConf
    from beartype import beartype as _beartype
    _checker = _beartype(conf=BeartypeConf(is_pep484_tower=True))   # int satisfies float; np.float64 too
except ImportError:                       # pragma: no cover  (prod w/o dev extra -> checks inert)
    _checker = None

shapecheck = jaxtyped(typechecker=_checker)     # test-enforced, prod-inert
shapecheck_off = jaxtyped(typechecker=None)     # deliberately never checked (hot path) — annotation = docs
