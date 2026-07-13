"""Tensor-shape enforcement decorator (bd cardiac-seg-zwno).

`@shapecheck` on a tensor-boundary function makes its jaxtyping annotations (`Float[Tensor, "b c h w"]`)
LIVE: a wrong-shape / wrong-dtype array raises at the call instead of silently broadcasting. The runtime
checker is beartype, a dev/test-only dependency — so in a production install (beartype absent) the
decorator degrades to `jaxtyped(typechecker=None)`, which is inert (no per-call cost, annotations are
just documentation). Scoped by explicit decoration, not an import hook, so exactly the annotated
boundaries are checked and nothing leaks to unrelated modules.
"""
from jaxtyping import jaxtyped

try:
    from beartype import beartype as _checker
except ImportError:                       # pragma: no cover  (prod w/o dev extra -> checks inert)
    _checker = None

shapecheck = jaxtyped(typechecker=_checker)
