# Type-checker evaluation: ty vs pyright vs mypy (+ shape tools)

**bd cardiac-seg-rp8l.** Which static type checker (if any) to adopt as the deferred `ANN*`/type axis,
and how. Measured on this repo (`core` + `cardioseg`, annotation-light torch/numpy/polars code) on
2026-07-13. Not a literature survey — the numbers below are from running each tool here.

## Measured surface

| tool | version | surface (core) | surface (core+cardioseg) | venv resolution | speed | maturity |
|------|---------|----------------|--------------------------|-----------------|-------|----------|
| **ty** (astral) | 0.0.59 | 40 | 62 | **auto** (uses the uv env) | Rust, ~instant | **preview / pre-1.0** |
| **mypy** | 2.3.0 | 30 (`--ignore-missing-imports`) | — | needs `--ignore-missing-imports` or stubs | slow (compiled helps) | reference impl, PEP-driving |
| **pyright** | 1.1.411 | 221* | — | needs a `pyrightconfig`/`--venvpath` | Node, fast | mature, strictest inference (powers Pylance) |

\* pyright's 221 is **inflated**: run via `uvx` it did not resolve the project `.venv`, so most errors are
`reportMissingImports` on `torch`/`numpy` (false). Point it at `.venv` and the real surface would fall to
the same ~tens-of-errors band as ty/mypy. ty auto-detected the uv environment (it noted the editable
installs) — no config needed.

## Signal quality (the crux on annotation-light torch code)

ty's 62 split into two kinds:

- **Real bug-class signal (keep):** `unresolved-attribute: Attribute 'head' is not defined on 'None' in
  union 'Unknown | None | DataFrame'`, `Attribute 'resident' is not defined on 'None'` — i.e. an Optional
  used without narrowing → a latent `AttributeError`. This is the SAME class ruff's F821 caught during the
  lint work (a real defect, not style). This is the reason to run a type checker at all.
- **Stub-precision noise:** the bulk (`invalid-argument-type` on `Measure.ejection_fraction`,
  `label_volume_ml`, …) trace to a `spacing` param typed loosely (`tuple[float, ...]`) vs a
  `tuple(float(s) for s in case["spacing"])` value, plus incomplete torch/numpy stubs producing
  `no-matching-overload` / `not-subscriptable`. Low signal until the boundary annotations tighten.

So a big-bang strict gate would drown the ~handful of real Optional-narrowing bugs in stub noise. Gradual,
boundary-first typing is the only sane adoption path here.

## Shape tools — the higher-value axis for THIS codebase

None of the three checkers see tensor **shape/dtype** — yet the code is dense with `[B, C, H, W]` /
`[D, H, W]` invariants (every synth/measure/inference boundary). The tool that matches this code:

- **jaxtyping** (`Float[Tensor, "B C H W"]`) — shape+dtype in the annotation, works for torch/numpy. Static
  checkers treat it as a plain annotation; the payoff is **runtime** enforcement via **beartype** (or
  typeguard) — decorate the boundary fns, and a wrong-shape tensor raises AT the call in tests instead of
  a silent broadcast. `torchtyping` is deprecated → jaxtyping is the successor. `nptyping` is unmaintained.
- This catches the bug class the `[B,C,H,W]`-comments are guarding against by hand today, and it's
  orthogonal to (composes with) whichever static checker we pick.

## Recommendation

1. **Static checker: adopt `ty`, advisory-first.** It aligns with the toolchain already in use (ruff 0.15 +
   uv are both astral; ty shares that config/speed/venv-resolution story — zero-config here, Rust-fast). Run
   it **advisory** in CI now (`continue-on-error`, like jscpd was) to surface the real Optional-narrowing
   bugs. Do **not** hard-gate yet: at v0.0.59 the rules/output still churn. Graduate to blocking when ty
   reaches ~0.1+/stabilizes (mirror the jscpd/coverage ratchet). If a mature blocking gate is wanted before
   then, pyright-in-`basic`-mode with a `.venv`-pointed config is the fallback (mature, but Node dep + noisier).
2. **Adoption strategy: gradual, boundary-first.** Annotate the core kernel's public boundaries (the
   `Measure`/`Evaluate`/`Inference`/`Preprocess` signatures + the Optional-returning store/split fns) first —
   that both kills the stub-noise `invalid-argument-type` cluster AND is where the real narrowing bugs live.
   This IS the `ANN*` ruff axis, done incrementally, not a strict big-bang.
3. **Shape safety: adopt jaxtyping + beartype at the tensor boundaries** (higher ROI than generic typing for
   this code). Annotate the synth/measure/inference tensor fns with `Float[Tensor, "..."]` and turn on
   beartype in the test env so shape mismatches fail loudly in tests. Track as its own bead.

**Net:** type-checking is worth it here mainly for (a) Optional-narrowing bugs (ty advisory catches these
now) and (b) tensor-shape safety (jaxtyping+beartype, not any of the three checkers). Sequence: ty advisory
→ annotate boundaries → jaxtyping on tensor fns → graduate ty to blocking once stable.
