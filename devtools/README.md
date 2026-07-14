# devtools — the guardrail tools shipped with this project

Each tool is engine + config, invoked identically in `noxfile.py` and CI. Targets are the `packages`
list (space-separated below as `<packages>`). Every tool ships — the gates are the non-optional house bar.

- **graph.py** — import-graph arch fitness (god-module = fan-in AND fan-out over a degree / import cycle / god-file / **test-mirror gap**). Gate: `python -m devtools.graph --assert <packages>` (thresholds in `pyproject [tool.structure]`). Explorer: drop `--assert` for ranked fan-in/out/bottleneck/chokepoint tables.
- **test-mirror rule** — every logic module needs a unit test. `[tool.structure] test_layout` sets *where*: `"mirror"` (default, strict — `<pkg>/<path>/foo.py` → `tests/unit/<pkg>/<path>/test_foo.py`, one home per module), `"flat"` (a `test_foo.py` anywhere under `tests/`), or `"off"`. `__init__`/`__main__` and coverage-**omitted** shells (`[tool.coverage] omit`, via **omit.py**) are exempt — a non-unit-testable shell isn't forced to carry a stub.
- **sg-rules/** + **sgconfig.yml** — ast-grep module-shape rules (behaviour in a class, no import-time side effects). `uvx --from ast-grep-cli ast-grep scan -c devtools/sgconfig.yml <packages>` (enforced).
- **jscpd.json** — jscpd copy-paste (DRY) threshold. `npx jscpd <packages> --config devtools/jscpd.json` (advisory).
- **lcom.py** — LCOM4 cohesion: ranks concrete stateful classes whose methods split into disjoint-state groups. `python -m devtools.lcom <packages>`.
- **data_clumps.py** — Fowler data clumps: param sets that travel together across signatures (Introduce Parameter Object). `python -m devtools.data_clumps <packages>`.
- **state_candidates.py** — namespace classes with latent shared instance state. `python -m devtools.state_candidates <packages>`.

  The three class-shape tools are ADVISORY explorers — they print a ranked report and always exit 0, never blocking.
- **magic_literals.py** — the non-comparison, cross-file axis ruff PLR2004 can't see: recurring identifier-shaped string literals (>= 4x → `StrEnum`/named-constant candidate) + repeated dict key-sets (a drift-prone implicit record → dataclass/TypedDict). `python -m devtools.magic_literals <packages>` — an **ENFORCED count-ratchet**: the ceilings live in `[tool.magic_literals] max_strings/max_key_sets` (a fresh repo ships `0/0`), the current floor freezes as a ceiling, and a NEW recurring literal fails the merge (migrate it to an enum, or raise the ceiling in the same commit with a reason). Raise a ceiling only in the `# >>> LOCAL-SLOT: magic-ceilings` region. `--max-strings N`/`--max-key-sets N` override the config ad-hoc; delete the `[tool.magic_literals]` section to fall back to an advisory report.
- **analytics.py** — a one-shot **explorer** (not a gate, not CI-wired): per-area code lines, def count, McCabe branch-proxy, branches-per-def (logic leaking into fat leaves), src:test ratio, and the top-N largest files. `python -m devtools.analytics --areas <packages> devtools` (add `--flag-over 250` for a code-line budget list). Run it by hand when you want the size/complexity radar.
- **shape_contracts.py** — the ML-domain shape gate: a public method whose param/return is an array/tensor (`np.ndarray`/`Tensor`, or a repo alias listed in `[tool.shape_contracts] array_aliases`) must carry a **jaxtyping** shape (`Float[Tensor, "b c h w"]`), so the shape is a checked contract not a silent assumption. `python -m devtools.shape_contracts <packages>` (**advisory** report). Add `--assert` to block once your tree is clean — a new bare-array boundary then fails the merge.

## Making jaxtyping shapes LIVE — `@shapecheck`

A bare `Float[Tensor, "b c h w"]` annotation is **inert** at runtime — it documents the shape but checks
nothing. The scaffold ships the deps (`jaxtyping` + `beartype`) but **no package code** (the template is
guardrails-only). Add a one-line decorator in your own types module to make the annotations enforce:

```python
# core/types.py (or wherever your shared type vocabulary lives)
from beartype import BeartypeConf, beartype as _beartype
from jaxtyping import jaxtyped

# is_pep484_tower: an int also satisfies a float annotation (numpy scalars included).
shapecheck = jaxtyped(typechecker=_beartype(conf=BeartypeConf(is_pep484_tower=True)))
```

Then `@shapecheck` on a boundary makes a wrong-shape/dtype array raise at the call. It's O(1) per call
(reads `.shape`/`.dtype`), independent of tensor size — negligible at coarse seams. A genuinely hot call
that wants the annotation as docs only just leaves `@shapecheck` off. `shape_contracts.py` (above) is the
static ratchet that surfaces boundaries still missing a jaxtyping shape.


## Directional layer contracts — import-linter

`graph.py`'s structural checks are layer-AGNOSTIC and catch *cycles*, but not a one-way *forbidden*
import (`core -> trainer` is no cycle). That directional axis is import-linter's.

Shipped: `[tool.importlinter]` in `pyproject.toml` carries `root_packages` + a kernel-independence
starter contract (the first package imports none of the others) inside a `# >>> LOCAL-SLOT: import-contracts`
region — add your viewer/trainer or domain contracts there. Enforced in nox/CI/pre-commit via `lint-imports`.

