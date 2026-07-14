"""Task runner for cardioseg.

Every session shells out to ``uvx``/``uv run`` with pinned tool versions so a local ``nox`` run
executes exactly what CI executes. Sessions declare ``venv_backend="none"`` — nox does not build an
environment; ``uv`` owns dependency resolution.
"""

import nox

nox.options.sessions = ["lint", "test", "cov"]

RUFF = "ruff@0.15.13"
VULTURE = "vulture@2.16"
SELECT = "F,B,E501,I,T201,FBT,BLE001,S101,S110,C901,PLR0912,PLR0913,PLR0915,PLR2004,PLC0415,RUF100,N,E741,E742,E743,PLR0124,PLR1714,PLW3301,RUF012,RUF005,RUF007,RUF010,RUF022,RUF046,C408,C420,SIM,PERF401,PLW0108,E731,E402,ICN001,S603,S607,PTH123,SLF001"
LAYERS = ["core", "cardioseg"]
# ruff + jscpd are R1 HYGIENE gates — they may scan WIDER than the R2/R3 arch set LAYERS (a viewer / tests
# tree worth linting). Default = LAYERS; widen via lint_paths/jscpd_paths in .copier-answers.yml (bd 9mu).
LINT_LAYERS = ["core", "cardioseg", "cardioview", "tests"]
JSCPD_LAYERS = ["core", "cardioseg", "cardioview/web/src"]


@nox.session(venv_backend="none")
def lint(session: nox.Session) -> None:
    """ruff check (enforced) + ruff format --check (advisory) + vulture + import-linter + arch-fitness + ast-grep + jscpd."""
    # --select on the CLI bypasses pyproject [tool.ruff.lint] ignore, so the ml F722 waiver (jaxtyping dim
    # strings) is repeated here — matching CI (bd skr GAP1). Off domain=ml it's absent (no jaxtyping dep).
    session.run("uvx", RUFF, "check", *LINT_LAYERS, "--select", SELECT, "--ignore", "F722", external=True)
    # Advisory (mirrors CI, never blocks): full curated config over the whole tree + format drift.
    session.run("uvx", RUFF, "check", ".", "--statistics", external=True, success_codes=[0, 1])
    session.run("uvx", RUFF, "format", "--check", ".", external=True, success_codes=[0, 1])
    # vulture takes NO package args — scope is config-driven via [tool.vulture] paths (the vulture-scan
    # LOCAL-SLOT), same as CI + pre-commit. A repo widens the dead-code scan there, not on the CLI.
    session.run("uvx", VULTURE, "--min-confidence", "80", external=True)
    # Advisory: conf60 dead-code (real signal, but eats no-fan-in FPs so it can't block). vulture exits 3
    # when it finds dead code — success_codes swallows that (0=clean, 3=found), only a usage error blocks.
    session.run("uvx", VULTURE, "--min-confidence", "60", external=True, success_codes=[0, 3])
    session.run("uvx", "--from", "import-linter", "lint-imports", external=True)
    # --extra devtools: graph.py needs grimp+networkx (optional extra), same as CI's synced tests job.
    session.run(
        "uv", "run", "--extra", "devtools", "python", "-m", "devtools.graph", "--assert", *LAYERS, external=True
    )
    session.run(
        "uvx",
        "--from",
        "ast-grep-cli",
        "ast-grep",
        "scan",
        "-c",
        "devtools/sgconfig.yml",
        *LAYERS,
        external=True,
    )
    # DRY gate — ENFORCED (blocks over the jscpd.json threshold), matching the cardiac/mindscape majority.
    session.run(
        "npx",
        "--yes",
        "jscpd",
        *JSCPD_LAYERS,
        "--config",
        "devtools/jscpd.json",
        external=True,
    )
    # Advisory class-shape explorers — print a ranked report, always exit 0 (never block).
    for _tool in ("state_candidates", "lcom", "data_clumps"):
        session.run("uv", "run", "python", "-m", f"devtools.{_tool}", *LAYERS, external=True)
    # Advisory shape-contract gate — public array/tensor boundaries lacking a jaxtyping shape (aliases in
    # [tool.shape_contracts]). Report-only until a repo's tree is clean, then it graduates to `--assert`.
    session.run("uv", "run", "python", "-m", "devtools.shape_contracts", *LAYERS, external=True)
    # ENFORCED magic-literal ratchet — recurring string vocab + repeated dict schemas (StrEnum/record
    # candidates), the non-comparison axis ruff PLR2004 can't see. Blocks over the [tool.magic_literals]
    # ceiling (a per-repo FACT; fresh repo = 0/0). Migrate a new literal or raise the ceiling with a reason.
    session.run("uv", "run", "python", "-m", "devtools.magic_literals", *LAYERS, external=True)


@nox.session(venv_backend="none")
def test(session: nox.Session) -> None:
    """pytest."""
    session.run("uv", "run", "pytest", "tests", "-q", external=True)


@nox.session(venv_backend="none")
def cov(session: nox.Session) -> None:
    """pytest with coverage, failing under the floor."""
    session.run(
        "uv",
        "run",
        "pytest",
        "tests",
        "--cov",
        "--cov-report=term-missing",
        external=True,
    )
    session.run("uv", "run", "coverage", "report", "--fail-under=90", external=True)
    # Advisory: 95% target — warns, never fails (coverage exits 2 when under; success_codes swallows it).
    session.run("uv", "run", "coverage", "report", "--fail-under=95", external=True, success_codes=[0, 2])


@nox.session(venv_backend="none")
def gates(session: nox.Session) -> None:
    """Run every gate: lint + test + cov."""
    lint(session)
    test(session)
    cov(session)
