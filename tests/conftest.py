"""Suite-wide pytest config.

`--update-golden` regenerates committed golden fixtures from the live pipeline instead of asserting
against them — the reproducible replacement for a throwaway generator script (a golden test recomputes
its own fixture through the SAME builders it asserts with, so there is no separate, driftable generator).
Run: `uv run pytest tests/integration/test_synth_golden.py --update-golden`.
"""


def pytest_addoption(parser):
    parser.addoption("--update-golden", action="store_true", default=False,
                     help="Recompute golden fixtures from the live pipeline instead of asserting against them.")
