"""Unit tests for devtools/state_candidates.py — namespace latent-state promotion candidates."""

import sys

import pytest

from devtools.state_candidates import StateCandidates

_BAG = """
class Bag:
    @staticmethod
    def load(cfg, path):
        return cfg
    @staticmethod
    def save(cfg, data):
        return cfg
"""


def test_shared_state_flags_threaded_param(make_cls):
    shared = StateCandidates.shared_state(make_cls(_BAG))
    assert shared == {"cfg": 2}, f"a param shared by all staticmethods is latent state, got {shared}"


def test_shared_state_skips_stateful_class(make_cls):
    src = (
        "class Stateful:\n"
        "    def __init__(self, cfg):\n"
        "        self.cfg = cfg\n"
        "    @staticmethod\n"
        "    def load(cfg, path): ...\n"
        "    @staticmethod\n"
        "    def save(cfg, data): ...\n"
    )
    assert StateCandidates.shared_state(make_cls(src)) == {}, "a class with __init__ is already stateful — skip"


def test_shared_state_skips_pydantic_and_command(make_cls):
    pydantic = (
        "class Cfg(BaseModel):\n    @staticmethod\n    def a(cfg, x): ...\n    @staticmethod\n    def b(cfg, y): ...\n"
    )
    command = (
        "class Cmd:\n    @staticmethod\n    def add_args(cfg, p): ...\n    @staticmethod\n    def run(cfg, a): ...\n"
    )
    assert StateCandidates.shared_state(make_cls(pydantic)) == {}
    assert StateCandidates.shared_state(make_cls(command)) == {}


def test_shared_state_skips_autograd_function(make_cls):
    # forward/backward thread ctx by the torch.autograd.Function contract, not latent instance state (76i)
    src = (
        "class GradReverse(Function):\n"
        "    @staticmethod\n    def forward(ctx, x):\n        return x\n"
        "    @staticmethod\n    def backward(ctx, g):\n        return g\n"
    )
    assert StateCandidates.shared_state(make_cls(src)) == {}, "autograd.Function threads ctx by contract"


def test_scan_skips_coverage_omit_shells(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "shell.py").write_text(_BAG)  # a namespace bag that WOULD flag (load/save thread cfg)
    # coverage-omitted -> the shell is skipped (its shared params are data, not object identity)
    (tmp_path / "pyproject.toml").write_text('[tool.coverage.run]\nomit = ["pkg/shell.py"]\n')
    assert StateCandidates(["pkg"]).scan() == [], "a coverage-omitted shell is not a state-promotion candidate"
    # not omitted -> the same bag surfaces (proves the skip fires, not a broken scan)
    (tmp_path / "pyproject.toml").write_text("[tool.coverage.run]\nomit = []\n")
    assert StateCandidates(["pkg"]).scan(), "un-omitted, the namespace bag is flagged"


def test_state_candidates_main_requires_packages(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["devtools.state_candidates"])
    with pytest.raises(SystemExit) as exc:
        from devtools import state_candidates

        state_candidates.main()
    assert exc.value.code == 2, "no-arg invocation must be an argparse usage error"
