"""Unit tests for devtools/lcom.py — LCOM4 cohesion (positive split + the strategy-pattern exemptions)."""

import sys

import pytest

from devtools.lcom import Lcom

# two disjoint self-field groups: {a,b} touch self.x, {c,d} touch self.y -> LCOM4 == 2
_SPLIT = """
class Split:
    def a(self):
        return self.x
    def b(self):
        self.x += 1
    def c(self):
        return self.y
    def d(self):
        self.y += 1
"""

# every method touches self.x -> one component -> cohesive
_COHESIVE = """
class Cohesive:
    def a(self):
        return self.x
    def b(self):
        self.x += 1
    def c(self):
        return self.x * 2
"""


def test_lcom4_splits_disjoint_state(make_cls):
    score, comps = Lcom.lcom4(make_cls(_SPLIT))
    assert score == 2, f"two disjoint groups must give LCOM4==2, got {score}: {comps}"
    assert {frozenset(c) for c in comps} == {frozenset({"a", "b"}), frozenset({"c", "d"})}
    assert Lcom._is_split_candidate(make_cls(_SPLIT)) is not None


def test_lcom4_cohesive_is_one(make_cls):
    score, _ = Lcom.lcom4(make_cls(_COHESIVE))
    assert score == 1, "a class whose methods share a field is cohesive (LCOM4==1)"
    assert Lcom._is_split_candidate(make_cls(_COHESIVE)) is None


def test_lcom4_skips_interface_impl(make_cls):
    # subclasses a DOMAIN base -> its method split just mirrors the contract, not real fusion
    src = "class Backend(BaseStore):\n    def read(self):\n        return self.a\n    def draw(self):\n        return self.b\n"
    assert Lcom._is_split_candidate(make_cls(src)) is None


def test_lcom4_skips_abstract_and_trivial(make_cls):
    abstract = "class I(ABC):\n    def a(self):\n        return self.x\n    def b(self):\n        return self.y\n"
    trivial = "class Stub:\n    def a(self): ...\n    def b(self): ...\n"
    assert Lcom._is_split_candidate(make_cls(abstract)) is None
    assert Lcom._is_split_candidate(make_cls(trivial)) is None


def test_lcom4_ignores_fewer_than_two_methods(make_cls):
    src = "class One:\n    def a(self):\n        return self.x\n"
    assert Lcom._is_split_candidate(make_cls(src)) is None


def test_lcom_transformer_contract_excluded(make_cls):
    # fit writes learned state, transform reads different state -> raw LCOM4=2, but it's the sklearn
    # duck-typed contract (the split IS the interface), so it's exempt (bd 76i)
    src = "class Scaler:\n    def fit(self, X):\n        self.mean = X\n    def transform(self, X):\n        return self.scale\n"
    assert Lcom.lcom4(make_cls(src))[0] == 2, "fit/transform touch disjoint state -> raw LCOM4 is 2"
    assert Lcom._is_split_candidate(make_cls(src)) is None, "but fit+transform is exempt as the sklearn contract"


def test_lcom_fit_call_contract_excluded(make_cls):
    src = "class Est:\n    def fit(self, X):\n        self.a = X\n    def __call__(self, X):\n        return self.b\n"
    assert Lcom._is_split_candidate(make_cls(src)) is None, "fit + __call__ is the same duck-typed contract"


def test_lcom_main_requires_packages(monkeypatch):
    # nargs="+" -> no positional makes argparse exit(2), never a vacuous scan of a phantom 'src' (skr GAP2)
    monkeypatch.setattr(sys, "argv", ["devtools.lcom"])
    with pytest.raises(SystemExit) as exc:
        from devtools import lcom

        lcom.main()
    assert exc.value.code == 2, "no-arg invocation must be an argparse usage error"
