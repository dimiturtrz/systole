"""Unit tests for devtools/shape_contracts.py — jaxtyping boundary gate (ML domain)."""

import ast
import sys

import pytest

from devtools.shape_contracts import ShapeContracts


def _fn(make_cls, src):
    """The first function defined in the first class of a snippet."""
    return next(m for m in make_cls(src).body if isinstance(m, ast.FunctionDef))


def test_shape_contracts_flags_bare_array_boundary(make_cls):
    names = {"ndarray", "Tensor"}
    # a public method with a bare np.ndarray param + Tensor return, no jaxtyping shape -> both flagged
    src = "class Seg:\n    def run(self, x: np.ndarray) -> Tensor:\n        return x\n"
    assert ShapeContracts._bare_array_slots(_fn(make_cls, src), names) == ["x", "->return"]


def test_shape_contracts_jaxtyping_satisfies(make_cls):
    names = {"ndarray", "Tensor"}
    # a jaxtyping subscript IS the contract -> silent; `Float[Tensor, "..."] | None` still counts
    src = (
        "class Seg:\n    def run(self, x: Float[Tensor, 'b c h w']) -> Int[np.ndarray, 'n'] | None:\n        return x\n"
    )
    assert ShapeContracts._bare_array_slots(_fn(make_cls, src), names) == [], "jaxtyping boundaries satisfy"


def test_shape_contracts_private_and_scalar_exempt(make_cls):
    names = {"ndarray", "Tensor"}
    private = _fn(make_cls, "class C:\n    def _h(self, x: np.ndarray): ...\n")
    scalar = _fn(make_cls, "class C:\n    def go(self, n: int) -> float: ...\n")
    assert ShapeContracts._public(private) is False, "underscore-prefixed methods are interior, not boundaries"
    assert ShapeContracts._bare_array_slots(scalar, names) == [], "a non-array signature is not a boundary"


def test_shape_contracts_alias_config_from_pyproject(make_cls, tmp_path):
    pp = tmp_path / "pyproject.toml"
    pp.write_text('[tool.shape_contracts]\narray_aliases = ["Volume", "Mask"]\n')
    names = ShapeContracts.array_names(str(pp))
    assert names == {"ndarray", "Tensor", "Volume", "Mask"}, "builtin arrays plus the repo's alias slot"
    fn = _fn(make_cls, "class C:\n    def seg(self, v: Volume) -> Mask: ...\n")
    assert ShapeContracts._bare_array_slots(fn, names) == ["v", "->return"], "alias boundaries flag like ndarray"
    assert ShapeContracts.array_names(str(tmp_path / "none.toml")) == {"ndarray", "Tensor"}, "absent -> builtins"


def test_shape_contracts_scan_and_assert(write_pkg, tmp_path):
    names = {"ndarray", "Tensor"}
    pkg = write_pkg(tmp_path, "shp", "class S:\n    def go(self, x: np.ndarray): ...\n")
    rows = ShapeContracts([pkg]).scan(names)
    assert len(rows) == 1 and rows[0][2] == "S.go", "the bare boundary surfaces in a package scan"
    clean = write_pkg(tmp_path, "shp_ok", "class S:\n    def go(self, x: Float[Tensor, 'n']): ...\n")
    assert ShapeContracts([clean]).scan(names) == []


def test_shape_contracts_main_requires_packages(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["devtools.shape_contracts"])
    with pytest.raises(SystemExit) as exc:
        from devtools import shape_contracts

        shape_contracts.main()
    assert exc.value.code == 2, "no-arg invocation must be an argparse usage error"
