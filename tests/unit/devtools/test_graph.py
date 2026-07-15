"""Unit tests for devtools/graph.py — the arch-fitness gate + Martin stability metrics + test-mirror."""

import sys

import networkx as nx
import pytest

from devtools.graph import ImportGraph


def test_structure_cfg_merges_over_defaults(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool.structure]\nfile_max = 500\n")
    cfg = ImportGraph.load_structure_cfg(str(pyproject))
    assert cfg["file_max"] == 500, "an explicit threshold overrides the default"
    assert cfg["bottleneck_degree"] == 8, "an unspecified threshold keeps the default"


def test_structure_cfg_all_defaults_when_absent(tmp_path):
    cfg = ImportGraph.load_structure_cfg(str(tmp_path / "nope.toml"))
    assert cfg == {
        "bottleneck_degree": 8,
        "file_max": 750,
        "file_min": 0,
        "betweenness_max": 0.10,
        "main_sequence_max": 0.0,
        "test_layout": "mirror",
    }


def test_god_module_detected():
    g = nx.DiGraph()
    for i in range(9):  # fan-in 9 and fan-out 9, both > 8
        g.add_edge(f"in{i}", "god")
        g.add_edge("god", f"out{i}")
    assert ImportGraph._god_modules(g, 8), "fan-in AND fan-out both over the degree is a god-module"
    clean = nx.DiGraph([("a", "b"), ("b", "c")])
    assert ImportGraph._god_modules(clean, 8) == []


def test_import_cycle_detected():
    cyclic = nx.DiGraph([("a", "b"), ("b", "a")])
    assert ImportGraph._cycles(cyclic), "a strongly-connected component >1 is an import cycle"
    assert ImportGraph._cycles(nx.DiGraph([("a", "b"), ("b", "c")])) == []


def test_oversized_file_detected():
    assert ImportGraph._oversized([("big.py", 800)], 750), "a file over the ceiling is a god-file"
    assert ImportGraph._oversized([("small.py", 100)], 750) == []


def test_undersized_floor_advisory():
    assert ImportGraph._undersized([("tiny.py", 3)], 0) == [], "floor OFF at file_min<=0 (the default)"
    assert ImportGraph._undersized([("tiny.py", 3)], 10), "under the floor flags (advisory)"
    assert ImportGraph._undersized([("ok.py", 50)], 10) == [], "above the floor is silent"
    assert ImportGraph._undersized([("pkg/__init__.py", 1)], 10) == [], "package plumbing is exempt"


def test_assert_fitness_clean_vs_dirty():
    cfg = {"bottleneck_degree": 8, "file_max": 750, "file_min": 0, "betweenness_max": 0.10, "main_sequence_max": 0.0}
    clean = nx.DiGraph([("a", "b"), ("b", "c")])
    blocking, _ = ImportGraph.assert_fitness(clean, [("a.py", 100)], cfg)
    assert blocking == [], "a clean graph + small files has no blocking violations"
    cyclic = nx.DiGraph([("a", "b"), ("b", "a")])
    blocking, _ = ImportGraph.assert_fitness(cyclic, [("big.py", 900)], cfg)
    assert blocking, "a cycle + an oversized file must block"


def test_instability_ratio():
    g = nx.DiGraph()
    for src in ("a", "b", "c"):  # X imported by a,b,c (Ca=3) and imports Y (Ce=1) -> I = 1/(1+3)
        g.add_edge(src, "X")
    g.add_edge("X", "Y")
    inst = ImportGraph.instability(g)
    assert inst["X"] == 0.25, "I = Ce/(Ce+Ca) = out/(out+in)"
    assert inst["Y"] == 0.0, "Y only depended-on (Ce=0) -> maximally stable"
    assert inst["a"] == 1.0, "a only depends (Ca=0) -> maximally unstable"


def test_is_abstract_detects_abc_metaclass_abstractmethod(make_cls):
    assert ImportGraph._is_abstract(make_cls("class I(ABC):\n    pass\n"))
    assert ImportGraph._is_abstract(make_cls("class M(metaclass=ABCMeta):\n    pass\n"))
    assert ImportGraph._is_abstract(make_cls("class A:\n    @abstractmethod\n    def go(self): ...\n"))
    assert not ImportGraph._is_abstract(make_cls("class C:\n    def go(self):\n        return 1\n"))


def test_abstractness_ratio_from_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text(
        "from abc import ABC\nclass I(ABC): ...\nclass C:\n    def go(self):\n        return 1\n"
    )
    assert ImportGraph.abstractness("pkg.mod") == 0.5, "1 abstract + 1 concrete -> A = 0.5"
    (pkg / "noclass.py").write_text("X = 1\n")
    assert ImportGraph.abstractness("pkg.noclass") is None, "no classes -> A undefined"
    assert ImportGraph.abstractness("pkg.missing") is None, "no backing file -> None"


def test_main_sequence_distance_and_off_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "leaf.py").write_text("class C:\n    def go(self):\n        return 1\n")  # concrete, A=0
    g = nx.DiGraph()
    g.add_edge("importer", "pkg.leaf")  # leaf depended-on only: Ca=1, Ce=0 -> I=0 -> D=|0+0-1|=1.0
    assert ImportGraph.main_sequence_distance(g)["pkg.leaf"] == 1.0
    assert ImportGraph._off_main_sequence(g, 0.0) == [], "OFF by default (a concrete stable leaf sits at D≈1)"
    assert ImportGraph._off_main_sequence(g, 0.7), "opt in with a threshold -> the far-off module surfaces"


def test_unmirrored_flags_missing_mirror(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "foo.py").write_text("class Foo:\n    @staticmethod\n    def go(): ...\n")
    assert ImportGraph(["pkg"]).unmirrored(), "a logic module with no mirror test must be flagged"
    assert all("__init__" not in m for m in ImportGraph(["pkg"]).unmirrored())
    mirror = tmp_path / "tests" / "unit" / "pkg"
    mirror.mkdir(parents=True)
    (mirror / "test_foo.py").write_text("def test_foo(): pass\n")
    assert ImportGraph(["pkg"]).unmirrored() == [], "a module with its strict path-mirror test is satisfied"


def test_unmirrored_flat_layout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "foo.py").write_text("class Foo:\n    @staticmethod\n    def go(): ...\n")
    (tmp_path / "tests").mkdir()
    assert ImportGraph(["pkg"]).unmirrored("flat"), "flat: no test anywhere -> flagged"
    (tmp_path / "tests" / "test_foo.py").write_text("def test_foo(): pass\n")
    assert ImportGraph(["pkg"]).unmirrored("flat") == [], "flat: a test_foo.py under tests/ satisfies it"
    assert ImportGraph(["pkg"]).unmirrored("mirror"), "mirror still wants the strict path"


def test_unmirrored_off(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "foo.py").write_text("class Foo:\n    @staticmethod\n    def go(): ...\n")
    assert ImportGraph(["pkg"]).unmirrored("off") == [], "off = no test-existence gate"


def test_unmirrored_exempts_omitted_shell(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[tool.coverage.run]\nomit = ["pkg/shell.py"]\n')
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "shell.py").write_text("class Shell:\n    @staticmethod\n    def run(): ...\n")
    assert ImportGraph(["pkg"]).unmirrored() == [], "a coverage-omitted shell needs no mirror test"


def test_graph_main_requires_packages(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["devtools.graph"])
    with pytest.raises(SystemExit) as exc:
        from devtools import graph

        graph.main()
    assert exc.value.code == 2, "no-arg invocation must be an argparse usage error"
