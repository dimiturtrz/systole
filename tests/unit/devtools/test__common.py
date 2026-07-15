"""Unit tests for devtools/_common.py — the shared source-tree walk + pyproject reader primitives."""

import ast

from devtools._common import Pyproject, Trees


def test_trees_walk_yields_path_and_parsed_ast(write_pkg, tmp_path):
    pkg = write_pkg(tmp_path, "walk_pos", "x = 1\nclass C: ...\n")
    walked = list(Trees([pkg]).walk())
    # __init__.py + mod.py, each as (Path, parsed Module)
    names = sorted(p.name for p, _ in walked)
    assert names == ["__init__.py", "mod.py"]
    assert all(isinstance(tree, ast.Module) for _, tree in walked)
    mod_tree = next(tree for p, tree in walked if p.name == "mod.py")
    assert any(isinstance(n, ast.ClassDef) for n in ast.walk(mod_tree)), "mod.py parses to a real AST"


def test_trees_files_lists_py_without_parsing(write_pkg, tmp_path):
    pkg = write_pkg(tmp_path, "files_pos", "y = 2\n")
    files = Trees([pkg]).files()
    assert sorted(p.name for p in files) == ["__init__.py", "mod.py"], "every .py under the package, no parse"


def test_pyproject_tool_section_reads_and_defaults(tmp_path):
    pp = tmp_path / "pyproject.toml"
    pp.write_text("[tool.structure]\nfile_max = 500\n")
    assert Pyproject.tool_section("structure", str(pp)) == {"file_max": 500}, "the named [tool.<section>] table"
    assert Pyproject.tool_section("absent", str(pp)) == {}, "a missing section is an empty dict"
    assert Pyproject.tool_section("structure", str(tmp_path / "none.toml")) == {}, "a missing file is an empty dict"
