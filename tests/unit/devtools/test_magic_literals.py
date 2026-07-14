"""Unit tests for the magic-literal detector (bd cardiac-seg-wir7): the recurring-string smell (value
tokens, with dict-key / subscript / framework exclusions) and the repeated dict-key-set smell."""
import ast
import textwrap

from devtools.magic_literals import _key_sets, _string_literals, scan_key_sets, scan_strings


def _tree(src):
    return ast.parse(textwrap.dedent(src))


def test_value_tokens_counted_keys_and_subscripts_excluded():
    """A string in a VALUE/arg position counts; the same string as a dict KEY or a subscript index is a
    field ref (schema, not value) and is excluded."""
    toks = _string_literals(_tree("""
        x = f("GE")
        rec = {"vendor": "GE"}          # "vendor" is a key (excluded); "GE" is a value (counted)
        v = row["vendor"]               # subscript index -> field ref (excluded)
        y = "GE"
    """))
    assert sorted(toks) == ["GE", "GE", "GE"]   # 3 value-position "GE" (call arg, dict value, assign); no "vendor"


def test_comparison_operands_excluded():
    """A string in a comparison is ruff PLR2004's domain (bd 1ln7), so the detector defers it; the same
    token in a value/arg position still counts."""
    toks = _string_literals(_tree("""
        if x == "GE": pass          # comparison operand -> ruff's job (excluded)
        if "GE" != y: pass          # left operand too (excluded)
        z = tag("GE")               # value/arg position (counted)
    """))
    assert toks == ["GE"]           # only the arg-position one


def test_framework_and_prose_not_counted():
    """argparse action literals (stoplist) and prose/messages (have spaces -> not identifier-shaped) drop."""
    toks = _string_literals(_tree('''
        """A module docstring with spaces."""
        ap.add_argument("--x", action="store_true")
        log.info("a long human message")
        d = "acdc"
    '''))
    assert toks == ["acdc"]                 # store_true stoplisted; docstring/message have spaces


def test_scan_strings_threshold(tmp_path):
    """Only tokens at or above the frequency threshold (4) surface."""
    (tmp_path / "m.py").write_text('a="GE"\nb="GE"\nc="GE"\nd="GE"\ne="rare"\n')
    rows = dict(scan_strings([str(tmp_path)]))
    assert rows["GE"] == 4 and "rare" not in rows


def test_key_sets_same_schema_flagged(tmp_path):
    """A constant-string-key dict schema built in >= 2 places is flagged; a one-off is not."""
    (tmp_path / "a.py").write_text('r = {"ef": 1, "edv": 2, "esv": 3}\n')
    (tmp_path / "b.py").write_text('s = {"edv": 9, "esv": 8, "ef": 7}\nt = {"solo": 1, "one": 2}\n')
    rows = scan_key_sets([str(tmp_path)])
    schemas = {keys for _, keys, _ in rows}
    assert ("edv", "ef", "esv") in schemas          # same 3-key set, 2 sites
    assert ("one", "solo") not in schemas           # single site -> not flagged


def test_small_or_dynamic_dicts_ignored():
    """A 1-key dict, or a dict with a non-constant key, isn't a schema."""
    assert _key_sets(_tree('x = {"only": 1}')) == []           # < min size 2
    assert _key_sets(_tree('x = {k: 1, "b": 2}')) == []        # non-constant key -> not a fixed schema
