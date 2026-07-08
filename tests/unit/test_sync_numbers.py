"""Doc number-syncing (sync_numbers) — the marker-block injection (pure string transform) + the table
renderers (given the committed RESULTS.json, they must emit well-formed markdown). The per-file
read/write loop in main() is the shell; inject_blocks + the block fns are pure and tested here.
"""
from cardioseg.evaluation import sync_numbers as sn
from cardioseg.evaluation.sync_numbers import BLOCKS, inject_blocks


# --- inject_blocks: substitution equivalence classes ---
def test_inject_replaces_matching_block():
    """Present class: a marker span gets its body replaced with the fn output between the markers."""
    txt = "before\n<!-- results:x -->\nOLD\n<!-- /results:x -->\nafter"
    out, n = inject_blocks(txt, {"x": lambda: "NEW"})
    assert n == 1
    assert out == "before\n<!-- results:x -->\nNEW\n<!-- /results:x -->\nafter"


def test_inject_absent_block_no_change():
    """Absent class: a block key not present in the text -> untouched, count 0."""
    txt = "no markers here"
    out, n = inject_blocks(txt, {"x": lambda: "NEW"})
    assert out == txt and n == 0


def test_inject_leaves_unlisted_markers():
    """Selective class: a marker in the text whose key isn't in `blocks` stays as-is."""
    txt = "<!-- results:y -->\nKEEP\n<!-- /results:y -->"
    out, n = inject_blocks(txt, {"x": lambda: "NEW"})
    assert out == txt and n == 0


def test_inject_is_idempotent():
    """Idempotence class: re-injecting the same block twice yields the identical text."""
    txt = "<!-- results:x -->\nstale\n<!-- /results:x -->"
    once, _ = inject_blocks(txt, {"x": lambda: "V"})
    twice, _ = inject_blocks(once, {"x": lambda: "V"})
    assert once == twice


def test_inject_counts_multiple_blocks():
    """Multi class: two present markers -> count 2, both replaced."""
    txt = "<!-- results:a -->\n1\n<!-- /results:a -->\n<!-- results:b -->\n2\n<!-- /results:b -->"
    out, n = inject_blocks(txt, {"a": lambda: "A", "b": lambda: "B"})
    assert n == 2 and "A" in out and "B" in out


# --- block renderers: well-formed markdown off the committed RESULTS.json ---
def test_all_blocks_render_nonempty_markdown():
    """Every registered block returns a non-empty string; table blocks carry a header separator row."""
    for key, fn in BLOCKS.items():
        s = fn()
        assert isinstance(s, str) and s.strip(), key


def test_table_blocks_have_header_and_separator():
    """The tabular blocks are valid markdown tables: >=2 lines, a `|---|` separator second line."""
    for key in ("compare", "acdc", "axis", "cardcompare", "nnucompare"):
        rows = BLOCKS[key]().splitlines()
        assert len(rows) >= 2 and set(rows[1]) <= set("|-"), key


def test_headline_is_prose_with_numbers():
    """headline is prose (no table), naming the unseen vendors + a Dice value."""
    h = sn.headline()
    assert "Dice" in h and "|" not in h
