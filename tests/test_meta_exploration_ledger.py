"""Tests for the auto-cube hinter ledger (~/auto_cube/hints.json).

Phase 4 foundation: the cross-session disposition record that mirrors
upstream Auto-CUBE's hinter pipeline. These tests pin the on-disk shape,
the round-trip, the level-2/3/4 enum validation, and the merge semantics
of ``upsert_entry``.

Tests use ``CH_AUTO_CUBE_DIR`` to redirect the ledger to a per-test
``tmp_path`` so the real ``~/auto_cube/hints.json`` is never touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cube_harness.meta_exploration.ledger import (
    DISPOSITIONS,
    HINT_TYPES,
    PROMOTED_TO_VALUES,
    HintLedgerEntry,
    auto_cube_dir,
    ledger_path,
    load_ledger,
    make_ledger_key,
    save_ledger,
    upsert_entry,
)


@pytest.fixture
def isolated_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point CH_AUTO_CUBE_DIR at a tmp dir for the test, return ledger path."""
    monkeypatch.setenv("CH_AUTO_CUBE_DIR", str(tmp_path))
    return tmp_path / "hints.json"


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def test_auto_cube_dir_honors_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CH_AUTO_CUBE_DIR", str(tmp_path / "elsewhere"))
    assert auto_cube_dir() == tmp_path / "elsewhere"


def test_auto_cube_dir_defaults_to_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CH_AUTO_CUBE_DIR", raising=False)
    assert auto_cube_dir() == Path.home() / "auto_cube"


def test_ledger_path_is_dir_plus_hints_json(isolated_ledger: Path) -> None:
    assert ledger_path() == isolated_ledger
    assert ledger_path().name == "hints.json"


# ---------------------------------------------------------------------------
# Key format
# ---------------------------------------------------------------------------


def test_make_ledger_key_canonical_form() -> None:
    assert make_ledger_key("miniwob", "use-slider") == "miniwob|use-slider"


@pytest.mark.parametrize("cube,task_id", [("", "x"), ("x", ""), ("", "")])
def test_make_ledger_key_rejects_empty(cube: str, task_id: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        make_ledger_key(cube, task_id)


@pytest.mark.parametrize("cube,task_id", [("a|b", "x"), ("x", "a|b")])
def test_make_ledger_key_rejects_pipe_in_components(cube: str, task_id: str) -> None:
    with pytest.raises(ValueError, match="reserved separator"):
        make_ledger_key(cube, task_id)


# ---------------------------------------------------------------------------
# Empty / missing ledger handling
# ---------------------------------------------------------------------------


def test_load_ledger_missing_file_returns_empty(isolated_ledger: Path) -> None:
    assert not isolated_ledger.exists()
    assert load_ledger() == {}


def test_load_ledger_malformed_json_returns_empty(isolated_ledger: Path) -> None:
    isolated_ledger.parent.mkdir(parents=True, exist_ok=True)
    isolated_ledger.write_text("{ not valid json")
    assert load_ledger() == {}


def test_load_ledger_non_dict_root_returns_empty(isolated_ledger: Path) -> None:
    isolated_ledger.parent.mkdir(parents=True, exist_ok=True)
    isolated_ledger.write_text("[1, 2, 3]")
    assert load_ledger() == {}


def test_load_ledger_drops_malformed_entries(isolated_ledger: Path) -> None:
    isolated_ledger.parent.mkdir(parents=True, exist_ok=True)
    isolated_ledger.write_text(
        json.dumps(
            {
                "bad_no_pipe": {"disposition": "steered"},  # no pipe in key
                "miniwob|good": {"disposition": "promoted"},  # OK
                "miniwob|broken": "not_a_dict",  # not a dict value
            }
        )
    )
    out = load_ledger()
    assert set(out.keys()) == {"miniwob|good"}
    assert out["miniwob|good"].disposition == "promoted"


# ---------------------------------------------------------------------------
# Round-trip save/load
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip(isolated_ledger: Path) -> None:
    entries = {
        "miniwob|use-slider": HintLedgerEntry(
            disposition="promoted",
            hint_type="task_specific",
            hint_text="Use PageUp for big slider gaps.",
            promoted_to="task_hints",
            last_session="sess-2026-05-25",
            notes={"confidence": 4},
        ),
        "terminal_bench_2|distribution-search": HintLedgerEntry(
            disposition="cheat_only",
            hint_type="task_specific",
            hint_text="See pre-existing solution sketch.",
            last_session="sess-2026-05-25",
        ),
    }
    save_ledger(entries)
    assert isolated_ledger.exists()
    loaded = load_ledger()
    assert set(loaded.keys()) == set(entries.keys())
    assert loaded["miniwob|use-slider"].disposition == "promoted"
    assert loaded["miniwob|use-slider"].promoted_to == "task_hints"
    assert loaded["miniwob|use-slider"].notes == {"confidence": 4}
    assert loaded["terminal_bench_2|distribution-search"].promoted_to is None


def test_save_ledger_atomic_via_tmp_rename(isolated_ledger: Path) -> None:
    """The save path uses a .tmp.<uuid> sidecar then atomic replace.
    After save, no .tmp.* files should leak even on success."""
    save_ledger({"miniwob|foo": HintLedgerEntry(disposition="open")})
    siblings = list(isolated_ledger.parent.iterdir())
    assert isolated_ledger in siblings
    leaked = [p for p in siblings if ".tmp." in p.name]
    assert leaked == [], f"unexpected tmp leakage: {leaked}"


def test_save_ledger_creates_parent_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deep = tmp_path / "deep" / "nested" / "dir"
    monkeypatch.setenv("CH_AUTO_CUBE_DIR", str(deep))
    assert not deep.exists()
    save_ledger({"miniwob|foo": HintLedgerEntry(disposition="open")})
    assert (deep / "hints.json").exists()


# ---------------------------------------------------------------------------
# upsert_entry semantics
# ---------------------------------------------------------------------------


def test_upsert_creates_entry_if_missing(isolated_ledger: Path) -> None:
    entry = upsert_entry(
        "miniwob",
        "click-button",
        session_id="sess-A",
        disposition="steered",
        hint_type="task_specific",
        hint_text="click the bid, not the text.",
    )
    assert entry.disposition == "steered"
    assert entry.last_session == "sess-A"
    loaded = load_ledger()
    assert loaded["miniwob|click-button"].hint_text == "click the bid, not the text."


def test_upsert_preserves_unspecified_fields(isolated_ledger: Path) -> None:
    upsert_entry(
        "miniwob",
        "use-slider",
        session_id="sess-A",
        disposition="steered",
        hint_text="initial hint",
    )
    # Second call only mutates disposition + session; hint_text + hint_type left alone.
    upsert_entry(
        "miniwob",
        "use-slider",
        session_id="sess-B",
        disposition="promoted",
        promoted_to="task_hints",
    )
    loaded = load_ledger()
    e = loaded["miniwob|use-slider"]
    assert e.disposition == "promoted"
    assert e.hint_text == "initial hint"
    assert e.promoted_to == "task_hints"
    assert e.last_session == "sess-B"


def test_upsert_merges_notes_shallowly(isolated_ledger: Path) -> None:
    upsert_entry("miniwob", "x", session_id="A", notes={"confidence": 3, "tries": 1})
    upsert_entry("miniwob", "x", session_id="B", notes={"tries": 2, "cost_usd": 0.05})
    loaded = load_ledger()
    assert loaded["miniwob|x"].notes == {"confidence": 3, "tries": 2, "cost_usd": 0.05}


# ---------------------------------------------------------------------------
# Enum validation (levels 2/3/4)
# ---------------------------------------------------------------------------


def test_upsert_rejects_bad_disposition(isolated_ledger: Path) -> None:
    with pytest.raises(ValueError, match="unknown disposition"):
        upsert_entry("miniwob", "x", session_id="A", disposition="bogus")  # type: ignore[arg-type]


def test_upsert_rejects_bad_hint_type(isolated_ledger: Path) -> None:
    with pytest.raises(ValueError, match="unknown hint_type"):
        upsert_entry("miniwob", "x", session_id="A", hint_type="bogus")  # type: ignore[arg-type]


def test_upsert_rejects_bad_promoted_to(isolated_ledger: Path) -> None:
    with pytest.raises(ValueError, match="unknown promoted_to"):
        upsert_entry("miniwob", "x", session_id="A", promoted_to="bogus")  # type: ignore[arg-type]


def test_taxonomy_constants_match_module() -> None:
    # Pin the level-2/3/4 enums against accidental drift. If a value is
    # added or removed upstream's auto-cube ledger format breaks, this
    # test surfaces it explicitly.
    assert set(HINT_TYPES) == {"clarification", "task_specific", "general_guidance"}
    assert set(DISPOSITIONS) == {
        "steered",
        "promoted",
        "cheat_only",
        "not_a_hint",
        "unsteerable",
        "open",
    }
    assert set(PROMOTED_TO_VALUES) == {
        "task_hints",
        "benchmark_hint_prompt",
        "task_clarification",
        "description_overrides",
        "new_action",
        "system_prompt",
    }


# ---------------------------------------------------------------------------
# On-disk shape (matches upstream auto-cube hinter SKILL.md spec)
# ---------------------------------------------------------------------------


def test_on_disk_shape_matches_auto_cube_spec(isolated_ledger: Path) -> None:
    """Upstream's hinter/SKILL.md documents an exact JSON shape we must
    match so a Mode-C Claude Code session can read this ledger and ship
    upstream PRs from it. Pin the field names + nesting here."""
    upsert_entry(
        "miniwob",
        "use-slider",
        session_id="sess-A",
        disposition="promoted",
        hint_type="task_specific",
        hint_text="Use PageUp.",
        promoted_to="task_hints",
        promotion_pr="https://github.com/example/repo/pull/123",
    )
    raw = json.loads(isolated_ledger.read_text())
    assert "miniwob|use-slider" in raw
    entry = raw["miniwob|use-slider"]
    # Field names match upstream auto_cube/use_cases/hinter/SKILL.md §Cross-session state.
    assert entry["disposition"] == "promoted"
    assert entry["hint_text"] == "Use PageUp."
    assert entry["promoted_to"] == "task_hints"
    assert entry["promotion_pr"] == "https://github.com/example/repo/pull/123"
    assert entry["last_session"] == "sess-A"
