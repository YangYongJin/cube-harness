"""Cross-session auto-cube ledger — ``~/auto_cube/hints.json``.

Discipline-overlay record that any Auto-CUBE use case (``hinter`` / future
``meta_exploration``) reads and writes to track per-task hint decisions
across sessions. Lives **alongside** (not replacing) the cube's
``benchmark_clarifications.py`` sidecar: that file is the runtime prompt
injection format that flows through ``GennyConfig.with_benchmark_clarifications``;
this ledger is the discipline metadata layer recording **what was decided
about each task's hint candidates** across sessions, sweeps, and modes.

The ledger encodes three of the four levels of Auto-CUBE's hinter taxonomy:

  * Level 2 — ``hint_type`` (per-hint)
      ``clarification | task_specific | general_guidance``
      *Why* the hint was emitted (Investigator-recipe output;
      see ``cube_harness.analyze.investigator.use_cases.hinter``).

  * Level 3 — ``disposition`` (per-task)
      ``steered | promoted | cheat_only | not_a_hint | unsteerable``
      *What we did* with the hint after observing its effect.

  * Level 4 — ``promoted_to`` (per-task)
      ``task_hints | benchmark_hint_prompt | task_clarification |
       description_overrides | new_action | system_prompt | null``
      The destination rung on the regularization ladder when promoted
      (corresponds to ``GennyConfig`` fields + cube/tool PR targets;
      see ``auto_cube/use_cases/hinter/SKILL.md`` §regularization ladder).

Level 1 (Investigator recipe selection: hinter / general_blame /
agent_scaffolding / profiling) lives at trajectory-dispatch time in the
outer-loop SDK driver, not in this ledger.

File format (JSON; one entry per ``<cube>|<task_id>``):

    {
      "miniwob|use-slider": {
        "disposition": "promoted",
        "hint_type": "task_specific",
        "hint_text": "Use PageUp/PageDown for large slider gaps; ArrowUp is 1-per-press.",
        "promoted_to": "task_hints",
        "promotion_pr": null,
        "last_session": "session-2026-05-26T07-25-01Z-abc123"
      }
    }

Default location: ``~/auto_cube/hints.json`` (matches
``auto_cube/README.md``'s session-root layout). Override via the
``CH_AUTO_CUBE_DIR`` env var (used by tests to point at a tmp dir;
matches the ``CH_EXP_DIR`` convention).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Auto-CUBE hinter taxonomy types (levels 2, 3, 4)
# ---------------------------------------------------------------------------

# Level 2 — per-hint classification (Investigator hinter-recipe output).
HintType = Literal["clarification", "task_specific", "general_guidance"]
HINT_TYPES: tuple[HintType, ...] = (
    "clarification",
    "task_specific",
    "general_guidance",
)

# Level 3 — per-task outer-loop disposition.
Disposition = Literal[
    "steered",  # low-reg hint flipped this task to success
    "promoted",  # the steer generalized + shipped as a higher-reg artefact
    "cheat_only",  # steer works but is genuinely task-specific (no PR)
    "not_a_hint",  # failure root cause is a real bug (route to debug)
    "unsteerable",  # capability ceiling; no hint helps
    "open",  # initial state — no decision recorded yet this session
]
DISPOSITIONS: tuple[Disposition, ...] = (
    "steered",
    "promoted",
    "cheat_only",
    "not_a_hint",
    "unsteerable",
    "open",
)

# Level 4 — where a promoted hint landed on the regularization ladder.
# `None` until/unless disposition becomes "promoted".
PromotedTo = Literal[
    "task_hints",  # low — local task hint cheat
    "benchmark_hint_prompt",  # high — benchmark-wide orientation
    "task_clarification",  # high — per-task wording fix
    "description_overrides",  # high — action wording override
    "new_action",  # high — added/changed action surface
    "system_prompt",  # high — generalist system prompt change
]
PROMOTED_TO_VALUES: tuple[PromotedTo, ...] = (
    "task_hints",
    "benchmark_hint_prompt",
    "task_clarification",
    "description_overrides",
    "new_action",
    "system_prompt",
)


# ---------------------------------------------------------------------------
# Ledger entry — what we record per (cube, task_id)
# ---------------------------------------------------------------------------


@dataclass
class HintLedgerEntry:
    """One row in ``~/auto_cube/hints.json``, keyed by ``<cube>|<task_id>``."""

    disposition: Disposition = "open"
    hint_type: HintType | None = None
    hint_text: str = ""
    promoted_to: PromotedTo | None = None
    promotion_pr: str | None = None
    last_session: str = ""
    # Free-form metadata bag (cost, # re-test rounds, episode_id chain, etc.).
    # Kept loosely typed so the ledger doesn't constrain future Investigator
    # output. Reserved keys: "confidence" (0-5 int from TaskHint), "rationale".
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Strip None values for a cleaner on-disk shape; readers default-init
        # missing fields, so this round-trips losslessly.
        return {k: v for k, v in d.items() if v not in (None, "", {})}

    @classmethod
    def from_dict(cls, raw: dict) -> "HintLedgerEntry":
        if not isinstance(raw, dict):
            raise ValueError(f"ledger entry must be dict, got {type(raw).__name__}")
        return cls(
            disposition=raw.get("disposition", "open"),
            hint_type=raw.get("hint_type"),
            hint_text=raw.get("hint_text", ""),
            promoted_to=raw.get("promoted_to"),
            promotion_pr=raw.get("promotion_pr"),
            last_session=raw.get("last_session", ""),
            notes=dict(raw.get("notes", {})),
        )


def make_ledger_key(cube: str, task_id: str) -> str:
    """The canonical ``<cube>|<task_id>`` key used by ``hints.json``."""
    if not cube or not task_id:
        raise ValueError(f"ledger key requires non-empty cube + task_id; got ({cube!r}, {task_id!r})")
    if "|" in cube or "|" in task_id:
        raise ValueError(f"'|' is a reserved separator; got cube={cube!r} task_id={task_id!r}")
    return f"{cube}|{task_id}"


# ---------------------------------------------------------------------------
# Ledger I/O
# ---------------------------------------------------------------------------


def auto_cube_dir() -> Path:
    """Where ``hints.json`` (and future ``coverage.json``) lives.

    ``CH_AUTO_CUBE_DIR`` env var wins (used by tests). Default
    ``~/auto_cube/`` matches Auto-CUBE's session-root layout per
    ``cube_harness/auto_cube/README.md``.
    """
    override = os.environ.get("CH_AUTO_CUBE_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / "auto_cube"


def ledger_path() -> Path:
    """Absolute path to the hints ledger JSON file."""
    return auto_cube_dir() / "hints.json"


def load_ledger() -> dict[str, HintLedgerEntry]:
    """Read the ledger file; missing/empty file → empty dict.

    Malformed entries are dropped with a warning rather than crashing —
    a corrupt ledger should never block the meta-loop from running.
    """
    path = ledger_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("auto_cube ledger %s unreadable (%s); treating as empty", path, e)
        return {}
    if not isinstance(raw, dict):
        logger.warning("auto_cube ledger %s root must be dict; got %s; treating as empty", path, type(raw).__name__)
        return {}
    out: dict[str, HintLedgerEntry] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or "|" not in key:
            logger.warning("auto_cube ledger %s: dropping malformed key %r", path, key)
            continue
        try:
            out[key] = HintLedgerEntry.from_dict(value)
        except ValueError as e:
            logger.warning("auto_cube ledger %s: dropping malformed entry %r (%s)", path, key, e)
    return out


def save_ledger(entries: dict[str, HintLedgerEntry]) -> None:
    """Atomically write the ledger to disk (tmp file + replace).

    Safe under concurrent Ray workers — the rename is atomic on POSIX
    filesystems. Last writer wins on collision; we don't merge here
    because the driver is the only writer in practice.
    """
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: entry.to_dict() for key, entry in entries.items()}
    tmp = path.with_suffix(f".tmp.{uuid.uuid4().hex[:8]}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    tmp.replace(path)


def upsert_entry(
    cube: str,
    task_id: str,
    *,
    session_id: str,
    disposition: Disposition | None = None,
    hint_type: HintType | None = None,
    hint_text: str | None = None,
    promoted_to: PromotedTo | None = None,
    promotion_pr: str | None = None,
    notes: dict | None = None,
) -> HintLedgerEntry:
    """Update one ledger entry in-place. Idempotent.

    Reads the whole ledger, mutates the named row, writes it back. Fields
    left as ``None`` preserve the existing value (except ``notes`` which
    merges shallowly when given). ``session_id`` is always written into
    ``last_session`` so downstream tooling can see which run touched this
    task most recently.
    """
    ledger = load_ledger()
    key = make_ledger_key(cube, task_id)
    entry = ledger.get(key, HintLedgerEntry())
    if disposition is not None:
        if disposition not in DISPOSITIONS:
            raise ValueError(f"unknown disposition {disposition!r}; expected one of {DISPOSITIONS}")
        entry.disposition = disposition
    if hint_type is not None:
        if hint_type not in HINT_TYPES:
            raise ValueError(f"unknown hint_type {hint_type!r}; expected one of {HINT_TYPES}")
        entry.hint_type = hint_type
    if hint_text is not None:
        entry.hint_text = hint_text
    if promoted_to is not None:
        if promoted_to not in PROMOTED_TO_VALUES:
            raise ValueError(f"unknown promoted_to {promoted_to!r}; expected one of {PROMOTED_TO_VALUES}")
        entry.promoted_to = promoted_to
    if promotion_pr is not None:
        entry.promotion_pr = promotion_pr
    if notes:
        entry.notes = {**entry.notes, **notes}
    entry.last_session = session_id
    ledger[key] = entry
    save_ledger(ledger)
    return entry


__all__ = [
    "DISPOSITIONS",
    "Disposition",
    "HintLedgerEntry",
    "HintType",
    "HINT_TYPES",
    "PROMOTED_TO_VALUES",
    "PromotedTo",
    "auto_cube_dir",
    "ledger_path",
    "load_ledger",
    "make_ledger_key",
    "save_ledger",
    "upsert_entry",
]
