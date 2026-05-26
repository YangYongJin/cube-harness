"""Tests for the meta_exploration EpisodeConfig planner (Pivot 4).

Pure-function unit tests over the heuristic policy. The planner's
correctness is the decision tree — each rule gets a test that names
which rule fires + which alternatives get recorded.
"""

from __future__ import annotations

import pytest

from cube_harness.meta_exploration.ledger import HintLedgerEntry
from cube_harness.meta_exploration.planner import (
    DEFAULT,
    EPISODE_CONFIG_MENU,
    EXPLORATORY_BREADTH,
    PROFILING_DIAGNOSIS,
    LONG_BUILD_TASK,
    PlannerDecision,
    PlannerState,
    RETEST_PROMOTION,
    SCAFFOLDING_DIAGNOSIS,
    STRONG_MODEL,
    pick_episode_config,
    plan_iter,
    summarize_plan,
)


# ---------------------------------------------------------------------------
# State builder
# ---------------------------------------------------------------------------


def _state(
    cube: str = "terminal_bench_2",
    *,
    ledger: dict[str, HintLedgerEntry] | None = None,
    recent_rewards: dict[str, list[float]] | None = None,
    budget_remaining_usd: float = 1e9,
) -> PlannerState:
    return PlannerState(
        cube=cube,
        ledger=ledger or {},
        recent_rewards=recent_rewards or {},
        budget_remaining_usd=budget_remaining_usd,
    )


# ---------------------------------------------------------------------------
# Rule 2 — brand new task → DEFAULT
# ---------------------------------------------------------------------------


def test_brand_new_task_routes_to_default() -> None:
    d = pick_episode_config("foo", _state())
    assert d.config_name == "DEFAULT"
    assert d.config == DEFAULT
    assert "first attempt" in d.rationale.lower()


def test_default_alternatives_include_breadth_and_strong_model() -> None:
    d = pick_episode_config("foo", _state())
    names = {alt[0] for alt in d.alternatives_considered}
    # The default-pick should record that breadth+strong-model were
    # considered and rejected because we don't have prior signal.
    assert "EXPLORATORY_BREADTH" in names
    assert "STRONG_MODEL" in names


# ---------------------------------------------------------------------------
# Rule 5 — one fail with default → EXPLORATORY_BREADTH (no special notes)
# ---------------------------------------------------------------------------


def test_one_fail_default_routes_to_exploratory_breadth() -> None:
    d = pick_episode_config(
        "foo",
        _state(recent_rewards={"foo": [0.0]}),
    )
    assert d.config_name == "EXPLORATORY_BREADTH"
    assert d.config == EXPLORATORY_BREADTH
    assert "almost did" in d.rationale.lower() or "broaden" in d.rationale.lower()


# ---------------------------------------------------------------------------
# Rule 5 + loop pattern note → SCAFFOLDING_DIAGNOSIS
# ---------------------------------------------------------------------------


def test_loop_pattern_note_routes_to_scaffolding_diagnosis() -> None:
    entry = HintLedgerEntry(notes={"loop_pattern_suspected": True})
    d = pick_episode_config(
        "foo",
        _state(
            ledger={"terminal_bench_2|foo": entry},
            recent_rewards={"foo": [0.0]},
        ),
    )
    assert d.config_name == "SCAFFOLDING_DIAGNOSIS"
    assert "loop" in d.rationale.lower()


# ---------------------------------------------------------------------------
# Rule 5 + build-timeout note → LONG_BUILD_TASK
# ---------------------------------------------------------------------------


def test_build_timeout_note_routes_to_long_build_task() -> None:
    entry = HintLedgerEntry(notes={"build_timeout_suspected": True})
    d = pick_episode_config(
        "build-cython-ext",
        _state(
            ledger={"terminal_bench_2|build-cython-ext": entry},
            recent_rewards={"build-cython-ext": [0.0]},
        ),
    )
    assert d.config_name == "LONG_BUILD_TASK"
    assert "600" in d.rationale or "timeout" in d.rationale.lower()


# ---------------------------------------------------------------------------
# Rule 3 — near-miss → STRONG_MODEL
# ---------------------------------------------------------------------------


def test_near_miss_routes_to_strong_model() -> None:
    d = pick_episode_config(
        "foo",
        _state(recent_rewards={"foo": [0.3]}),
    )
    assert d.config_name == "STRONG_MODEL"
    assert d.config == STRONG_MODEL
    assert "capability" in d.rationale.lower() or "stronger" in d.rationale.lower()


def test_near_miss_after_one_fail_still_picks_strong_model() -> None:
    """Order of recent_rewards shouldn't matter — any near-miss wins."""
    d = pick_episode_config(
        "foo",
        _state(recent_rewards={"foo": [0.0, 0.3, 0.0]}),
    )
    assert d.config_name == "STRONG_MODEL"


# ---------------------------------------------------------------------------
# Rule 4 — stuck streak → PROFILING_DIAGNOSIS
# ---------------------------------------------------------------------------


def test_stuck_failure_streak_routes_to_profiling_diagnosis() -> None:
    d = pick_episode_config(
        "foo",
        _state(recent_rewards={"foo": [0.0, 0.0, 0.0]}),
    )
    assert d.config_name == "PROFILING_DIAGNOSIS"
    assert "unsteerable" in d.rationale.lower()


def test_two_fails_not_yet_stuck_routes_to_breadth() -> None:
    """Streak threshold is 3 — 2 fails should still try breadth."""
    d = pick_episode_config(
        "foo",
        _state(recent_rewards={"foo": [0.0, 0.0]}),
    )
    assert d.config_name == "EXPLORATORY_BREADTH"


def test_a_pass_in_the_streak_blocks_unsteerable_routing() -> None:
    """If any reward in window is a pass, we're not stuck."""
    d = pick_episode_config(
        "foo",
        _state(recent_rewards={"foo": [0.0, 1.0, 0.0, 0.0]}),
    )
    # Has passes, so not stuck-streak (rule 4 won't fire); should
    # fall to rule 5's breadth (still failing in recent attempts).
    # But actually any(r >= 0.5) is True, so rule 5 doesn't fire either.
    # Falls through to rule 8 fallback → DEFAULT.
    assert d.config_name == "DEFAULT"


# ---------------------------------------------------------------------------
# Rule 1 — cheat_only + 2+ attempts → RETEST_PROMOTION
# ---------------------------------------------------------------------------


def test_cheat_only_2plus_attempts_triggers_retest() -> None:
    entry = HintLedgerEntry(disposition="cheat_only", hint_text="t")
    d = pick_episode_config(
        "foo",
        _state(
            ledger={"terminal_bench_2|foo": entry},
            recent_rewards={"foo": [1.0, 1.0]},
        ),
    )
    assert d.config_name == "RETEST_PROMOTION"
    assert d.config == RETEST_PROMOTION
    assert d.config.apply_promotion is True


def test_cheat_only_1_attempt_does_not_yet_retest() -> None:
    """Need 2+ attempts before triggering retest."""
    entry = HintLedgerEntry(disposition="cheat_only", hint_text="t")
    d = pick_episode_config(
        "foo",
        _state(
            ledger={"terminal_bench_2|foo": entry},
            recent_rewards={"foo": [1.0]},
        ),
    )
    # Has a pass in recent_rewards → rule 5's no-signal branch doesn't
    # fire; falls to fallback DEFAULT.
    assert d.config_name == "DEFAULT"


# ---------------------------------------------------------------------------
# Budget downgrade
# ---------------------------------------------------------------------------


def test_budget_too_low_for_strong_model_downgrades_to_cap() -> None:
    d = pick_episode_config(
        "foo",
        _state(
            recent_rewards={"foo": [0.3]},  # would normally pick STRONG_MODEL
            budget_remaining_usd=0.01,       # too low for $0.50
        ),
    )
    assert d.config_name == "PROFILING_DIAGNOSIS"
    assert "DOWNGRADED" in d.rationale
    assert "STRONG_MODEL" in d.rationale
    # The original (rejected) STRONG_MODEL should be in alternatives.
    alt_names = {alt[0] for alt in d.alternatives_considered}
    assert "STRONG_MODEL" in alt_names


def test_budget_high_for_default_is_no_downgrade() -> None:
    """DEFAULT is cheap ($0.05); $1 budget is plenty."""
    d = pick_episode_config(
        "foo",
        _state(budget_remaining_usd=1.0),  # plenty for default
    )
    assert d.config_name == "DEFAULT"
    assert "DOWNGRADED" not in d.rationale


# ---------------------------------------------------------------------------
# plan_iter
# ---------------------------------------------------------------------------


def test_plan_iter_fills_task_id() -> None:
    plan = plan_iter(["a", "b", "c"], _state())
    assert set(plan.keys()) == {"a", "b", "c"}
    for tid, d in plan.items():
        assert d.task_id == tid


def test_plan_iter_preserves_order() -> None:
    plan = plan_iter(["zebra", "apple", "mango"], _state())
    assert list(plan.keys()) == ["zebra", "apple", "mango"]


def test_plan_iter_routes_each_task_independently() -> None:
    """Different per-task histories should yield different configs."""
    state = _state(
        recent_rewards={
            "fresh": [],            # → DEFAULT
            "near": [0.3],          # → STRONG_MODEL
            "stuck": [0.0, 0.0, 0.0],  # → PROFILING_DIAGNOSIS
        }
    )
    plan = plan_iter(["fresh", "near", "stuck"], state)
    assert plan["fresh"].config_name == "DEFAULT"
    assert plan["near"].config_name == "STRONG_MODEL"
    assert plan["stuck"].config_name == "PROFILING_DIAGNOSIS"


# ---------------------------------------------------------------------------
# summarize_plan
# ---------------------------------------------------------------------------


def test_summarize_plan_counts_by_config_name() -> None:
    state = _state(
        recent_rewards={
            "a": [], "b": [],         # both DEFAULT
            "c": [0.3],               # STRONG_MODEL
            "d": [0.0, 0.0, 0.0],     # PROFILING_DIAGNOSIS
        }
    )
    plan = plan_iter(["a", "b", "c", "d"], state)
    counts = summarize_plan(plan)
    assert counts["DEFAULT"] == 2
    assert counts["STRONG_MODEL"] == 1
    assert counts["PROFILING_DIAGNOSIS"] == 1
    # Sum equals total tasks
    assert sum(counts.values()) == 4


def test_summarize_plan_empty() -> None:
    assert summarize_plan({}) == {}


# ---------------------------------------------------------------------------
# Menu integrity
# ---------------------------------------------------------------------------


def test_every_menu_entry_is_unique_episode_config() -> None:
    """No two named configs should be identical — that's a menu bug."""
    seen = set()
    for name, cfg in EPISODE_CONFIG_MENU.items():
        key = (cfg.model, cfg.k_candidates, cfg.k_plans, cfg.perturbations,
               cfg.bash_default_timeout, cfg.investigator_recipe,
               cfg.enable_refiner, cfg.apply_promotion)
        assert key not in seen, f"{name} duplicates another menu entry"
        seen.add(key)


def test_menu_includes_all_documented_configs() -> None:
    """SKILL.md promises these configs exist. Pin against drift."""
    required = {
        "DEFAULT", "EXPLORATORY_BREADTH", "STRONG_MODEL",
        "REFINER_ENABLED", "RETEST_PROMOTION", "PROFILING_DIAGNOSIS",
        "LONG_BUILD_TASK", "SCAFFOLDING_DIAGNOSIS",
    }
    assert required.issubset(EPISODE_CONFIG_MENU.keys())


def test_default_config_is_zero_cost_baseline() -> None:
    """DEFAULT must be the cheapest baseline — no fancy knobs flipped."""
    assert DEFAULT.k_candidates == 1
    assert DEFAULT.k_plans == 1
    assert DEFAULT.perturbations == ()
    assert DEFAULT.enable_refiner is False
    assert DEFAULT.apply_promotion is False
    assert DEFAULT.investigator_recipe is None
