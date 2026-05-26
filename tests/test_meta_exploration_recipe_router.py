"""Tests for L1 Investigator recipe routing (Phase 4 chunk 3)."""

from __future__ import annotations

import pytest

from cube_harness.meta_exploration.recipe_router import (
    RECIPES,
    RouteDecision,
    _RECIPE_DISPATCH_INSTRUCTION,
    classify_recipe,
    format_routing_for_prompt,
    route_iter_tasks,
    summarize_routing,
)


# ---------------------------------------------------------------------------
# Per-task classify_recipe (the core decision tree)
# ---------------------------------------------------------------------------


def test_success_routes_to_hinter_noop() -> None:
    """Passing tasks always route to hinter — hinter defaults to no-op
    on passes per the paper-faithful default."""
    d = classify_recipe(task_id="x", reward=1.0, n_episodes=1)
    assert d.recipe == "hinter"
    assert "success" in d.rationale.lower()


def test_success_above_threshold_routes_to_hinter() -> None:
    d = classify_recipe(task_id="x", reward=0.5, n_episodes=1)
    assert d.recipe == "hinter"


def test_failure_with_high_repeat_routes_to_agent_scaffolding() -> None:
    """≥3 consecutive same-action invocations → agent_scaffolding."""
    d = classify_recipe(task_id="x", reward=0.0, n_episodes=1, max_repeat=3)
    assert d.recipe == "agent_scaffolding"
    assert "loop" in d.rationale.lower()
    assert "3 consecutive" in d.rationale


def test_failure_with_repeat_below_threshold_routes_to_hinter() -> None:
    """max_repeat=2 is NOT enough — must be ≥3."""
    d = classify_recipe(task_id="x", reward=0.0, n_episodes=1, max_repeat=2)
    assert d.recipe == "hinter"


def test_failure_with_high_tokens_per_step_routes_to_profiling() -> None:
    """tokens_per_step above the 50k threshold → profiling."""
    d = classify_recipe(
        task_id="x", reward=0.0, n_episodes=1, tokens_per_step=80_000.0
    )
    assert d.recipe == "profiling"
    assert "quantitative" in d.rationale.lower()


def test_failure_with_normal_tokens_per_step_routes_to_hinter() -> None:
    d = classify_recipe(
        task_id="x", reward=0.0, n_episodes=1, tokens_per_step=10_000.0
    )
    assert d.recipe == "hinter"


def test_scaffolding_wins_over_profiling_when_both_signal() -> None:
    """When a failure is BOTH loopy AND token-heavy, scaffolding wins —
    the loop is the proximate cause; profiling is a downstream symptom."""
    d = classify_recipe(
        task_id="x", reward=0.0, n_episodes=1,
        max_repeat=5, tokens_per_step=100_000.0,
    )
    assert d.recipe == "agent_scaffolding"


def test_default_failure_routes_to_hinter() -> None:
    """The 'I don't know what kind of failure this is' default is hinter —
    paper-faithful with general_blame fallback at SDK time."""
    d = classify_recipe(task_id="x", reward=0.0, n_episodes=1)
    assert d.recipe == "hinter"
    assert "default route" in d.rationale.lower()


# ---------------------------------------------------------------------------
# Custom success threshold
# ---------------------------------------------------------------------------


def test_custom_success_threshold_lowers_pass_bar() -> None:
    """A task with reward 0.3 fails the default 0.5 threshold but passes
    a custom 0.2 — should route to the success branch in that case."""
    d_default = classify_recipe(task_id="x", reward=0.3, n_episodes=1)
    assert d_default.recipe == "hinter"
    assert "default route" in d_default.rationale  # failed branch
    d_custom = classify_recipe(
        task_id="x", reward=0.3, n_episodes=1, success_threshold=0.2
    )
    assert d_custom.recipe == "hinter"
    assert "success" in d_custom.rationale.lower()


# ---------------------------------------------------------------------------
# Iter-level routing
# ---------------------------------------------------------------------------


def test_route_iter_tasks_minimal_inputs() -> None:
    """With only per_task_rewards (no optional signals), every failure
    goes to hinter, every pass goes to hinter no-op."""
    decisions = route_iter_tasks(per_task_rewards={"a": 1.0, "b": 0.0, "c": 0.4})
    assert set(decisions.keys()) == {"a", "b", "c"}
    assert all(d.recipe == "hinter" for d in decisions.values())


def test_route_iter_tasks_with_repeats_routes_one_to_scaffolding() -> None:
    decisions = route_iter_tasks(
        per_task_rewards={"a": 0.0, "b": 0.0, "c": 0.0},
        per_task_max_repeat={"a": 3, "b": 2, "c": 5},
    )
    assert decisions["a"].recipe == "agent_scaffolding"
    assert decisions["b"].recipe == "hinter"           # below threshold
    assert decisions["c"].recipe == "agent_scaffolding"


def test_route_iter_tasks_with_tokens_per_step_routes_one_to_profiling() -> None:
    decisions = route_iter_tasks(
        per_task_rewards={"a": 0.0, "b": 0.0},
        per_task_tokens_per_step={"a": 100_000.0, "b": 20_000.0},
    )
    assert decisions["a"].recipe == "profiling"
    assert decisions["b"].recipe == "hinter"


def test_route_iter_tasks_missing_task_in_optional_dicts() -> None:
    """A task missing from an optional dict just doesn't get that signal —
    routing should still work, no KeyError."""
    decisions = route_iter_tasks(
        per_task_rewards={"a": 0.0, "b": 0.0},
        per_task_max_repeat={"a": 5},   # only 'a' has this signal
    )
    assert decisions["a"].recipe == "agent_scaffolding"
    assert decisions["b"].recipe == "hinter"


# ---------------------------------------------------------------------------
# summarize_routing
# ---------------------------------------------------------------------------


def test_summarize_routing_counts_all_recipes() -> None:
    decisions = {
        "t1": RouteDecision("t1", "hinter", "ok"),
        "t2": RouteDecision("t2", "hinter", "ok"),
        "t3": RouteDecision("t3", "agent_scaffolding", "loop"),
        "t4": RouteDecision("t4", "profiling", "thrash"),
    }
    counts = summarize_routing(decisions)
    assert counts == {
        "hinter": 2, "general_blame": 0,
        "agent_scaffolding": 1, "profiling": 1,
    }


def test_summarize_routing_empty() -> None:
    counts = summarize_routing({})
    assert counts == {r: 0 for r in RECIPES}


# ---------------------------------------------------------------------------
# format_routing_for_prompt (the SDK user-prompt section)
# ---------------------------------------------------------------------------


def test_format_routing_for_prompt_empty_returns_empty_string() -> None:
    assert format_routing_for_prompt({}) == ""


def test_format_routing_for_prompt_includes_header_and_table() -> None:
    decisions = {
        "use-slider": RouteDecision("use-slider", "hinter", "default route"),
        "stuck-loop": RouteDecision("stuck-loop", "agent_scaffolding", "4 repeats"),
    }
    rendered = format_routing_for_prompt(decisions)
    assert "## Investigator recipe routing (L1)" in rendered
    assert "| task_id | recipe | rationale |" in rendered
    assert "`use-slider`" in rendered
    assert "`stuck-loop`" in rendered
    assert "hinter" in rendered
    assert "agent_scaffolding" in rendered


def test_format_routing_for_prompt_sorts_tasks_for_stable_diffs() -> None:
    decisions = {
        "c": RouteDecision("c", "hinter", "x"),
        "a": RouteDecision("a", "hinter", "y"),
        "b": RouteDecision("b", "hinter", "z"),
    }
    rendered = format_routing_for_prompt(decisions)
    a_pos = rendered.index("`a`")
    b_pos = rendered.index("`b`")
    c_pos = rendered.index("`c`")
    assert a_pos < b_pos < c_pos


def test_format_routing_for_prompt_includes_summary_when_nonempty() -> None:
    decisions = {
        "t1": RouteDecision("t1", "hinter", "r1"),
        "t2": RouteDecision("t2", "agent_scaffolding", "r2"),
    }
    rendered = format_routing_for_prompt(decisions)
    assert "Summary:" in rendered
    assert "hinter=1" in rendered
    assert "agent_scaffolding=1" in rendered
    # general_blame and profiling not present in this decision set
    # → should NOT appear in summary
    assert "general_blame=" not in rendered
    assert "profiling=" not in rendered


# ---------------------------------------------------------------------------
# Module export sanity
# ---------------------------------------------------------------------------


def test_recipes_constant_matches_upstream_use_cases() -> None:
    """RECIPES must mirror upstream
    `analyze/investigator/use_cases/<name>/`. If upstream adds or
    removes one, this test fires loudly so we know to re-sync."""
    assert set(RECIPES) == {
        "hinter", "general_blame", "agent_scaffolding", "profiling",
    }


def test_recipe_dispatch_instruction_names_all_4_recipes() -> None:
    """The SDK system-prompt augmentation must mention every recipe so
    the SDK can pick correctly. A missing recipe means the SDK might
    silently misroute it to whatever fallback it picks."""
    for recipe in RECIPES:
        assert f"`{recipe}`" in _RECIPE_DISPATCH_INSTRUCTION, (
            f"recipe {recipe!r} missing from _RECIPE_DISPATCH_INSTRUCTION"
        )
