"""Tests for the meta_exploration promotion module — candidate
detection + re-test gate.

Pure-function unit tests. End-to-end integration with the outer-loop
SDK driver is covered by smoke runs, not here (those run real LLM
episodes).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from cube_harness.meta_exploration.promotion import (
    PromotionCandidate,
    apply_promotion_to_config,
    candidate_from_task_hints_overlap,
    decide_verdict,
    pick_held_out,
    stable_id,
)

# ---------------------------------------------------------------------------
# Mock config (avoids importing the full GennyConfig)
# ---------------------------------------------------------------------------


@dataclass
class _MockAgentConfig:
    benchmark_hint_prompt: str | None = None


# ---------------------------------------------------------------------------
# stable_id
# ---------------------------------------------------------------------------


def test_stable_id_deterministic() -> None:
    a = stable_id("miniwob", "benchmark_hint_prompt", "  hello\n")
    b = stable_id("miniwob", "benchmark_hint_prompt", "hello")
    assert a == b, "leading/trailing whitespace must be stripped before hashing"


def test_stable_id_differs_by_benchmark() -> None:
    a = stable_id("miniwob", "benchmark_hint_prompt", "x")
    b = stable_id("terminal_bench_2", "benchmark_hint_prompt", "x")
    assert a != b


def test_stable_id_differs_by_rung() -> None:
    a = stable_id("miniwob", "benchmark_hint_prompt", "x")
    b = stable_id("miniwob", "task_clarification", "x")
    assert a != b


# ---------------------------------------------------------------------------
# candidate_from_task_hints_overlap — the v1 detector
# ---------------------------------------------------------------------------


def test_no_task_hints_no_candidates() -> None:
    out = candidate_from_task_hints_overlap(
        benchmark="miniwob",
        task_hints_after={},
        per_task_rewards={"a": 0.0},
    )
    assert out == []


def test_below_threshold_no_candidate() -> None:
    """Same hint in 2 tasks, threshold=3 → no candidate."""
    out = candidate_from_task_hints_overlap(
        benchmark="miniwob",
        task_hints_after={"a": "use PageUp", "b": "use PageUp"},
        per_task_rewards={"a": 0.0, "b": 0.0},
        min_observations=3,
    )
    assert out == []


def test_at_threshold_yields_candidate() -> None:
    out = candidate_from_task_hints_overlap(
        benchmark="terminal_bench_2",
        task_hints_after={"a": "use timeout=600", "b": "use timeout=600", "c": "use timeout=600"},
        per_task_rewards={"a": 0.0, "b": 0.0, "c": 0.0},
        min_observations=3,
    )
    assert len(out) == 1
    cand = out[0]
    assert cand.rung == "benchmark_hint_prompt"
    assert cand.hint_text == "use timeout=600"
    assert sorted(cand.affected_task_ids) == ["a", "b", "c"]
    assert cand.candidate_id.startswith("terminal_bench_2:benchmark_hint_prompt:")


def test_unchanged_hints_dont_recandidate() -> None:
    out = candidate_from_task_hints_overlap(
        benchmark="miniwob",
        task_hints_after={"a": "X", "b": "X", "c": "X"},
        task_hints_before={"a": "X", "b": "X", "c": "X"},
        per_task_rewards={"a": 0.0, "b": 0.0, "c": 0.0},
        min_observations=3,
    )
    assert out == []


def test_partially_changed_hints_count_only_new() -> None:
    out = candidate_from_task_hints_overlap(
        benchmark="miniwob",
        task_hints_after={"a": "X", "b": "X", "c": "X"},
        task_hints_before={"a": "X"},  # 'a' unchanged; b,c new — only 2 fresh
        per_task_rewards={"a": 0.0, "b": 0.0, "c": 0.0},
        min_observations=3,
    )
    assert out == []


def test_empty_text_skipped() -> None:
    out = candidate_from_task_hints_overlap(
        benchmark="miniwob",
        task_hints_after={"a": "  ", "b": "", "c": "real"},
        per_task_rewards={"a": 0.0, "b": 0.0, "c": 0.0},
        min_observations=1,
    )
    assert len(out) == 1
    assert out[0].hint_text == "real"
    assert out[0].affected_task_ids == ["c"]


def test_candidate_ids_stable_across_calls() -> None:
    a = candidate_from_task_hints_overlap(
        benchmark="miniwob",
        task_hints_after={"a": "same text", "b": "same text", "c": "same text"},
        per_task_rewards={"a": 0.0, "b": 0.0, "c": 0.0},
        min_observations=3,
    )[0]
    b = candidate_from_task_hints_overlap(
        benchmark="miniwob",
        task_hints_after={"x": "same text", "y": "same text", "z": "same text"},
        per_task_rewards={"x": 0.0, "y": 0.0, "z": 0.0},
        min_observations=3,
    )[0]
    assert a.candidate_id == b.candidate_id


# ---------------------------------------------------------------------------
# pick_held_out
# ---------------------------------------------------------------------------


def test_pick_held_out_excludes_affected() -> None:
    cand = PromotionCandidate(
        candidate_id="x",
        rung="benchmark_hint_prompt",
        hint_text="t",
        source_keys=[],
        affected_task_ids=["a", "b"],
        rationale="r",
    )
    held = pick_held_out(
        candidate=cand,
        per_task_rewards={"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0},
        n_held_out=2,
        seed=42,
    )
    assert set(held).isdisjoint({"a", "b"})
    assert len(held) == 2


def test_pick_held_out_empty_pool_returns_empty() -> None:
    cand = PromotionCandidate(
        candidate_id="x",
        rung="benchmark_hint_prompt",
        hint_text="t",
        source_keys=[],
        affected_task_ids=["a", "b"],
        rationale="r",
    )
    held = pick_held_out(
        candidate=cand,
        per_task_rewards={"a": 0.0, "b": 0.0},
        n_held_out=3,
        seed=42,
    )
    assert held == []


def test_pick_held_out_smaller_pool_returns_all() -> None:
    cand = PromotionCandidate(
        candidate_id="x",
        rung="benchmark_hint_prompt",
        hint_text="t",
        source_keys=[],
        affected_task_ids=["a"],
        rationale="r",
    )
    held = pick_held_out(
        candidate=cand,
        per_task_rewards={"a": 0.0, "b": 0.0, "c": 0.0},
        n_held_out=5,
        seed=42,
    )
    assert set(held) == {"b", "c"}


def test_pick_held_out_seed_reproducible() -> None:
    cand = PromotionCandidate(
        candidate_id="x",
        rung="benchmark_hint_prompt",
        hint_text="t",
        source_keys=[],
        affected_task_ids=["a"],
        rationale="r",
    )
    pool = {f"task{i}": 0.0 for i in range(20)}
    pool["a"] = 0.0
    h1 = pick_held_out(candidate=cand, per_task_rewards=pool, n_held_out=3, seed=1)
    h2 = pick_held_out(candidate=cand, per_task_rewards=pool, n_held_out=3, seed=1)
    h3 = pick_held_out(candidate=cand, per_task_rewards=pool, n_held_out=3, seed=2)
    assert h1 == h2
    assert h1 != h3


# ---------------------------------------------------------------------------
# decide_verdict — the gate logic
# ---------------------------------------------------------------------------


def _cand(affected: list[str]) -> PromotionCandidate:
    return PromotionCandidate(
        candidate_id="cand-x",
        rung="benchmark_hint_prompt",
        hint_text="t",
        source_keys=[],
        affected_task_ids=affected,
        rationale="r",
    )


def test_verdict_ok_when_no_regressions() -> None:
    v = decide_verdict(
        candidate=_cand(["a", "b"]),
        affected_pre={"a": 1.0, "b": 0.0},
        affected_post={"a": 1.0, "b": 1.0},
        held_out_pre={"c": 1.0, "d": 0.0},
        held_out_post={"c": 1.0, "d": 0.0},
    )
    assert v.outcome == "ok"
    assert "no regressions" in v.reason.lower()


def test_verdict_affected_regressed() -> None:
    v = decide_verdict(
        candidate=_cand(["a", "b"]),
        affected_pre={"a": 1.0, "b": 1.0},
        affected_post={"a": 0.0, "b": 1.0},
        held_out_pre={"c": 1.0},
        held_out_post={"c": 1.0},
    )
    assert v.outcome == "affected_regressed"
    assert "'a'" in v.reason


def test_verdict_held_out_regressed() -> None:
    v = decide_verdict(
        candidate=_cand(["a"]),
        affected_pre={"a": 0.0},
        affected_post={"a": 1.0},
        held_out_pre={"c": 1.0, "d": 1.0},
        held_out_post={"c": 1.0, "d": 0.0},
    )
    assert v.outcome == "held_out_regressed"
    assert "'d'" in v.reason


def test_verdict_affected_regression_takes_precedence() -> None:
    v = decide_verdict(
        candidate=_cand(["a"]),
        affected_pre={"a": 1.0},
        affected_post={"a": 0.0},
        held_out_pre={"c": 1.0},
        held_out_post={"c": 0.0},
    )
    assert v.outcome == "affected_regressed"


def test_verdict_fail_to_fail_not_a_regression() -> None:
    v = decide_verdict(
        candidate=_cand(["a"]),
        affected_pre={"a": 0.0},
        affected_post={"a": 0.0},
        held_out_pre={"c": 0.0},
        held_out_post={"c": 0.0},
    )
    assert v.outcome == "ok"


# ---------------------------------------------------------------------------
# apply_promotion_to_config
# ---------------------------------------------------------------------------


def test_apply_benchmark_hint_prompt_sets_field_when_empty() -> None:
    cfg = _MockAgentConfig(benchmark_hint_prompt=None)
    cand = _cand(["a"])
    apply_promotion_to_config(cfg, cand)
    assert cfg.benchmark_hint_prompt == "t"


def test_apply_benchmark_hint_prompt_appends_when_already_set() -> None:
    cfg = _MockAgentConfig(benchmark_hint_prompt="existing prompt")
    cand = _cand(["a"])
    apply_promotion_to_config(cfg, cand)
    assert "existing prompt" in cfg.benchmark_hint_prompt
    assert "t" in cfg.benchmark_hint_prompt
    assert cfg.benchmark_hint_prompt.count("\n\n") >= 1


def test_apply_rejects_unimplemented_rung() -> None:
    cfg = _MockAgentConfig()
    cand = PromotionCandidate(
        candidate_id="x",
        rung="task_clarification",
        hint_text="t",
        source_keys=[],
        affected_task_ids=["a"],
        rationale="r",
    )
    with pytest.raises(NotImplementedError, match="task_clarification"):
        apply_promotion_to_config(cfg, cand)
