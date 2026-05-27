"""Tests for the outer-loop SDK driver (Pivot 7 — v1 wires A + B + F)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cube_harness.auto_cube.options import (
    COMBINED,
    HINTER_ONLY,
    META_EXPLORATION_ONLY,
    WEAK_NOOP,
    AutoCubeOptions,
    IncoherentOptionsError,
)
from cube_harness.auto_cube.python_driver import (
    make_session_id,
    run_outer_loop,
    stage_a_plan_iter,
    stage_b_launch_episode,
    stage_c_dispatch_investigator,
    stage_d_author_hints,
    stage_e_phase2_promotion,
    stage_f_write_ledger,
)
from cube_harness.meta_exploration.planner import BASELINE, EpisodeConfig, PlannerDecision


@pytest.fixture
def isolated_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CH_AUTO_CUBE_DIR", str(tmp_path))
    return tmp_path / "hints.json"


# ---------------------------------------------------------------------------
# make_session_id
# ---------------------------------------------------------------------------


def test_session_id_format() -> None:
    sid = make_session_id("foo")
    parts = sid.split("-")
    assert parts[0] == "foo"
    # Timestamp segment ends with Z (UTC).
    assert parts[1].endswith("Z")
    # Nonce is 6 hex chars.
    assert len(parts[2]) == 6


def test_session_id_unique_per_call() -> None:
    assert make_session_id() != make_session_id()


# ---------------------------------------------------------------------------
# Stage A — planning
# ---------------------------------------------------------------------------


def test_stage_a_returns_baseline_when_config_policy_off(isolated_ledger) -> None:
    plan = stage_a_plan_iter(
        cube="terminalbench2",
        task_ids=["a", "b"],
        opts=WEAK_NOOP,  # enable_config_policy=False
        recent_rewards={"a": [], "b": []},
        budget_remaining_usd=10.0,
    )
    assert set(plan.keys()) == {"a", "b"}
    for tid, d in plan.items():
        assert d.config_name == "BASELINE"
        assert d.config == BASELINE
        assert "enable_config_policy=False" in d.rationale


def test_stage_a_invokes_planner_when_config_policy_on(isolated_ledger) -> None:
    plan = stage_a_plan_iter(
        cube="terminalbench2",
        task_ids=["fresh", "near"],
        opts=META_EXPLORATION_ONLY,  # config_policy on, escalation allowed
        recent_rewards={"fresh": [], "near": [0.3]},
        budget_remaining_usd=10.0,
    )
    # Fresh task → BASELINE (rule 2); near-miss → ESCALATE_MODEL (rule 3).
    assert plan["fresh"].config_name == "BASELINE"
    assert plan["near"].config_name == "ESCALATE_MODEL"


# ---------------------------------------------------------------------------
# Stage B — episode launch + honest-split assertion
# ---------------------------------------------------------------------------


def test_stage_b_fires_honest_split_assertion_on_promotion_with_wrong_model() -> None:
    """The load-bearing guard: re-test episode using a non-inference
    model must raise BEFORE the runner is invoked."""
    opts = AutoCubeOptions(inference_model="azure/gpt-5-mini")
    bad_cfg = EpisodeConfig(model="azure/gpt-5", apply_promotion=True)
    decision = PlannerDecision(
        task_id="x",
        config_name="RETEST",
        config=bad_cfg,
        rationale="r",
        alternatives_considered=(),
        expected_information_value="",
    )

    def runner_should_not_be_called(task_id, cfg):  # noqa: ARG001
        pytest.fail("runner invoked despite assertion failure")

    with pytest.raises(IncoherentOptionsError, match="re-test gate violation"):
        stage_b_launch_episode(
            opts=opts,
            task_id="x",
            decision=decision,
            benchmark_config=None,
            runner=runner_should_not_be_called,
        )


def test_stage_b_passes_when_promotion_uses_inference_model() -> None:
    opts = AutoCubeOptions(inference_model="azure/gpt-5-mini")
    ok_cfg = EpisodeConfig(model="azure/gpt-5-mini", apply_promotion=True)
    decision = PlannerDecision(
        task_id="x",
        config_name="RETEST",
        config=ok_cfg,
        rationale="r",
        alternatives_considered=(),
        expected_information_value="",
    )
    reward = stage_b_launch_episode(
        opts=opts,
        task_id="x",
        decision=decision,
        benchmark_config=None,
        runner=lambda tid, cfg: 1.0,
    )
    assert reward == 1.0


def test_stage_b_passes_for_non_promotion_episode_with_strong_model() -> None:
    """Training-time episodes (apply_promotion=False) can use any model."""
    opts = AutoCubeOptions(inference_model="azure/gpt-5-mini")
    cfg = EpisodeConfig(model="azure/gpt-5", apply_promotion=False)
    decision = PlannerDecision(
        task_id="x",
        config_name="ESCALATE_MODEL",
        config=cfg,
        rationale="r",
        alternatives_considered=(),
        expected_information_value="",
    )
    reward = stage_b_launch_episode(
        opts=opts,
        task_id="x",
        decision=decision,
        benchmark_config=None,
        runner=lambda tid, cfg: 0.5,
    )
    assert reward == 0.5


def test_stage_b_no_runner_raises_with_actionable_message() -> None:
    """Default (no runner) raises NotImplementedError with contract hint."""
    opts = AutoCubeOptions()
    decision = PlannerDecision(
        task_id="x",
        config_name="BASELINE",
        config=BASELINE,
        rationale="r",
        alternatives_considered=(),
        expected_information_value="",
    )
    with pytest.raises(NotImplementedError, match="runner"):
        stage_b_launch_episode(
            opts=opts,
            task_id="x",
            decision=decision,
            benchmark_config=None,
        )


# ---------------------------------------------------------------------------
# Stage C / D / E — stubs raise NotImplementedError with contract
# ---------------------------------------------------------------------------


def test_stage_c_stubbed_with_contract() -> None:
    with pytest.raises(NotImplementedError, match="Contract"):
        stage_c_dispatch_investigator(
            opts=WEAK_NOOP,
            trajectory=None,
            decision=PlannerDecision(
                task_id="x",
                config_name="BASELINE",
                config=BASELINE,
                rationale="r",
                alternatives_considered=(),
                expected_information_value="",
            ),
        )


def test_stage_d_stubbed_with_contract() -> None:
    with pytest.raises(NotImplementedError, match="Contract"):
        stage_d_author_hints(opts=HINTER_ONLY, per_task_findings={}, agent_config=None)


def test_stage_e_stubbed_with_contract() -> None:
    with pytest.raises(NotImplementedError, match="Contract"):
        stage_e_phase2_promotion(
            opts=COMBINED,
            cube="x",
            session_id="s",
            hints_before={},
            hints_after={},
            per_task_rewards={},
            benchmark_config=None,
        )


# ---------------------------------------------------------------------------
# Stage F — ledger writes
# ---------------------------------------------------------------------------


def test_stage_f_writes_per_task_entries(isolated_ledger) -> None:
    stage_f_write_ledger(
        opts=WEAK_NOOP,
        cube="terminalbench2",
        session_id="sid-A",
        iter_idx=1,
        per_task_rewards={"task_a": 1.0, "task_b": 0.0},
        per_task_decisions={
            "task_a": PlannerDecision(
                task_id="task_a",
                config_name="BASELINE",
                config=BASELINE,
                rationale="r",
                alternatives_considered=(),
                expected_information_value="",
            ),
        },
    )
    from cube_harness.meta_exploration.ledger import load_ledger

    ledger = load_ledger()
    assert "terminalbench2|task_a" in ledger
    assert "terminalbench2|task_b" in ledger
    assert ledger["terminalbench2|task_a"].disposition == "open"
    assert ledger["terminalbench2|task_a"].notes["reward"] == 1.0
    assert ledger["terminalbench2|task_a"].notes["config_name"] == "BASELINE"


def test_stage_f_skipped_when_ledger_writes_disabled(isolated_ledger) -> None:
    opts = AutoCubeOptions(enable_ledger_writes=False)
    stage_f_write_ledger(
        opts=opts,
        cube="cube",
        session_id="sid",
        iter_idx=1,
        per_task_rewards={"a": 1.0},
        per_task_decisions={},
    )
    from cube_harness.meta_exploration.ledger import load_ledger

    assert load_ledger() == {}


def test_stage_f_records_inference_model_per_entry(isolated_ledger) -> None:
    """The ledger must carry the inference_model so reviewers can audit
    'was this entry generated under the right model?'"""
    stage_f_write_ledger(
        opts=AutoCubeOptions(inference_model="azure/gpt-5"),
        cube="cube",
        session_id="sid",
        iter_idx=1,
        per_task_rewards={"task_a": 0.5},
        per_task_decisions={},
    )
    from cube_harness.meta_exploration.ledger import load_ledger

    ledger = load_ledger()
    assert ledger["cube|task_a"].notes["inference_model"] == "azure/gpt-5"


# ---------------------------------------------------------------------------
# run_outer_loop — end-to-end skeleton with stubbed runner
# ---------------------------------------------------------------------------


def test_run_outer_loop_completes_with_stub_runner(isolated_ledger, tmp_path) -> None:
    """v1 end-to-end smoke: Stage A + B (with injected runner) + F all
    fire across multiple iters; Stage C/D/E are no-ops with warnings."""
    result = run_outer_loop(
        opts=META_EXPLORATION_ONLY,
        cube="terminalbench2",
        task_ids=["t1", "t2"],
        iterations=2,
        benchmark_config=None,
        output_dir=tmp_path / "results",
        runner=lambda tid, cfg: 0.5 if tid == "t1" else 0.0,
    )
    assert result.iterations_run == 2
    assert result.cube == "terminalbench2"
    assert result.inference_model == "azure/gpt-5-mini"
    assert all(r.iter_idx in {1, 2} for r in result.iters)
    # Per-task reward path: t1 always 0.5, t2 always 0.0
    for ir in result.iters:
        assert ir.per_task_rewards["t1"] == 0.5
        assert ir.per_task_rewards["t2"] == 0.0


def test_run_outer_loop_with_unwired_runner_still_completes(isolated_ledger, tmp_path) -> None:
    """When no runner is passed, Stage B raises NotImplementedError —
    the driver should catch it and continue with reward=0, so the iter
    structure still exercises Stage A + F."""
    result = run_outer_loop(
        opts=WEAK_NOOP,
        cube="cube",
        task_ids=["x"],
        iterations=1,
        benchmark_config=None,
        output_dir=tmp_path / "results",
        # no runner — Stage B will skip with reward=0
    )
    assert result.iterations_run == 1
    assert result.iters[0].per_task_rewards["x"] == 0.0


def test_run_outer_loop_validates_options(isolated_ledger, tmp_path) -> None:
    """Incoherent options must raise BEFORE any iter starts."""
    bad = AutoCubeOptions(enable_config_promotion=True)  # missing config_policy
    with pytest.raises(IncoherentOptionsError):
        run_outer_loop(
            opts=bad,
            cube="cube",
            task_ids=["x"],
            iterations=1,
            benchmark_config=None,
            output_dir=tmp_path / "results",
        )


def test_run_outer_loop_records_session_in_ledger(isolated_ledger, tmp_path) -> None:
    """The session_id should appear in every ledger entry written
    during the run — lets a Mode C session correlate which sweep
    touched which task."""
    result = run_outer_loop(
        opts=WEAK_NOOP,
        cube="cube",
        task_ids=["x"],
        iterations=1,
        benchmark_config=None,
        output_dir=tmp_path / "r",
        runner=lambda tid, cfg: 1.0,
    )
    from cube_harness.meta_exploration.ledger import load_ledger

    ledger = load_ledger()
    assert ledger["cube|x"].last_session == result.session_id
