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
    episode_config_to_agent_config,
    make_session_id,
    run_outer_loop,
    stage_a_plan_iter,
    stage_b_launch_episode,
    stage_c_dispatch_investigator,
    stage_d_author_hints,
    stage_e_phase2_promotion,
    stage_f_write_ledger,
)
from cube_harness.llm import LLMConfig
from cube_harness.meta_exploration.agent_config import MetaExplorationGennyConfig
from cube_harness.meta_exploration.planner import (
    BASELINE,
    DIAGNOSE_PROFILING,
    DIAGNOSE_SCAFFOLDING,
    ENABLE_REFINER,
    EPISODE_CONFIG_MENU,
    ESCALATE_MODEL,
    EXTEND_TIMEOUT,
    RETEST_PROMOTION,
    WIDEN_SEARCH,
    EpisodeConfig,
    PlannerDecision,
)


def _base_agent_config(model: str = "azure/gpt-5-mini") -> MetaExplorationGennyConfig:
    """Smallest valid base config for translator tests — no mechanical
    knobs set, so the translator's effect is visible per-field."""
    return MetaExplorationGennyConfig(llm_config=LLMConfig(model_name=model))


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


# ---------------------------------------------------------------------------
# episode_config_to_agent_config — the EpisodeConfig → agent-config translator
# ---------------------------------------------------------------------------


def test_translator_baseline_is_identity_modulo_model_copy() -> None:
    """BASELINE has the EpisodeConfig defaults, so the translation should
    produce a config that agrees with the base on every observable field."""
    base = _base_agent_config(model="azure/gpt-5-mini")
    out = episode_config_to_agent_config(episode_config=BASELINE, base_agent_config=base)
    assert out.llm_config.model_name == "azure/gpt-5-mini"
    assert out.k_candidates == 1
    assert out.k_plans == 1
    assert out.perturbations == []
    assert out.enable_refiner is False


def test_translator_returns_a_new_instance_does_not_mutate_base() -> None:
    """``model_copy`` semantics: the translator must NOT mutate the base
    config — that would silently entangle iters/tasks via shared state."""
    base = _base_agent_config()
    out = episode_config_to_agent_config(episode_config=ESCALATE_MODEL, base_agent_config=base)
    assert out is not base
    assert out.llm_config is not base.llm_config
    # Base still at gpt-5-mini even though out is gpt-5.
    assert base.llm_config.model_name == "azure/gpt-5-mini"
    assert out.llm_config.model_name == "azure/gpt-5"


def test_translator_escalate_model_threads_into_llm_config() -> None:
    base = _base_agent_config(model="azure/gpt-5-mini")
    out = episode_config_to_agent_config(episode_config=ESCALATE_MODEL, base_agent_config=base)
    assert out.llm_config.model_name == "azure/gpt-5"


def test_translator_widen_search_applies_k_candidates_and_perturbations() -> None:
    base = _base_agent_config()
    out = episode_config_to_agent_config(episode_config=WIDEN_SEARCH, base_agent_config=base)
    assert out.k_candidates == 3
    assert out.perturbations == ["topk_branch"]
    # WIDEN_SEARCH doesn't touch the model.
    assert out.llm_config.model_name == "azure/gpt-5-mini"


def test_translator_enable_refiner_flips_the_flag() -> None:
    base = _base_agent_config()
    out = episode_config_to_agent_config(episode_config=ENABLE_REFINER, base_agent_config=base)
    assert out.enable_refiner is True


def test_translator_extend_timeout_does_not_touch_agent_config() -> None:
    """``bash_default_timeout`` belongs on the benchmark side. The
    translator MUST NOT silently swallow it into agent_config (would be
    a wrong/lossy mapping). This test pins the explicit non-translation —
    if/when an upstream RFC adds a benchmark-side hook, the wiring lives
    in a separate function, not this one."""
    base = _base_agent_config()
    out = episode_config_to_agent_config(episode_config=EXTEND_TIMEOUT, base_agent_config=base)
    # Nothing on the agent config differs from BASELINE — bash_default_timeout
    # is benchmark-side only.
    assert out.llm_config.model_name == base.llm_config.model_name
    assert out.k_candidates == 1
    assert out.perturbations == []
    assert out.enable_refiner is False


def test_translator_retest_promotion_passes_through_without_model_change() -> None:
    """``apply_promotion=True`` is consumed by the honest-split assertion
    + Stage E, not by the agent config. The translator should leave the
    model alone (it defaults to gpt-5-mini in RETEST_PROMOTION)."""
    base = _base_agent_config(model="azure/gpt-5-mini")
    out = episode_config_to_agent_config(episode_config=RETEST_PROMOTION, base_agent_config=base)
    assert out.llm_config.model_name == "azure/gpt-5-mini"
    assert out.k_candidates == 1


def test_translator_diagnose_recipes_dont_change_agent_config() -> None:
    """``investigator_recipe`` is a Stage C signal, not an agent_config
    field. Picking DIAGNOSE_* should leave the agent config indistinguishable
    from BASELINE on the translator's outputs."""
    base = _base_agent_config()
    for cfg in (DIAGNOSE_SCAFFOLDING, DIAGNOSE_PROFILING):
        out = episode_config_to_agent_config(episode_config=cfg, base_agent_config=base)
        assert out.llm_config.model_name == base.llm_config.model_name
        assert out.k_candidates == 1
        assert out.k_plans == 1
        assert out.perturbations == []
        assert out.enable_refiner is False


def test_translator_preserves_unrelated_base_fields() -> None:
    """The base config's other fields (task_hints, hint, flat_history,
    etc.) must survive translation unchanged — translation only touches
    the documented EpisodeConfig-related fields."""
    base = MetaExplorationGennyConfig(
        llm_config=LLMConfig(model_name="azure/gpt-5-mini"),
        hint="custom global hint",
        task_hints={"task_a": "task-specific tip"},
        flat_history=True,
    )
    out = episode_config_to_agent_config(episode_config=ESCALATE_MODEL, base_agent_config=base)
    assert out.hint == "custom global hint"
    assert out.task_hints == {"task_a": "task-specific tip"}
    assert out.flat_history is True


def test_translator_covers_every_menu_entry_smoke() -> None:
    """Sweep: every entry in EPISODE_CONFIG_MENU must translate without
    raising. Future menu additions automatically join this smoke test —
    we just want a fast assertion that no menu entry breaks the contract."""
    base = _base_agent_config()
    for name, cfg in EPISODE_CONFIG_MENU.items():
        out = episode_config_to_agent_config(episode_config=cfg, base_agent_config=base)
        assert out is not None, f"translator returned None for menu entry {name!r}"
        assert isinstance(out, MetaExplorationGennyConfig)


# ---------------------------------------------------------------------------
# stage_b_launch_episode with base_agent_config — 3-arg runner contract
# ---------------------------------------------------------------------------


def test_stage_b_3arg_runner_receives_translated_agent_config() -> None:
    """When ``base_agent_config`` is provided, the runner is called with
    (task_id, translated_agent_config, episode_config). The agent_config
    must reflect the EpisodeConfig — verifying via the model_name."""
    opts = AutoCubeOptions(inference_model="azure/gpt-5-mini")
    base = _base_agent_config(model="azure/gpt-5-mini")
    decision = PlannerDecision(
        task_id="t",
        config_name="ESCALATE_MODEL",
        config=ESCALATE_MODEL,
        rationale="r",
        alternatives_considered=(),
        expected_information_value="",
    )

    captured: dict[str, object] = {}

    def runner(task_id: str, agent_cfg: MetaExplorationGennyConfig, ep_cfg: EpisodeConfig) -> float:
        captured["task_id"] = task_id
        captured["agent_cfg"] = agent_cfg
        captured["ep_cfg"] = ep_cfg
        return 0.75

    reward = stage_b_launch_episode(
        opts=opts,
        task_id="t",
        decision=decision,
        benchmark_config=None,
        base_agent_config=base,
        runner=runner,
    )
    assert reward == 0.75
    assert captured["task_id"] == "t"
    assert captured["ep_cfg"] is ESCALATE_MODEL  # raw episode_config passed through
    agent_cfg = captured["agent_cfg"]
    assert isinstance(agent_cfg, MetaExplorationGennyConfig)
    # ESCALATE_MODEL bumps the model name.
    assert agent_cfg.llm_config.model_name == "azure/gpt-5"


def test_stage_b_3arg_runner_still_fires_honest_split_assertion() -> None:
    """Promotion + wrong model must raise BEFORE the runner is called,
    same as the 2-arg path. The 3-arg path doesn't weaken the guard."""
    opts = AutoCubeOptions(inference_model="azure/gpt-5-mini")
    base = _base_agent_config(model="azure/gpt-5-mini")
    bad_cfg = EpisodeConfig(model="azure/gpt-5", apply_promotion=True)
    decision = PlannerDecision(
        task_id="x",
        config_name="BAD",
        config=bad_cfg,
        rationale="r",
        alternatives_considered=(),
        expected_information_value="",
    )

    def runner_should_not_be_called(task_id, agent_cfg, ep_cfg):  # noqa: ARG001
        pytest.fail("runner invoked despite honest-split assertion failure")

    with pytest.raises(IncoherentOptionsError, match="re-test gate violation"):
        stage_b_launch_episode(
            opts=opts,
            task_id="x",
            decision=decision,
            benchmark_config=None,
            base_agent_config=base,
            runner=runner_should_not_be_called,
        )


def test_stage_b_2arg_path_is_unchanged_when_no_base_agent_config() -> None:
    """Backwards-compat: existing callers that don't pass
    ``base_agent_config`` still see the legacy 2-arg runner contract."""
    opts = AutoCubeOptions()
    decision = PlannerDecision(
        task_id="t",
        config_name="BASELINE",
        config=BASELINE,
        rationale="r",
        alternatives_considered=(),
        expected_information_value="",
    )
    captured: dict[str, object] = {}

    def runner_2arg(task_id, ep_cfg):
        captured["task_id"] = task_id
        captured["ep_cfg"] = ep_cfg
        return 0.42

    reward = stage_b_launch_episode(
        opts=opts,
        task_id="t",
        decision=decision,
        benchmark_config=None,
        runner=runner_2arg,
    )
    assert reward == 0.42
    assert captured["ep_cfg"] is BASELINE


def test_run_outer_loop_with_base_agent_config_threads_translation(isolated_ledger, tmp_path) -> None:
    """End-to-end: run_outer_loop with base_agent_config calls the
    3-arg runner per task. Smoke that the parameter plumbs through."""
    base = _base_agent_config(model="azure/gpt-5-mini")
    seen_model_names: list[str] = []

    def runner(task_id, agent_cfg, ep_cfg):  # noqa: ARG001
        seen_model_names.append(agent_cfg.llm_config.model_name)
        return 1.0

    result = run_outer_loop(
        opts=META_EXPLORATION_ONLY,  # planner runs; some near-miss tasks pick ESCALATE_MODEL
        cube="cube",
        task_ids=["fresh"],  # fresh task → BASELINE pick → no model change
        iterations=1,
        benchmark_config=None,
        output_dir=tmp_path / "r",
        base_agent_config=base,
        runner=runner,
    )
    assert result.iterations_run == 1
    assert seen_model_names == ["azure/gpt-5-mini"]  # BASELINE inherits base model
