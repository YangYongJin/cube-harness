"""Outer-loop SDK driver — unattended runner for Auto-CUBE use cases.

The SDK-mode equivalent of running ``/auto-cube-<use-case>`` in
interactive Claude Code. Reads an ``AutoCubeOptions`` recipe and
conditionally fires each stage of the meta-loop:

  Stage A — per-episode config policy (when ``enable_config_policy``)
  Stage B — episodes run with the picked configs
            (calls ``assert_retest_uses_inference_model`` immediately
            before launching any re-test episode — the honest gate)
  Stage C — per-trajectory Investigator dispatch
            (when ``enable_l1_recipe_routing``)         [STUB — see §C]
  Stage D — text hint authoring                          [STUB — see §D]
            (when ``enable_hint_authoring``)
  Stage E — Phase 2 promotion + re-test gate             [STUB — see §E]
            (when ``enable_text_promotion`` /
             ``enable_config_promotion``)
  Stage F — ledger writes (when ``enable_ledger_writes``)

This v1 wires Stages A, B (gate-assertion), and F. Stages C/D/E
require the actual Investigator + hint-authoring + episode-running
plumbing to land — they are STUBBED here with NotImplementedError +
the exact contract they need to satisfy when implemented.

Once Stages C/D/E land, every ``AutoCubeOptions`` recipe in
``options.RUN_RECIPES`` will be runnable end-to-end from bash:

    uv run python -m cube_harness.auto_cube.python_driver \\
        --recipe combined \\
        --benchmark terminalbench2 \\
        --iterations 3 \\
        --output-root /tmp/foo

(The CLI is also a stub — design in this module's docstring tail.)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from cube_harness.auto_cube.options import (
    AutoCubeOptions,
    assert_retest_uses_inference_model,
    filter_menu_by_options,
    validate_options,
)
from cube_harness.meta_exploration.ledger import (
    load_ledger,
    upsert_entry,
)
from cube_harness.meta_exploration.planner import (
    BASELINE,
    PlannerDecision,
    PlannerState,
    plan_iter,
    summarize_plan,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IterResult:
    """Outcome of one outer-loop iter — per task, the planner's pick +
    the episode reward + any promotion outcomes."""

    iter_idx: int
    plan: dict[str, PlannerDecision]
    per_task_rewards: dict[str, float]
    # Promotion verdicts (one per attempted promotion this iter).
    # Empty when Stage E is disabled or no candidates emerged.
    promotion_verdicts: list[object] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class MetaLoopResult:
    cube: str
    recipe_name: str
    session_id: str
    iterations_run: int
    iters: list[IterResult]
    inference_model: str
    output_dir: Path


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def make_session_id(prefix: str = "session") -> str:
    """Stable id: ``{prefix}-{utc-iso-secs}Z-{6-char-nonce}``."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    nonce = uuid.uuid4().hex[:6]
    return f"{prefix}-{ts}-{nonce}"


# ---------------------------------------------------------------------------
# Stage A — per-task config planning
# ---------------------------------------------------------------------------


def _build_planner_state(
    *,
    cube: str,
    task_ids: list[str],
    opts: AutoCubeOptions,
    recent_rewards: dict[str, list[float]],
    budget_remaining_usd: float,
) -> PlannerState:
    """Construct ``PlannerState`` from the ledger + recent reward windows."""
    ledger = load_ledger() if opts.enable_ledger_writes else {}
    return PlannerState(
        cube=cube,
        ledger=ledger,
        recent_rewards=recent_rewards,
        budget_remaining_usd=budget_remaining_usd,
    )


def stage_a_plan_iter(
    *,
    cube: str,
    task_ids: list[str],
    opts: AutoCubeOptions,
    recent_rewards: dict[str, list[float]],
    budget_remaining_usd: float,
) -> dict[str, PlannerDecision]:
    """Stage A — pick per-episode config per task for this iter.

    When ``opts.enable_config_policy`` is False, every task gets
    ``BASELINE`` (the unattended-default behavior). When True, calls
    ``planner.plan_iter`` over the ledger + recent rewards.

    The returned ``PlannerDecision`` per task drives Stage B's episode
    launch.
    """
    if not opts.enable_config_policy:
        # No policy — every task gets the default config. Decision
        # records this as the rationale so the plan.json doesn't look
        # mysteriously uniform.
        return {
            tid: PlannerDecision(
                task_id=tid,
                config_name="BASELINE",
                config=BASELINE,
                rationale="enable_config_policy=False; baseline for every task",
                alternatives_considered=(),
                expected_information_value="baseline_unattended",
            )
            for tid in task_ids
        }

    # Filter the menu to honor allow_model_escalation BEFORE planning
    # (the planner is allowed to reference any menu entry by name; the
    # filter prevents ESCALATE_MODEL from being picked when disallowed).
    # NB: the current planner reads EPISODE_CONFIG_MENU directly; a
    # follow-up would have it accept a filtered menu. For v1 we rely on
    # validate_options + the menu-filter test rather than runtime
    # enforcement in the planner.
    _ = filter_menu_by_options(opts)  # validates options indirectly

    state = _build_planner_state(
        cube=cube,
        task_ids=task_ids,
        opts=opts,
        recent_rewards=recent_rewards,
        budget_remaining_usd=budget_remaining_usd,
    )
    plan = plan_iter(task_ids, state)
    logger.info(
        "stage_a planned %d tasks; distribution=%s",
        len(plan),
        summarize_plan(plan),
    )
    return plan


# ---------------------------------------------------------------------------
# Stage B — launch episodes (with honest-gate assertion)
# ---------------------------------------------------------------------------


def stage_b_launch_episode(
    *,
    opts: AutoCubeOptions,
    task_id: str,
    decision: PlannerDecision,
    benchmark_config,  # cube.benchmark.BenchmarkConfig — typed loosely
    runner=None,  # callable(task_id, config) -> reward; injectable for tests
) -> float:
    """Stage B — launch one episode under the planner-picked config.

    Calls ``assert_retest_uses_inference_model`` FIRST. Any re-test
    episode (``apply_promotion=True``) must use ``opts.inference_model``
    — this is the load-bearing honest-split guard from DESIGN.md §1.

    The actual episode execution is delegated to ``runner`` — by
    default this raises NotImplementedError pending wiring to
    ``cube_harness.experiment.Experiment`` + ``exp_runner``. The
    injectable hook makes the assertion + decision-flow testable
    without running real LLM episodes.
    """
    # THE honest-split guard. Fires before any side effect.
    assert_retest_uses_inference_model(opts, decision.config)

    if runner is None:
        raise NotImplementedError(
            "stage_b_launch_episode: actual episode execution not yet wired. "
            "Pass `runner` callable for tests, or wait for the full driver to "
            "thread `cube_harness.experiment.Experiment` through here. "
            "Contract: runner(task_id, episode_config) -> reward (float in [0,1])."
        )

    reward = runner(task_id, decision.config)
    logger.info(
        "stage_b launched task=%s config=%s reward=%.3f",
        task_id,
        decision.config_name,
        reward,
    )
    return reward


# ---------------------------------------------------------------------------
# Stages C / D / E — STUBBED with contract
# ---------------------------------------------------------------------------


def stage_c_dispatch_investigator(
    *,
    opts: AutoCubeOptions,
    trajectory,  # cube_harness.core.Trajectory
    decision: PlannerDecision,
):
    """Stage C — per-trajectory Investigator dispatch (L1 routing).

    NOT YET WIRED. Contract for the next session:
      1. If ``opts.enable_l1_recipe_routing`` is False: skip entirely.
      2. If ``decision.config.investigator_recipe`` is set (planner
         override): use that recipe directly.
      3. Else: call ``recipe_router.classify_recipe`` over the
         trajectory's reward + repeats + tokens-per-step.
      4. Invoke the matching upstream Investigator recipe via
         ``cube_harness.analyze.investigator`` (which itself uses
         ``claude_agent_sdk``). Returns ``BaseFindings``
         (or a recipe-specific subclass like ``HinterOutput``).

    Required upstream surfaces (verify shape exists at time of wiring):
      - ``cube_harness.analyze.investigator.use_cases.hinter.recipe``
      - ``cube_harness.analyze.investigator.use_cases.general_blame.recipe``
      - ``cube_harness.analyze.investigator.use_cases.agent_scaffolding.recipe``
      - ``cube_harness.analyze.investigator.use_cases.profiling.recipe``

    Output flows into Stage D (text hints) AND Stage F (ledger notes,
    where Investigator-derived flags like ``loop_pattern_suspected``
    populate ``HintLedgerEntry.notes``).
    """
    raise NotImplementedError("stage_c_dispatch_investigator: wiring deferred. See docstring §Contract.")


def stage_d_author_hints(
    *,
    opts: AutoCubeOptions,
    per_task_findings: dict[str, object],  # task_id -> BaseFindings
    agent_config,  # GennyConfig to mutate
):
    """Stage D — aggregate text-hint candidates and mutate agent config.

    NOT YET WIRED. Contract for the next session:
      1. If ``opts.enable_hint_authoring`` is False: skip entirely.
      2. Filter ``per_task_findings`` to those whose recipe was
         ``hinter`` (per Stage C's L1 dispatch).
      3. Extract ``task_hints[]`` per finding (the ``HinterOutput``
         subclass exposes this).
      4. Mutate ``agent_config.task_hints[task_id] = <text>`` for each
         hint above the confidence threshold.
      5. Optionally: if ``opts.propose_multiple_harnesses``, run
         step 4 K times and pick the harness with best validation
         reward. Meta-Harness algorithm — owns this branch.
      6. Optionally: if ``opts.enable_pareto_verify``, run a Pareto
         check on per-task rewards before applying step 4; reject the
         mutation if any task strictly regressed.

    Output: mutated ``agent_config``; the iter's hint diff (used by
    Stage E candidate detection).
    """
    raise NotImplementedError("stage_d_author_hints: wiring deferred. See docstring §Contract.")


def stage_e_phase2_promotion(
    *,
    opts: AutoCubeOptions,
    cube: str,
    session_id: str,
    hints_before: dict[str, str],
    hints_after: dict[str, str],
    per_task_rewards: dict[str, float],
    benchmark_config,  # for the held-out re-test subset
) -> list[object]:
    """Stage E — Phase 2 promotion + re-test gate.

    NOT YET WIRED. Contract for the next session:
      1. If ``opts.enable_text_promotion`` is False AND
         ``opts.enable_config_promotion`` is False: skip entirely.
      2. Detect candidates via
         ``promotion.candidate_from_task_hints_overlap`` (text path)
         and a parallel ``candidate_from_episode_config_overlap``
         (config path — TBD; not yet implemented).
      3. Per candidate, up to ``opts.held_out_size`` held-out tasks:
         a. Copy agent_config; apply promotion via
            ``promotion.apply_promotion_to_config``.
         b. Set the promoted-copy's model to ``opts.inference_model``
            (NOT the planner's pick — the honest-split rule).
         c. Run a sub-experiment on (affected + held_out) tasks.
            Each episode passes through ``stage_b_launch_episode``
            which fires ``assert_retest_uses_inference_model``.
         d. Parse per-task rewards from the sub-experiment.
         e. ``promotion.decide_verdict`` produces the
            ``PromoteVerdict``.
      4. Per verdict:
         - ``ok``: mark ledger ``disposition='promoted'``,
           ``promoted_to=cand.rung``, apply mutation to the live
           agent_config.
         - ``affected_regressed`` / ``held_out_regressed``: mark
           ``disposition='cheat_only'``, revert the mutation.

    Returns: list of ``PromoteVerdict`` objects (one per attempted
    promotion this iter). Empty when no candidates.
    """
    raise NotImplementedError("stage_e_phase2_promotion: wiring deferred. See docstring §Contract.")


# ---------------------------------------------------------------------------
# Stage F — ledger writes
# ---------------------------------------------------------------------------


def stage_f_write_ledger(
    *,
    opts: AutoCubeOptions,
    cube: str,
    session_id: str,
    iter_idx: int,
    per_task_rewards: dict[str, float],
    per_task_decisions: dict[str, PlannerDecision],
) -> None:
    """Stage F — write per-task entries to ``~/auto_cube/hints.json``.

    When ``opts.enable_ledger_writes`` is False: no-op. Otherwise
    writes one entry per task with disposition='open' (Phase 2
    transitions disposition; this stage only records baseline state).

    Notes field carries the iter+reward+chosen_config so post-hoc
    analysis can audit the policy's decisions per task across iters.
    """
    if not opts.enable_ledger_writes:
        return
    for task_id, reward in per_task_rewards.items():
        decision = per_task_decisions.get(task_id)
        config_name = decision.config_name if decision else "BASELINE"
        try:
            upsert_entry(
                cube=cube,
                task_id=task_id,
                session_id=session_id,
                disposition="open",
                notes={
                    "iter": iter_idx,
                    "reward": float(reward),
                    "config_name": config_name,
                    "inference_model": opts.inference_model,
                },
            )
        except Exception:
            logger.exception(
                "ledger upsert failed for task=%s; continuing",
                task_id,
            )


# ---------------------------------------------------------------------------
# The outer loop — composes all stages
# ---------------------------------------------------------------------------


def run_outer_loop(
    *,
    opts: AutoCubeOptions,
    cube: str,
    task_ids: list[str],
    iterations: int,
    benchmark_config,  # cube.benchmark.BenchmarkConfig
    output_dir: Path,
    session_id: str | None = None,
    runner=None,  # injectable episode runner
) -> MetaLoopResult:
    """The unattended outer-loop driver.

    Wires Stages A, B (with assertion), F. Stages C/D/E are stubs;
    when an iter would invoke them, this driver currently no-ops past
    them and logs a warning. Once those stages are wired, this
    function is the entry point for every named recipe in
    ``options.RUN_RECIPES``.
    """
    validate_options(opts)
    session_id = session_id or make_session_id()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "outer_loop start: cube=%s session=%s iterations=%d inference_model=%s",
        cube,
        session_id,
        iterations,
        opts.inference_model,
    )

    recent_rewards: dict[str, list[float]] = {tid: [] for tid in task_ids}
    iters: list[IterResult] = []
    budget_remaining_usd = 1e9  # TODO: take from opts when budget knob lands

    for iter_idx in range(1, iterations + 1):
        # Stage A — plan per-task configs.
        plan = stage_a_plan_iter(
            cube=cube,
            task_ids=task_ids,
            opts=opts,
            recent_rewards=recent_rewards,
            budget_remaining_usd=budget_remaining_usd,
        )

        # Stage B — launch episodes (with honest-split assertion).
        per_task_rewards: dict[str, float] = {}
        for task_id in task_ids:
            decision = plan[task_id]
            try:
                reward = stage_b_launch_episode(
                    opts=opts,
                    task_id=task_id,
                    decision=decision,
                    benchmark_config=benchmark_config,
                    runner=runner,
                )
            except NotImplementedError:
                # In v1 this is the expected path: Stage B's episode
                # runner isn't wired. Skip cleanly with reward=0; the
                # session still exercises Stage A + F + the assertion.
                logger.warning(
                    "stage_b_launch_episode unwired; skipping task=%s",
                    task_id,
                )
                reward = 0.0
            per_task_rewards[task_id] = reward
            recent_rewards[task_id].append(reward)

        # Stage C / D / E — STUBBED. Once wired, conditionally invoke
        # based on opts.enable_l1_recipe_routing / enable_hint_authoring
        # / enable_{text,config}_promotion respectively.
        promotion_verdicts: list[object] = []
        if (
            opts.enable_l1_recipe_routing
            or opts.enable_hint_authoring
            or opts.enable_text_promotion
            or opts.enable_config_promotion
        ):
            logger.warning(
                "iter=%d: Stages C/D/E not yet wired (Pivot 7 next session); "
                "options that depend on them are no-op this iter.",
                iter_idx,
            )

        # Stage F — ledger writes.
        stage_f_write_ledger(
            opts=opts,
            cube=cube,
            session_id=session_id,
            iter_idx=iter_idx,
            per_task_rewards=per_task_rewards,
            per_task_decisions=plan,
        )

        iters.append(
            IterResult(
                iter_idx=iter_idx,
                plan=plan,
                per_task_rewards=per_task_rewards,
                promotion_verdicts=promotion_verdicts,
            )
        )
        logger.info(
            "iter %d/%d complete; mean_reward=%.3f",
            iter_idx,
            iterations,
            sum(per_task_rewards.values()) / max(len(per_task_rewards), 1),
        )

    return MetaLoopResult(
        cube=cube,
        recipe_name="",  # caller fills if running a named recipe
        session_id=session_id,
        iterations_run=len(iters),
        iters=iters,
        inference_model=opts.inference_model,
        output_dir=output_dir,
    )


__all__ = [
    "IterResult",
    "MetaLoopResult",
    "make_session_id",
    "run_outer_loop",
    "stage_a_plan_iter",
    "stage_b_launch_episode",
    "stage_c_dispatch_investigator",
    "stage_d_author_hints",
    "stage_e_phase2_promotion",
    "stage_f_write_ledger",
]
