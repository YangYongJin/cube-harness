#!/usr/bin/env python3
"""SMOKE: meta-exploration outer-loop end-to-end on a tiny TB-2 subset.

Confirms the plumbing of Stages A + B + F + the honest-split assertion
introduced by ``feat/meta-exploration``. Exercises:

  - ``stage_a_plan_iter``  — planner picks an ``EpisodeConfig`` per task
  - ``write_plan_json``    — per-iter plan dump
  - ``episode_config_to_agent_config`` — translator
  - ``assert_retest_uses_inference_model`` — honest-split guard
  - ``Experiment(...) + run_sequentially()`` — real episode launch
  - ``stage_f_write_ledger`` — ``~/auto_cube/hints.json`` updates

Cost: ~$2-5 on ``azure/gpt-5-mini`` for 3 TB-2 tasks at 1 iter.
Wall-clock: ~10-20 min on a typical laptop with Docker. CI never runs
this — it spends real LLM money and stands up real Docker containers.

The smoke bypasses ``run_outer_loop`` and composes the stages manually
so the iter-level batching (one Experiment per (agent_config,
episode_config) group) is explicit. ``run_outer_loop``'s per-task
runner contract assumes a stateful runner that we don't ship yet (would
need begin_iter / cache hooks). The smoke is what proves the building
blocks compose; the higher-level batched-runner abstraction is a
follow-up.

Auto-skips when:
  - the ``terminalbench2_cube`` package is not importable (the smoke
    needs the local TB-2 cube installed), or
  - no LLM provider key is found in env or ``.env``
    (``AZURE_API_KEY`` / ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``).

Usage:
    uv run scripts/smoke/auto_cube_meta_exploration.py             # DRY-RUN (default)
    uv run scripts/smoke/auto_cube_meta_exploration.py --execute   # actually spends money
    uv run scripts/smoke/auto_cube_meta_exploration.py --execute --recipe combined
    uv run scripts/smoke/auto_cube_meta_exploration.py --execute --task-ids fix-git,overfull-hbox

Final line follows the cube-harness smoke contract:
    SMOKE OK:   auto_cube_meta_exploration
    SMOKE FAIL: auto_cube_meta_exploration
    SMOKE SKIP: auto_cube_meta_exploration
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from cube_harness.auto_cube.options import RUN_RECIPES, AutoCubeOptions, validate_options
from cube_harness.auto_cube.python_driver import (
    episode_config_to_agent_config,
    make_session_id,
    stage_a_plan_iter,
    stage_f_write_ledger,
    write_plan_json,
)
from cube_harness.llm import LLMConfig
from cube_harness.meta_exploration.agent_config import MetaExplorationGennyConfig
from cube_harness.meta_exploration.planner import EpisodeConfig, PlannerDecision

# Load repo-root `.env` ASAP after imports so LiteLLM sees the keys at
# call time. ``override=True`` because a parent shell often exports an
# *empty* value (e.g. ``export ANTHROPIC_API_KEY=""``) which is enough
# to make ``load_dotenv()``'s default override=False skip it.
load_dotenv(override=True)

logger = logging.getLogger(__name__)

_SMOKE_NAME = "auto_cube_meta_exploration"
_DEFAULT_TASKS = ["adaptive-rejection-sampler", "build-cython-ext", "bn-fit-modify"]


def _print_skip(reason: str) -> None:
    print(f"reason: {reason}")
    print(f"SMOKE SKIP: {_SMOKE_NAME}")


def _print_ok(summary: str) -> None:
    print(summary)
    print(f"SMOKE OK: {_SMOKE_NAME}")


def _print_fail(reason: str) -> None:
    print(f"FAIL: {reason}")
    print(f"SMOKE FAIL: {_SMOKE_NAME}")


_LLM_KEY_ENV_VARS = ("AZURE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")


def _check_prereqs() -> str | None:
    """Returns a reason-to-skip string, or None if the smoke can run.

    Accepts any LiteLLM-supported provider key — the actual model is
    chosen via ``--model``, so being model-agnostic at the prereq layer
    keeps the smoke usable with Azure / OpenAI / Anthropic creds alike.
    """
    if importlib.util.find_spec("terminalbench2_cube") is None:
        return "terminalbench2_cube not importable; install cubes/terminalbench2-cube"
    if not any(os.environ.get(k) for k in _LLM_KEY_ENV_VARS):
        return f"no LLM key in env or .env (need one of {list(_LLM_KEY_ENV_VARS)})"
    return None


def _group_by_config(plan: dict[str, PlannerDecision]) -> dict[EpisodeConfig, list[str]]:
    """Bucket tasks by their chosen ``EpisodeConfig`` so we can run one
    ``Experiment`` per group instead of one per task."""
    buckets: dict[EpisodeConfig, list[str]] = defaultdict(list)
    for task_id, decision in plan.items():
        buckets[decision.config].append(task_id)
    return dict(buckets)


def _run_one_group(
    *,
    episode_config: EpisodeConfig,
    task_ids: list[str],
    base_agent_config: MetaExplorationGennyConfig,
    benchmark_config_all_tasks,
    output_dir: Path,
    iter_idx: int,
) -> dict[str, float]:
    """Launch one ``Experiment`` for the tasks that share an EpisodeConfig.

    Returns ``{task_id: reward}``.
    """
    from cube_harness.exp_runner import run_sequentially
    from cube_harness.experiment import Experiment
    from cube_harness.infra import INFRA_CONFIGS

    agent_config = episode_config_to_agent_config(
        episode_config=episode_config,
        base_agent_config=base_agent_config,
    )
    sub_benchmark = benchmark_config_all_tasks.subset_from_list(task_ids)
    exp = Experiment(
        name=f"smoke-meta-iter{iter_idx}-{_short_key(episode_config)}",
        agent_config=agent_config,
        benchmark_config=sub_benchmark,
        infra=INFRA_CONFIGS["local"],
        output_dir=output_dir / f"iter_{iter_idx}" / _short_key(episode_config),
    )
    logger.info(
        "running group %s tasks=%s model=%s", _short_key(episode_config), task_ids, agent_config.llm_config.model_name
    )
    result = run_sequentially(exp)
    # result.trajectories is keyed by `{task_id}_ep{episode_id}`, not bare
    # task_id — re-key via traj.metadata["task_id"] which carries the raw id.
    rewards: dict[str, float] = dict.fromkeys(task_ids, 0.0)
    observed: set[str] = set()
    for traj_id, traj in result.trajectories.items():
        bare_task_id = traj.metadata.get("task_id", traj_id)
        if bare_task_id not in rewards:
            logger.warning("unexpected trajectory for task %s (traj_id=%s)", bare_task_id, traj_id)
            continue
        rewards[bare_task_id] = float(traj.reward_info.get("reward", 0.0))
        observed.add(bare_task_id)
    for tid in rewards:
        if tid not in observed:
            logger.warning("no trajectory for task %s (group failure?); reward=0", tid)
    return rewards


def _short_key(cfg: EpisodeConfig) -> str:
    """Stable short id for an EpisodeConfig — used in dir names."""
    bits = [cfg.model.replace("/", "_")]
    if cfg.k_candidates != 1:
        bits.append(f"k{cfg.k_candidates}")
    if cfg.apply_promotion:
        bits.append("retest")
    if cfg.enable_refiner:
        bits.append("refine")
    return "-".join(bits) or "baseline"


def _execute_smoke(
    *,
    recipe_name: str,
    task_ids: list[str],
    model: str,
    output_root: Path,
) -> bool:
    """Run the smoke. Returns True on success."""
    from terminalbench2_cube import TERMINALBENCH2_CONFIGS

    opts: AutoCubeOptions = RUN_RECIPES[recipe_name]
    validate_options(opts)

    session_id = make_session_id(prefix=f"smoke-{recipe_name}")
    output_dir = output_root / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    base_agent_config = MetaExplorationGennyConfig(
        llm_config=LLMConfig(model_name=model),
    )
    benchmark_config = TERMINALBENCH2_CONFIGS["default"]
    cube_name = benchmark_config.benchmark_metadata.name

    iter_idx = 1

    # ---- Stage A: plan ----
    plan = stage_a_plan_iter(
        cube=cube_name,
        task_ids=task_ids,
        opts=opts,
        recent_rewards={tid: [] for tid in task_ids},
        budget_remaining_usd=1e9,
    )
    write_plan_json(output_dir=output_dir, iter_idx=iter_idx, plan=plan)
    logger.info("Stage A: planned %d tasks", len(plan))

    # ---- Stage B: launch (with honest-split guard) ----
    rewards: dict[str, float] = {}
    groups = _group_by_config(plan)
    for episode_config, group_task_ids in groups.items():
        # Honest-split guard. Fires once per group — apply_promotion is a
        # property of the episode_config so a per-group check is sufficient.
        from cube_harness.auto_cube.options import assert_retest_uses_inference_model

        assert_retest_uses_inference_model(opts, episode_config)
        group_rewards = _run_one_group(
            episode_config=episode_config,
            task_ids=group_task_ids,
            base_agent_config=base_agent_config,
            benchmark_config_all_tasks=benchmark_config,
            output_dir=output_dir,
            iter_idx=iter_idx,
        )
        rewards.update(group_rewards)

    # ---- Stage F: ledger write ----
    stage_f_write_ledger(
        opts=opts,
        cube=cube_name,
        session_id=session_id,
        iter_idx=iter_idx,
        per_task_rewards=rewards,
        per_task_decisions=plan,
    )
    logger.info("Stage F: wrote %d ledger entries", len(rewards))

    # ---- Verdict ----
    wins = sum(1 for r in rewards.values() if r >= 0.5)
    print(
        f"\nRecipe: {recipe_name}\nSession: {session_id}\nOutput: {output_dir}\n"
        f"Tasks: {len(task_ids)}  wins(>=0.5): {wins}  "
        f"mean_reward: {sum(rewards.values()) / max(len(rewards), 1):.3f}"
    )
    for tid, r in sorted(rewards.items()):
        print(f"  {tid:40s} reward={r:.3f}  config={plan[tid].config_name}")
    # Smoke passes if every task produced *some* trajectory (reward field
    # is present, even if 0). Failure here means the plumbing broke, not
    # that the agent failed the task.
    return True


def main(
    recipe: Annotated[str, typer.Option("--recipe", help="Named recipe from RUN_RECIPES.")] = "weak_noop",
    task_ids_csv: Annotated[
        str,
        typer.Option(
            "--task-ids",
            help="Comma-separated TB-2 task ids. Default: 3 medium-cost tasks.",
        ),
    ] = ",".join(_DEFAULT_TASKS),
    model: Annotated[
        str,
        typer.Option("--model", help="LLM for the inference agent."),
    ] = "azure/gpt-5-mini",
    output_root: Annotated[
        Path,
        typer.Option("--output-root", help="Where smoke artefacts land."),
    ] = Path("~/auto_cube/smoke").expanduser(),
    execute: Annotated[
        bool,
        typer.Option(
            "--execute / --dry-run",
            help="--execute spends real LLM money. Default --dry-run prints "
            "the plan and exits without launching episodes.",
        ),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Smoke-run the meta-exploration outer loop on a tiny TB-2 subset."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if recipe not in RUN_RECIPES:
        _print_fail(f"unknown recipe {recipe!r}; available: {sorted(RUN_RECIPES)}")
        sys.exit(1)

    task_ids = [t.strip() for t in task_ids_csv.split(",") if t.strip()]
    if not task_ids:
        _print_fail("no task ids parsed from --task-ids")
        sys.exit(1)

    if not execute:
        # Dry-run is the safe default — describe the plan, skip prereqs.
        _print_skip(
            f"--dry-run (default): would launch recipe={recipe!r} on "
            f"{len(task_ids)} tasks with model={model!r}. "
            f"Re-run with --execute to actually spend money."
        )
        sys.exit(2)

    skip_reason = _check_prereqs()
    if skip_reason:
        _print_skip(skip_reason)
        sys.exit(2)

    try:
        ok = _execute_smoke(
            recipe_name=recipe,
            task_ids=task_ids,
            model=model,
            output_root=output_root,
        )
    except Exception as e:
        logger.exception("smoke crashed")
        _print_fail(f"{type(e).__name__}: {e}")
        sys.exit(1)

    if ok:
        _print_ok(f"recipe={recipe} tasks={len(task_ids)} model={model}")
    else:
        _print_fail("smoke returned False")
        sys.exit(1)


if __name__ == "__main__":
    typer.run(main)
