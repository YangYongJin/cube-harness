#!/usr/bin/env python3
"""Uniform-config scaling sweep — answers Tier 2's load-bearing question.

**Is exploration actually the bottleneck?** If win-rate plateaus quickly
as we throw more compute at a *fixed* config (more replicas per task,
longer rollouts), then meta-exploration's per-episode-config policy is
speculative — there's no headroom for a smarter explorer to claim. If
win-rate keeps climbing with extra compute, exploration IS the bottleneck
and Tier 3 (meta-exploration tests + algorithm ports) is worth the spend.

The sweep holds the agent config fixed (no planner — every episode runs
under ``WEAK_NOOP`` by default) and varies two compute knobs:

  - ``max_steps``         — per-episode rollout budget (100 / 200 / 400)
  - ``replicas``          — independent runs per task (1 / 3 / 5 / 10)

Per cell ``(max_steps, replicas)``, the script:
  1. Launches ``replicas`` independent ``Experiment`` runs over the task
     subset, each with a unique ``output_dir``.
  2. Collects per-task per-replica rewards into a tidy long-format CSV.
  3. Writes one ``cell.json`` summary per (max_steps, replicas) cell.

After all cells finish, the script writes a top-level ``sweep_report.md``
with the per-cell mean win-rate matrix — the table that informs the
go/no-go on Tier 3.

Cost (HANDOFF estimate ~$20-50): 3 max_steps × 4 replicas-counts × 3-5
tasks × ~10 replicas avg ≈ 50-100 episodes on ``azure/gpt-5-mini``.
Wall-clock: hours, not minutes.

Auto-skips when:
  - the ``terminalbench2_cube`` package is not importable, or
  - no LLM provider key is found in env or ``.env``
    (``AZURE_API_KEY`` / ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``).

Usage:
    # Dry-run (default) — print the cells and estimated episode count:
    uv run scripts/auto_cube/scaling_sweep.py

    # Narrow scope for a cheap pilot:
    uv run scripts/auto_cube/scaling_sweep.py --execute \\
        --max-steps-values 100,200 --replicas-values 1,3 \\
        --task-ids fix-git,overfull-hbox

    # Full sweep — pricey:
    uv run scripts/auto_cube/scaling_sweep.py --execute

Final line:
    SWEEP OK:   scaling_sweep
    SWEEP FAIL: scaling_sweep
    SWEEP SKIP: scaling_sweep
"""

from __future__ import annotations

import csv
import importlib.util
import json
import logging
import os
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from cube_harness.auto_cube.options import RUN_RECIPES
from cube_harness.auto_cube.python_driver import make_session_id
from cube_harness.llm import LLMConfig
from cube_harness.meta_exploration.agent_config import MetaExplorationGennyConfig

# Same .env auto-load posture as the smoke — see
# scripts/smoke/auto_cube_meta_exploration.py for the override=True rationale.
load_dotenv(override=True)

logger = logging.getLogger(__name__)

_SWEEP_NAME = "scaling_sweep"
_DEFAULT_TASKS = ["adaptive-rejection-sampler", "build-cython-ext", "bn-fit-modify"]
_DEFAULT_MAX_STEPS_VALUES = "100,200,400"
_DEFAULT_REPLICAS_VALUES = "1,3,5,10"


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _print_skip(reason: str) -> None:
    print(f"reason: {reason}")
    print(f"SWEEP SKIP: {_SWEEP_NAME}")


def _print_ok(summary: str) -> None:
    print(summary)
    print(f"SWEEP OK: {_SWEEP_NAME}")


def _print_fail(reason: str) -> None:
    print(f"FAIL: {reason}")
    print(f"SWEEP FAIL: {_SWEEP_NAME}")


_LLM_KEY_ENV_VARS = ("AZURE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")


def _check_prereqs() -> str | None:
    if importlib.util.find_spec("terminalbench2_cube") is None:
        return "terminalbench2_cube not importable; install cubes/terminalbench2-cube"
    if not any(os.environ.get(k) for k in _LLM_KEY_ENV_VARS):
        return f"no LLM key in env or .env (need one of {list(_LLM_KEY_ENV_VARS)})"
    return None


def _parse_int_csv(s: str, *, what: str) -> list[int]:
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    except ValueError as e:
        raise typer.BadParameter(f"{what} must be comma-separated integers: {e}") from e


# ---------------------------------------------------------------------------
# Per-cell execution
# ---------------------------------------------------------------------------


def _run_one_replica(
    *,
    cell_dir: Path,
    replica_idx: int,
    max_steps: int,
    base_agent_config: MetaExplorationGennyConfig,
    benchmark_config_all_tasks,
    task_ids: list[str],
) -> dict[str, float]:
    """Launch one ``Experiment`` covering all the cell's tasks. Returns
    ``{task_id: reward}``."""
    from cube_harness.exp_runner import run_sequentially
    from cube_harness.experiment import Experiment
    from cube_harness.infra import INFRA_CONFIGS

    sub_benchmark = benchmark_config_all_tasks.subset_from_list(task_ids)
    exp = Experiment(
        name=f"scaling-ms{max_steps}-rep{replica_idx}",
        agent_config=base_agent_config,
        benchmark_config=sub_benchmark,
        infra=INFRA_CONFIGS["local"],
        output_dir=cell_dir / f"replica_{replica_idx}",
        max_steps=max_steps,
    )
    logger.info(
        "running replica %d max_steps=%d tasks=%d",
        replica_idx,
        max_steps,
        len(task_ids),
    )
    result = run_sequentially(exp)
    rewards: dict[str, float] = {}
    for task_id in task_ids:
        traj = result.trajectories.get(task_id)
        if traj is None:
            logger.warning("no trajectory for task %s replica %d; reward=0", task_id, replica_idx)
            rewards[task_id] = 0.0
            continue
        rewards[task_id] = float(traj.reward_info.get("reward", 0.0))
    return rewards


def _execute_one_cell(
    *,
    output_root: Path,
    max_steps: int,
    replicas_count: int,
    base_agent_config: MetaExplorationGennyConfig,
    benchmark_config_all_tasks,
    task_ids: list[str],
) -> dict[str, list[float]]:
    """Run all replicas for one (max_steps, replicas) cell.

    Returns ``{task_id: [reward_replica_0, ..., reward_replica_{N-1}]}``.
    Writes ``cell.json`` summary into the cell dir.
    """
    cell_dir = output_root / f"ms{max_steps}_rep{replicas_count}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    cell_rewards: dict[str, list[float]] = defaultdict(list)
    for replica_idx in range(replicas_count):
        replica_rewards = _run_one_replica(
            cell_dir=cell_dir,
            replica_idx=replica_idx,
            max_steps=max_steps,
            base_agent_config=base_agent_config,
            benchmark_config_all_tasks=benchmark_config_all_tasks,
            task_ids=task_ids,
        )
        for task_id, reward in replica_rewards.items():
            cell_rewards[task_id].append(reward)
    summary = _summarize_cell(max_steps, replicas_count, cell_rewards)
    (cell_dir / "cell.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return dict(cell_rewards)


def _summarize_cell(
    max_steps: int,
    replicas_count: int,
    cell_rewards: dict[str, list[float]],
) -> dict[str, object]:
    """Per-cell stats: mean win-rate (reward >= 0.5), mean reward."""
    per_task: dict[str, dict[str, float]] = {}
    for task_id, rewards in cell_rewards.items():
        per_task[task_id] = {
            "mean_reward": sum(rewards) / max(len(rewards), 1),
            "win_rate": sum(1 for r in rewards if r >= 0.5) / max(len(rewards), 1),
            "n_replicas": len(rewards),
        }
    overall_win_rate = sum(per_task[t]["win_rate"] for t in per_task) / max(len(per_task), 1)
    overall_mean_reward = sum(per_task[t]["mean_reward"] for t in per_task) / max(len(per_task), 1)
    return {
        "max_steps": max_steps,
        "replicas": replicas_count,
        "n_tasks": len(per_task),
        "overall_win_rate": overall_win_rate,
        "overall_mean_reward": overall_mean_reward,
        "per_task": per_task,
    }


# ---------------------------------------------------------------------------
# Top-level reports
# ---------------------------------------------------------------------------


def _write_long_csv(
    output_root: Path,
    all_cell_rewards: dict[tuple[int, int], dict[str, list[float]]],
) -> Path:
    """Long-format: one row per (max_steps, replicas, task_id, replica_idx, reward)."""
    path = output_root / "rewards_long.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["max_steps", "replicas", "task_id", "replica_idx", "reward"])
        for (max_steps, replicas_count), cell in sorted(all_cell_rewards.items()):
            for task_id, rewards in sorted(cell.items()):
                for replica_idx, reward in enumerate(rewards):
                    writer.writerow([max_steps, replicas_count, task_id, replica_idx, reward])
    return path


def _write_sweep_report(
    output_root: Path,
    max_steps_values: list[int],
    replicas_values: list[int],
    all_cell_rewards: dict[tuple[int, int], dict[str, list[float]]],
    task_ids: list[str],
    recipe_name: str,
    model: str,
) -> Path:
    """Markdown report: per-cell mean win-rate matrix + the interpretive
    'is exploration the bottleneck?' table."""
    path = output_root / "sweep_report.md"
    lines: list[str] = []
    lines.append("# Scaling sweep report\n")
    lines.append(f"- **Recipe (held fixed):** `{recipe_name}`")
    lines.append(f"- **Model:** `{model}`")
    lines.append(f"- **Tasks:** {len(task_ids)} — `{', '.join(task_ids)}`")
    lines.append(f"- **max_steps swept:** {max_steps_values}")
    lines.append(f"- **replicas swept:** {replicas_values}")
    lines.append("\n## Win-rate matrix (rows: max_steps, cols: replicas)\n")
    header = "| max_steps \\ replicas | " + " | ".join(str(r) for r in replicas_values) + " |"
    sep = "|" + "---|" * (len(replicas_values) + 1)
    lines.append(header)
    lines.append(sep)
    for ms in max_steps_values:
        row_cells: list[str] = [str(ms)]
        for r in replicas_values:
            cell = all_cell_rewards.get((ms, r))
            if cell is None:
                row_cells.append("—")
                continue
            summary = _summarize_cell(ms, r, cell)
            row_cells.append(f"{summary['overall_win_rate']:.2f}")
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append(
        "\n## Interpretation\n\n"
        "If the win-rate plateaus across the row (more replicas don't help) "
        "AND across the column (more steps don't help), exploration is **NOT** "
        "the bottleneck — Tier 3 meta-exploration tests are speculative and "
        "priorities should shift toward hint quality (Tier 4).\n\n"
        "If win-rate keeps climbing with either knob, exploration IS the "
        "bottleneck — Tier 3 is worth the spend."
    )
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _execute_sweep(
    *,
    recipe_name: str,
    task_ids: list[str],
    model: str,
    output_root: Path,
    max_steps_values: list[int],
    replicas_values: list[int],
) -> bool:
    from terminalbench2_cube import TERMINALBENCH2_CONFIGS

    session_id = make_session_id(prefix=f"sweep-{recipe_name}")
    sweep_dir = output_root / session_id
    sweep_dir.mkdir(parents=True, exist_ok=True)
    logger.info("sweep session=%s out=%s", session_id, sweep_dir)

    base_agent_config = MetaExplorationGennyConfig(
        llm_config=LLMConfig(model_name=model),
    )
    benchmark_config = TERMINALBENCH2_CONFIGS["default"]

    all_cell_rewards: dict[tuple[int, int], dict[str, list[float]]] = {}
    for max_steps in max_steps_values:
        for replicas_count in replicas_values:
            logger.info("=== cell max_steps=%d replicas=%d ===", max_steps, replicas_count)
            cell_rewards = _execute_one_cell(
                output_root=sweep_dir,
                max_steps=max_steps,
                replicas_count=replicas_count,
                base_agent_config=base_agent_config,
                benchmark_config_all_tasks=benchmark_config,
                task_ids=task_ids,
            )
            all_cell_rewards[(max_steps, replicas_count)] = cell_rewards

    csv_path = _write_long_csv(sweep_dir, all_cell_rewards)
    report_path = _write_sweep_report(
        sweep_dir,
        max_steps_values,
        replicas_values,
        all_cell_rewards,
        task_ids,
        recipe_name,
        model,
    )
    print(f"\nWrote long CSV: {csv_path}")
    print(f"Wrote report:   {report_path}\n")
    return True


def _estimate_episodes(
    max_steps_values: Sequence[int],
    replicas_values: Sequence[int],
    n_tasks: int,
) -> int:
    return len(max_steps_values) * sum(replicas_values) * n_tasks


def main(
    recipe: Annotated[
        str,
        typer.Option("--recipe", help="Named recipe from RUN_RECIPES (held fixed across the sweep)."),
    ] = "weak_noop",
    task_ids_csv: Annotated[
        str,
        typer.Option("--task-ids", help="Comma-separated TB-2 task ids."),
    ] = ",".join(_DEFAULT_TASKS),
    model: Annotated[
        str,
        typer.Option("--model", help="LLM model name."),
    ] = "azure/gpt-5-mini",
    max_steps_values: Annotated[
        str,
        typer.Option("--max-steps-values", help="Comma-separated rollout step caps."),
    ] = _DEFAULT_MAX_STEPS_VALUES,
    replicas_values: Annotated[
        str,
        typer.Option("--replicas-values", help="Comma-separated replicas-per-cell counts."),
    ] = _DEFAULT_REPLICAS_VALUES,
    output_root: Annotated[
        Path,
        typer.Option("--output-root", help="Where the sweep dir lands."),
    ] = Path("~/auto_cube/scaling_sweeps").expanduser(),
    execute: Annotated[
        bool,
        typer.Option(
            "--execute / --dry-run",
            help="--execute actually launches Experiments (HANDOFF estimate ~$20-50). "
            "Default --dry-run prints the cells and episode-count estimate.",
        ),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Scaling sweep — does more compute at fixed config saturate win-rate?"""
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

    ms_values = _parse_int_csv(max_steps_values, what="--max-steps-values")
    rep_values = _parse_int_csv(replicas_values, what="--replicas-values")
    n_episodes = _estimate_episodes(ms_values, rep_values, len(task_ids))

    if not execute:
        _print_skip(
            f"--dry-run (default): would launch {n_episodes} episodes across "
            f"{len(ms_values)} max_steps × {len(rep_values)} replica-counts × "
            f"{len(task_ids)} tasks. recipe={recipe!r} model={model!r}. "
            f"Re-run with --execute to actually spend money."
        )
        sys.exit(2)

    skip_reason = _check_prereqs()
    if skip_reason:
        _print_skip(skip_reason)
        sys.exit(2)

    try:
        ok = _execute_sweep(
            recipe_name=recipe,
            task_ids=task_ids,
            model=model,
            output_root=output_root,
            max_steps_values=ms_values,
            replicas_values=rep_values,
        )
    except Exception as e:
        logger.exception("sweep crashed")
        _print_fail(f"{type(e).__name__}: {e}")
        sys.exit(1)

    if ok:
        _print_ok(f"recipe={recipe} episodes={n_episodes} model={model}")
    else:
        _print_fail("sweep returned False")
        sys.exit(1)


if __name__ == "__main__":
    typer.run(main)
