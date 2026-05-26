"""L1 — Investigator recipe routing (Phase 4 chunk 3).

Per failed episode, pick which of upstream auto-cube's 4 Investigator
recipes (`hinter` / `general_blame` / `agent_scaffolding` / `profiling`)
best fits the failure shape. The picked recipe shapes the discipline
the SDK applies when authoring hints for that episode.

The 4 recipes (per upstream
`third_party/cube-harness/src/cube_harness/analyze/investigator/use_cases/<name>/SKILL.md`):

  * `hinter` — extract `task_hints[task_id]` candidates; default for
    failures that look like "the agent didn't know what we expected."
  * `general_blame` — single-pass closed-world blame attribution;
    fallback when hinter returns an empty `task_hints[]` (signals real
    bug, not a steerable failure).
  * `agent_scaffolding` — loop-shaped failures (3+ repeated actions,
    oscillation, response/action mismatch).
  * `profiling` — quantitative failures (token-per-step ratio out of
    line, retry storms, budget thrash).

This first version (chunk 3, 2026-05-26) implements the **hybrid**
strategy from `docs/PHASE_4_CHUNK_3_DESIGN.md` §Recommended: a cheap
deterministic heuristic on per-task aggregates from the trace bundle.
Light tier only. Primary-tier LLM tie-break is a follow-up.

Inputs the router uses (kept narrow so the surface stays testable):

  - `per_task_rewards: dict[task_id, float]` — from `BundleStats`.
  - `per_task_n_episodes: dict[task_id, int]` — how many replicas
    landed for this task this iter.
  - `per_task_max_repeat: dict[task_id, int] | None` — optional;
    longest consecutive same-action run observed in any of the task's
    episodes. When omitted, agent_scaffolding can't fire.
  - `per_task_tokens_per_step: dict[task_id, float] | None` — optional;
    when omitted, profiling can't fire.

The router can be invoked with just the first two inputs (rewards +
episode counts) and degrades to "all failures → hinter, with
general_blame fallback at SDK time." That's the safe default — never
worse than pre-chunk-3 behavior. Richer inputs unlock the L1 vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# The 4 Investigator recipes mirror upstream — keep this list aligned with
# `third_party/cube-harness/src/cube_harness/analyze/investigator/use_cases/`.
Recipe = Literal["hinter", "general_blame", "agent_scaffolding", "profiling"]
RECIPES: tuple[Recipe, ...] = ("hinter", "general_blame", "agent_scaffolding", "profiling")


# Heuristic thresholds. Bias toward `hinter` (the paper-faithful primary):
# only route to `agent_scaffolding` / `profiling` when the signal is strong.
# These are starting values; tune via the ledger's distribution observation
# after the first sweep with chunk 3 enabled.

# A task is "loop-shaped" only when ≥3 consecutive same-action invocations
# appeared in at least one episode. Upstream `agent_scaffolding/SKILL.md`
# cites "same action emitted 3+ times in a row" as the canonical trigger.
_MIN_REPEAT_FOR_SCAFFOLDING = 3

# Profiling fires when token consumption per step is unusually high — proxy
# for "agent retried a giant operation many times" or "context-window thrash."
# 50000 tokens per step is ~2x our typical TB step (~25k input + ~500 output);
# above that consistently signals quantitative pathology.
_HIGH_TOKENS_PER_STEP = 50_000


@dataclass(frozen=True)
class RouteDecision:
    """One per-task L1 dispatch decision."""

    task_id: str
    recipe: Recipe
    rationale: str  # short human-readable explanation; goes into the SDK prompt


def classify_recipe(
    task_id: str,
    *,
    reward: float,
    n_episodes: int,
    max_repeat: int | None = None,
    tokens_per_step: float | None = None,
    success_threshold: float = 0.5,
) -> RouteDecision:
    """L1 recipe routing for one task in this iter's bundle.

    Returns `RouteDecision`. Decision tree (first match wins):
      1. reward >= success_threshold → `hinter` (no failure to route;
         hinter is the paper-faithful default — it will produce no edits
         if the task is already passing).
      2. max_repeat >= _MIN_REPEAT_FOR_SCAFFOLDING → `agent_scaffolding`.
      3. tokens_per_step > _HIGH_TOKENS_PER_STEP → `profiling`.
      4. default → `hinter` (with `general_blame` fallback applied at
         SDK time when hinter returns empty `task_hints`).

    The fallback in step 4 is documented in upstream's
    `investigator/use_cases/hinter/SKILL.md`: empty `task_hints[]` on a
    failed episode is a strong signal that the failure is a real bug
    (tool / eval / scaffolding) and should route to `debug` instead.
    PGEPA's SDK system prompt will encode this fallback explicitly.
    """
    if reward >= success_threshold:
        return RouteDecision(
            task_id=task_id,
            recipe="hinter",
            rationale=f"reward={reward:.2f} (success); hinter defaults to no-edit on passes",
        )

    if max_repeat is not None and max_repeat >= _MIN_REPEAT_FOR_SCAFFOLDING:
        return RouteDecision(
            task_id=task_id,
            recipe="agent_scaffolding",
            rationale=(
                f"failed (reward={reward:.2f}); observed {max_repeat} consecutive "
                f"same-action invocations (≥{_MIN_REPEAT_FOR_SCAFFOLDING}) — "
                f"loop-shaped, routes to scaffolding diagnosis not a hint"
            ),
        )

    if tokens_per_step is not None and tokens_per_step > _HIGH_TOKENS_PER_STEP:
        return RouteDecision(
            task_id=task_id,
            recipe="profiling",
            rationale=(
                f"failed (reward={reward:.2f}); tokens_per_step={tokens_per_step:.0f} "
                f"exceeds {_HIGH_TOKENS_PER_STEP} threshold — quantitative pathology, "
                f"routes to profile not a hint"
            ),
        )

    return RouteDecision(
        task_id=task_id,
        recipe="hinter",
        rationale=(
            f"failed (reward={reward:.2f}, n_episodes={n_episodes}); "
            f"default route — hinter primary, general_blame fallback if "
            f"hinter returns empty task_hints"
        ),
    )


def route_iter_tasks(
    *,
    per_task_rewards: dict[str, float],
    per_task_n_episodes: dict[str, int] | None = None,
    per_task_max_repeat: dict[str, int] | None = None,
    per_task_tokens_per_step: dict[str, float] | None = None,
    success_threshold: float = 0.5,
) -> dict[str, RouteDecision]:
    """Route every task in an iter's bundle. Returns task_id → RouteDecision.

    Missing optional inputs degrade gracefully — agent_scaffolding and
    profiling simply never fire if their inputs aren't provided. Reward
    is the only required input.
    """
    n_eps = per_task_n_episodes or {}
    max_reps = per_task_max_repeat or {}
    tpr = per_task_tokens_per_step or {}
    out: dict[str, RouteDecision] = {}
    for task_id, reward in per_task_rewards.items():
        out[task_id] = classify_recipe(
            task_id=task_id,
            reward=reward,
            n_episodes=n_eps.get(task_id, 1),
            max_repeat=max_reps.get(task_id),
            tokens_per_step=tpr.get(task_id),
            success_threshold=success_threshold,
        )
    return out


def summarize_routing(decisions: dict[str, RouteDecision]) -> dict[Recipe, int]:
    """Aggregate routing decisions by recipe — useful for telemetry."""
    counts: dict[Recipe, int] = {r: 0 for r in RECIPES}
    for d in decisions.values():
        counts[d.recipe] += 1
    return counts


def format_routing_for_prompt(decisions: dict[str, RouteDecision]) -> str:
    """Render the routing table for inclusion in the SDK user prompt.

    The SDK reads this section and, per the augmented system prompt
    (`_RECIPE_DISPATCH_INSTRUCTION` below), follows the matching
    recipe's discipline when authoring hints for each task.

    Format: simple markdown table; one row per task; recipe + rationale.
    """
    if not decisions:
        return ""
    lines = [
        "## Investigator recipe routing (L1) for this iter",
        "",
        "The outer-loop driver pre-classified each task's failure shape "
        "into one of the 4 Investigator recipes. Apply the matching "
        "recipe's discipline when authoring hints for each task. "
        "(See `auto_cube/use_cases/hinter/SKILL.md` and "
        "`analyze/investigator/use_cases/<recipe>/SKILL.md` for recipe semantics.)",
        "",
        "| task_id | recipe | rationale |",
        "|---|---|---|",
    ]
    for task_id in sorted(decisions.keys()):
        d = decisions[task_id]
        lines.append(f"| `{task_id}` | `{d.recipe}` | {d.rationale} |")
    counts = summarize_routing(decisions)
    summary_bits = ", ".join(f"{r}={n}" for r, n in counts.items() if n > 0)
    if summary_bits:
        lines.append("")
        lines.append(f"Summary: {summary_bits}.")
    return "\n".join(lines)


# Per-recipe authoring guidance fragments, mirrored from upstream
# SKILL.md files. Used by the outer-loop SDK driver to assemble the
# SDK system prompt's L1 dispatch instruction. Keep this in sync with
# upstream `analyze/investigator/use_cases/<recipe>/SKILL.md` when
# those evolve.
_RECIPE_DISPATCH_INSTRUCTION = """\
## Investigator-recipe-aware authoring (L1 dispatch)

The user prompt's "Investigator recipe routing" table tells you which
recipe to apply per task. The recipes and their disciplines (see
upstream `analyze/investigator/use_cases/<recipe>/SKILL.md` for full
spec):

- **`hinter`**: extract concrete, narrow hints that would plausibly let
  the agent succeed on a re-run. Hints land in
  `GennyConfig.task_hints[task_id]` (low-reg cheat) or higher-reg rungs
  per the regularization ladder (`benchmark_hint_prompt`,
  `task_clarification`, `description_overrides`). IF you'd return an
  empty hint list (the failure is a real bug — tool / eval /
  scaffolding), DO NOT invent a hint — instead, route the case to
  `general_blame` discipline.
- **`general_blame`**: closed-world blame attribution. Do not author a
  `task_hints` entry for this task. Instead, write the blame + evidence
  into the session journal so the user can investigate the right layer.
- **`agent_scaffolding`**: loop-shaped failure. Do not author a
  `task_hints` entry. The fix belongs in agent scaffolding (Genny
  config, step budget, prompt structure) — note the loop pattern in
  the session journal.
- **`profiling`**: quantitative failure (token thrash, budget bust).
  Do not author a `task_hints` entry. Suggest a budget change in the
  session journal.

The point of the L1 dispatch: **don't author a hint when the failure
isn't hint-shaped.** Hints that mask bugs are harmful — they inflate
the score without teaching the agent anything promotable up the ladder.
"""


__all__ = [
    "Recipe",
    "RECIPES",
    "RouteDecision",
    "classify_recipe",
    "route_iter_tasks",
    "summarize_routing",
    "format_routing_for_prompt",
    "_RECIPE_DISPATCH_INSTRUCTION",
]
