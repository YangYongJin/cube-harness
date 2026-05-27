"""EpisodeConfig planner — meta-exploration's per-episode policy.

The policy this module implements is the **information-gathering**
contract from
``cube_harness.auto_cube.use_cases.meta_exploration/SKILL.md``: per
task, pick the agent configuration whose trajectory will maximally
help the downstream pipeline (hinter authoring + Phase 2 promotion
gate) decide what to do next about this task.

This is a **heuristic v1** policy. The decision tree is pure-function
over ``PlannerState`` (per-task ledger history + recent reward
trajectory + budget remaining), so it's deterministic and unit-testable.
A future v2 would learn the policy from accumulated
``(state, config, outcome)`` triples via bandits over the structured
action space — same API shape, different internals.

The policy emits ``PlannerDecision`` per task. The decision carries
the picked ``EpisodeConfig`` PLUS a short rationale + the alternatives
considered, so post-hoc analysis can audit "would a different config
have been better here?" without re-running the policy.

This module **does not** invoke any LLM. The action menu is a
discrete bounded set of named ``EpisodeConfig`` constants; the
decision tree branches over ledger state. LLM tie-breaking on
ambiguous cases is a follow-up (called out in the SKILL.md primary-
tier opt-in).

What this module does NOT own:
  * actually running episodes with the picked configs (that's the
    outer-loop SDK driver, Pivot 7)
  * promoting config knobs to ``GennyConfig`` defaults via the re-test
    gate (that's ``promotion.py``)
  * authoring hint text (that's the ``hinter`` use case — composable
    via Option B, see SKILL.md)
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from cube_harness.meta_exploration.ledger import HintLedgerEntry
from cube_harness.meta_exploration.recipe_router import Recipe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EpisodeConfig — the action the policy emits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeConfig:
    """One per-episode configuration choice.

    All fields default to the "cheap baseline" — same as a vanilla
    ``noop`` run. The policy mutates fields by selecting a different
    named ``EpisodeConfig`` from the menu rather than constructing
    arbitrary combinations (keeps the action space bounded for ablation).

    The outer-loop SDK driver translates these fields into the
    appropriate ``GennyConfig`` / ``LLMConfig`` / ``PerturbationConfig``
    fields when constructing the per-episode agent. See Pivot 7.
    """

    model: str = "azure/gpt-5-mini"
    k_candidates: int = 1
    k_plans: int = 1
    perturbations: tuple[str, ...] = ()
    # Per-call bash timeout default; only meaningful for TB-style cubes.
    bash_default_timeout: int = 120
    # When set, overrides the L1 recipe_router's classification for this
    # episode. None → use the router's heuristic.
    investigator_recipe: Recipe | None = None
    enable_refiner: bool = False
    # When True, this episode is a re-test of a promotion candidate.
    # The outer-loop driver applies the candidate's promotion to the
    # agent config copy before running.
    apply_promotion: bool = False


# ---------------------------------------------------------------------------
# Named menu — the policy's action set
# ---------------------------------------------------------------------------

BASELINE = EpisodeConfig()
"""Cheap baseline. First attempt or fallback when no signal."""

WIDEN_SEARCH = EpisodeConfig(
    k_candidates=3,
    perturbations=("topk_branch",),
)
"""Wide action sampling + branching. Use when: no signal after 1 try;
hinter would benefit from seeing what the agent ALMOST did."""

ESCALATE_MODEL = EpisodeConfig(
    model="azure/gpt-5",
    k_candidates=1,
)
"""Bump to a stronger executor. Use when: near-miss observed (the
weaker model got partial reward); we want to disambiguate capability-
bound failure vs structural failure."""

ENABLE_REFINER = EpisodeConfig(
    enable_refiner=True,
)
"""Mid-rollout LLM refiner. Use when: prior trajectory shows the
agent stuck/looping; want to know if a goal-restating user-message
would have unstuck it."""

RETEST_PROMOTION = EpisodeConfig(
    k_candidates=1,
    apply_promotion=True,
)
"""Cheap re-test of a promoted hint candidate. Use when: this task
has been ``cheat_only`` across iters and we want to test promotion
generalization."""

EXTEND_TIMEOUT = EpisodeConfig(
    bash_default_timeout=600,
)
"""Bumped bash timeout. Use when: prior trajectory shows pip-install
/ build commands hitting the 120s default."""

DIAGNOSE_SCAFFOLDING = EpisodeConfig(
    k_candidates=1,
    investigator_recipe="agent_scaffolding",
)
"""Force scaffolding-recipe routing. Use when: hinter recipe keeps
returning empty (failure is structural, not steerable)."""

DIAGNOSE_PROFILING = EpisodeConfig(
    k_candidates=1,
    investigator_recipe="profiling",
)
"""Minimum-cost confirmation config. Use when: budget tight OR task
streak-failed and we want a cheap final check before marking
unsteerable. Profiling recipe is the cheapest Investigator dispatch
(no hint authoring, just quantitative analysis)."""


EPISODE_CONFIG_MENU: dict[str, EpisodeConfig] = {
    "BASELINE": BASELINE,
    "WIDEN_SEARCH": WIDEN_SEARCH,
    "ESCALATE_MODEL": ESCALATE_MODEL,
    "ENABLE_REFINER": ENABLE_REFINER,
    "RETEST_PROMOTION": RETEST_PROMOTION,
    "EXTEND_TIMEOUT": EXTEND_TIMEOUT,
    "DIAGNOSE_SCAFFOLDING": DIAGNOSE_SCAFFOLDING,
    "DIAGNOSE_PROFILING": DIAGNOSE_PROFILING,
}


# Approximate per-episode cost (USD) for the action-budget check.
# Order-of-magnitude — refined empirically; not load-bearing for the
# policy's correctness, only for "would this config blow the budget?"
_APPROX_EPISODE_COST_USD: dict[str, float] = {
    "BASELINE": 0.05,
    "DIAGNOSE_PROFILING": 0.03,
    "WIDEN_SEARCH": 0.15,
    "ESCALATE_MODEL": 0.50,
    "ENABLE_REFINER": 0.08,
    "RETEST_PROMOTION": 0.05,
    "EXTEND_TIMEOUT": 0.10,
    "DIAGNOSE_SCAFFOLDING": 0.04,
}


# ---------------------------------------------------------------------------
# State the policy reads
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannerState:
    """Per-task input to the policy.

    ``ledger`` is the post-Pivot-2 cross-session ledger; the planner
    reads ``disposition`` + ``hint_type`` + ``notes`` for each task.
    ``recent_rewards`` is the window of the last K iters' rewards for
    each task (most-recent last). ``budget_remaining_usd`` is the
    policy's per-iter spend cap.
    """

    cube: str
    ledger: dict[str, HintLedgerEntry] = field(default_factory=dict)
    recent_rewards: dict[str, list[float]] = field(default_factory=dict)
    budget_remaining_usd: float = 1e9

    def _key(self, task_id: str) -> str:
        return f"{self.cube}|{task_id}"

    def disposition(self, task_id: str) -> str:
        entry = self.ledger.get(self._key(task_id))
        return entry.disposition if entry else "open"

    def n_attempts(self, task_id: str) -> int:
        return len(self.recent_rewards.get(task_id, []))

    def rewards(self, task_id: str) -> list[float]:
        return list(self.recent_rewards.get(task_id, []))


# ---------------------------------------------------------------------------
# Policy output
# ---------------------------------------------------------------------------


Outcome = Literal["picked", "fallback"]


@dataclass(frozen=True)
class PlannerDecision:
    """One per-task policy output, written to ``iter_<k>/plan.json``."""

    task_id: str
    config_name: str
    config: EpisodeConfig
    rationale: str
    # ``alternatives_considered``: list of (config_name, why_not_picked)
    # — lets post-hoc analysis see what the policy thought about.
    alternatives_considered: tuple[tuple[str, str], ...] = ()
    # Tag for what info this choice is meant to surface (free-form for
    # v1; could become a structured enum if useful for analysis).
    expected_information_value: str = ""


# ---------------------------------------------------------------------------
# The policy — heuristic decision tree
# ---------------------------------------------------------------------------


_SUCCESS_THRESHOLD = 0.5
_NEAR_MISS_LOWER = 0.0  # exclusive
_NEAR_MISS_UPPER = 0.5  # exclusive


def pick_episode_config(
    task_id: str,
    state: PlannerState,
) -> PlannerDecision:
    """Per-task policy. Pure function of ``(task_id, state)``.

    Decision tree (first match wins):
      1. ``apply_promotion`` opportunity: disposition == cheat_only AND
         ≥2 attempts → RETEST_PROMOTION (cheap; the gate decides).
      2. Brand new task (no attempts) → BASELINE.
      3. Near-miss observed (any reward in (0, 0.5)) → ESCALATE_MODEL
         (disambiguate capability vs structural).
      4. Stuck streak (≥3 attempts, all 0.0) → DIAGNOSE_PROFILING +
         signal to caller to consider unsteerable disposition.
      5. One try, no signal → WIDEN_SEARCH (k=3 + perturbation).
      6. Notes flag a build/install timeout → EXTEND_TIMEOUT.
      7. Notes flag a loop pattern → DIAGNOSE_SCAFFOLDING.
      8. Default → BASELINE.

    Budget check: if the picked config's approx cost exceeds
    ``state.budget_remaining_usd``, downgrade to the next-cheapest
    config that still preserves the decision's intent (typically
    BASELINE or DIAGNOSE_PROFILING).

    Returns a ``PlannerDecision`` with rationale + alternatives. Never
    raises — degrades to BASELINE on any unexpected state.
    """
    disposition = state.disposition(task_id)
    rewards = state.rewards(task_id)
    n_attempts = state.n_attempts(task_id)

    # Notes flag inspection — read the ledger entry for hints from
    # prior Investigator findings.
    entry = state.ledger.get(state._key(task_id))
    notes = entry.notes if entry else {}
    saw_build_timeout = bool(notes.get("build_timeout_suspected"))
    saw_loop_pattern = bool(notes.get("loop_pattern_suspected"))

    alternatives: list[tuple[str, str]] = []

    # Rule 1 — promotion opportunity.
    if disposition == "cheat_only" and n_attempts >= 2:
        return _apply_budget_check(
            state,
            picked="RETEST_PROMOTION",
            rationale=(
                f"disposition=cheat_only for {n_attempts} attempts; "
                f"trigger Phase 2 re-test to decide promote vs cheat_only-stays."
            ),
            info_value="promotion_gate_outcome",
            alternatives=_record(
                alternatives,
                BASELINE="cheat_only with no new info wouldn't change disposition",
                WIDEN_SEARCH="more breadth on a confirmed-cheat doesn't advance disposition",
            ),
        )

    # Rule 2 — brand new task.
    if n_attempts == 0:
        return _apply_budget_check(
            state,
            picked="BASELINE",
            rationale="first attempt; baseline config establishes the prior.",
            info_value="baseline_trajectory",
            alternatives=_record(
                alternatives,
                WIDEN_SEARCH="expensive when we have no prior; default first",
                ESCALATE_MODEL="too costly for an unprobed task",
            ),
        )

    # Rule 3 — near-miss with weak model.
    if any(_NEAR_MISS_LOWER < r < _NEAR_MISS_UPPER for r in rewards):
        return _apply_budget_check(
            state,
            picked="ESCALATE_MODEL",
            rationale=(
                f"observed near-miss rewards {rewards}; "
                f"bump to stronger model to disambiguate capability- vs "
                f"structural-bound failure."
            ),
            info_value="capability_vs_structural",
            alternatives=_record(
                alternatives,
                WIDEN_SEARCH="breadth wouldn't tell us if capability is the bound",
                ENABLE_REFINER="refiner addresses scaffolding, not capability",
            ),
        )

    # Rule 4 — stuck failure streak.
    if n_attempts >= 3 and all(r <= 0 for r in rewards):
        return _apply_budget_check(
            state,
            picked="DIAGNOSE_PROFILING",
            rationale=(
                f"{n_attempts} attempts with all-zero rewards; cheap confirmation before committing to unsteerable."
            ),
            info_value="unsteerable_confirmation",
            alternatives=_record(
                alternatives,
                ESCALATE_MODEL="reasonable to try once but expensive after many fails",
                DIAGNOSE_SCAFFOLDING="only if we suspect loops; default to cheap",
            ),
        )

    # Rule 5 — one fail with default, time to explore.
    if n_attempts >= 1 and not any(r >= _SUCCESS_THRESHOLD for r in rewards):
        # Refine based on notes if available.
        if saw_loop_pattern:
            return _apply_budget_check(
                state,
                picked="DIAGNOSE_SCAFFOLDING",
                rationale=("prior trajectory flagged loop pattern; route to agent_scaffolding recipe + cheap config."),
                info_value="scaffolding_diagnosis",
                alternatives=_record(
                    alternatives,
                    ENABLE_REFINER="refiner could unstick a loop — alternative worth trying next",
                ),
            )
        if saw_build_timeout:
            return _apply_budget_check(
                state,
                picked="EXTEND_TIMEOUT",
                rationale=("prior trajectory flagged build/install timeout; bump bash default to 600s."),
                info_value="timeout_recovery",
                alternatives=_record(
                    alternatives,
                    BASELINE="re-running at 120s would just re-time-out",
                ),
            )
        return _apply_budget_check(
            state,
            picked="WIDEN_SEARCH",
            rationale=(
                "first failure with default config; broaden action "
                "sampling (k=3 + topk_branch) so hinter sees what the "
                "agent almost did."
            ),
            info_value="alternative_actions_surfaced",
            alternatives=_record(
                alternatives,
                ESCALATE_MODEL="cheaper to test breadth first; stronger model is escalation",
                ENABLE_REFINER="alternative — only if first failure looked stuck/looping",
            ),
        )

    # Rule 8 — fallback.
    return _apply_budget_check(
        state,
        picked="BASELINE",
        rationale="no rule matched; default config.",
        info_value="baseline_trajectory",
        alternatives=alternatives,
    )


def _record(
    alternatives: list[tuple[str, str]],
    **rejections: str,
) -> tuple[tuple[str, str], ...]:
    """Helper: capture (config_name, why_not_picked) pairs."""
    for name, why in rejections.items():
        alternatives.append((name, why))
    return tuple(alternatives)


def _apply_budget_check(
    state: PlannerState,
    *,
    picked: str,
    rationale: str,
    info_value: str,
    alternatives: tuple[tuple[str, str], ...],
) -> PlannerDecision:
    """If ``picked`` exceeds remaining budget, downgrade to cheapest."""
    cost = _APPROX_EPISODE_COST_USD.get(picked, 0.10)
    if cost > state.budget_remaining_usd:
        fallback = "DIAGNOSE_PROFILING"
        rationale_aug = (
            f"{rationale} | DOWNGRADED to {fallback}: picked={picked!r} "
            f"approx ${cost:.2f} > budget ${state.budget_remaining_usd:.2f}"
        )
        alt_aug = alternatives + (
            (picked, f"would have picked but budget too tight (${cost:.2f} > ${state.budget_remaining_usd:.2f})"),
        )
        return PlannerDecision(
            task_id="",  # caller fills in
            config_name=fallback,
            config=EPISODE_CONFIG_MENU[fallback],
            rationale=rationale_aug,
            alternatives_considered=alt_aug,
            expected_information_value="budget_constrained",
        )

    return PlannerDecision(
        task_id="",  # caller fills in
        config_name=picked,
        config=EPISODE_CONFIG_MENU[picked],
        rationale=rationale,
        alternatives_considered=alternatives,
        expected_information_value=info_value,
    )


# ---------------------------------------------------------------------------
# Iter-level — plan every task in one call
# ---------------------------------------------------------------------------


def plan_iter(
    task_ids: Iterable[str],
    state: PlannerState,
) -> dict[str, PlannerDecision]:
    """Plan the per-episode config for every task in this iter.

    Iter-level convenience. Calls ``pick_episode_config`` per task and
    fills in the ``task_id`` field of each ``PlannerDecision``. The
    returned dict is suitable for serialization to
    ``iter_<k>/plan.json``.

    The order of iteration over ``task_ids`` is preserved in the output
    (Python dicts are insertion-ordered). Budget bookkeeping is per-call
    — this version does NOT decrement state.budget_remaining_usd across
    tasks within the same iter (a real iter-level allocation pass would;
    deferred to v2).
    """
    out: dict[str, PlannerDecision] = {}
    for tid in task_ids:
        d = pick_episode_config(tid, state)
        out[tid] = PlannerDecision(
            task_id=tid,
            config_name=d.config_name,
            config=d.config,
            rationale=d.rationale,
            alternatives_considered=d.alternatives_considered,
            expected_information_value=d.expected_information_value,
        )
    return out


def summarize_plan(plan: dict[str, PlannerDecision]) -> dict[str, int]:
    """Aggregate plan by config_name — telemetry-friendly counts."""
    counts: dict[str, int] = {}
    for d in plan.values():
        counts[d.config_name] = counts.get(d.config_name, 0) + 1
    return counts


__all__ = [
    "BASELINE",
    "EPISODE_CONFIG_MENU",
    "WIDEN_SEARCH",
    "EpisodeConfig",
    "DIAGNOSE_PROFILING",
    "EXTEND_TIMEOUT",
    "PlannerDecision",
    "PlannerState",
    "ENABLE_REFINER",
    "RETEST_PROMOTION",
    "DIAGNOSE_SCAFFOLDING",
    "ESCALATE_MODEL",
    "pick_episode_config",
    "plan_iter",
    "summarize_plan",
]
