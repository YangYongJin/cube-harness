"""Promotion-candidate detection + re-test gate.

Auto-CUBE's hinter use case has a discipline gate between **steering**
(per-task low-reg cheat) and **promotion** (high-reg writes that ship
to the benchmark / tool / agent layer). The gate: before a hint is
promoted up the ladder to ``benchmark_hint_prompt`` /
``task_clarification`` / ``description_overrides``, run a small re-test
to confirm it (a) still makes the originally-affected tasks pass, and
(b) doesn't regress any held-out tasks.

This module owns the **promotion side**; the steering side is the
per-iter hint-update step in whatever outer-loop driver runs it. The
split is the Phase 1 / Phase 2 of ``auto_cube/use_cases/hinter/SKILL.md``.

Pure-function building blocks shipped here:
  * ``PromotionCandidate`` dataclass + ``_stable_id`` for deterministic IDs
  * ``pick_held_out`` — seeded held-out slice selection
  * ``decide_verdict`` — pass/fail gate logic
  * ``apply_promotion_to_config`` — mutates a ``GennyConfig`` to apply
    the proposed promotion (currently v1: ``benchmark_hint_prompt`` only)

What this module does NOT own (Pivot 3+ design call):
  * ``detect_promotion_candidates`` is deliberately left as a stub. The
    input shape depends on the outer-loop driver's hint-state model:
    - upstream auto-cube hinter rounds: round-by-round ``exp_config.py``
      diffs against the previous round's config
    - meta-exploration SDK driver: per-iter ``GennyConfig.task_hints``
      diffs against the previous iter's snapshot
    Both reduce to the same candidate type but the diff-source is
    different. Wire the detector at the call site rather than locking
    a shape here.
"""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass, field
from typing import Literal

from cube_harness.meta_exploration.ledger import PROMOTED_TO_VALUES, PromotedTo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Promotion candidate detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromotionCandidate:
    """One proposed promotion derived from the iter's hint diff."""

    candidate_id: str  # stable id: <bench>:<rung>:<short-text-hash>
    rung: PromotedTo  # which ladder rung (e.g. 'benchmark_hint_prompt')
    hint_text: str  # the verbatim text that would be applied
    source_keys: list[str]  # which 5-bucket keys this candidate aggregates
    # (e.g. ['SHARED_PATTERN[universal-pip-timeout]'])
    affected_task_ids: list[str]  # tasks the hint was scoped to (or all-tasks for wildcard)
    rationale: str  # short human-readable explanation


def stable_id(bench: str, rung: PromotedTo, hint_text: str) -> str:
    """Deterministic id from (bench, rung, normalized text).

    Same hint text in two iters → same candidate_id; comparable across
    sessions via the ledger.
    """
    h = hashlib.sha1((bench + "|" + rung + "|" + hint_text.strip()).encode("utf-8")).hexdigest()[:10]
    return f"{bench}:{rung}:{h}"


def candidate_from_task_hints_overlap(
    *,
    benchmark: str,
    task_hints_after: dict[str, str],
    task_hints_before: dict[str, str] | None = None,
    per_task_rewards: dict[str, float],
    min_observations: int = 3,
) -> list[PromotionCandidate]:
    """Detect promotion candidates from ``GennyConfig.task_hints`` overlap.

    The simplest detection rule: if the same hint text appears in
    ``min_observations`` or more ``task_hints[task_id]`` entries that
    are new or edited this iter, it's a candidate for promotion to
    ``benchmark_hint_prompt`` (a benchmark-wide overlay everyone sees).

    Designed for the meta-exploration SDK driver which mutates
    ``GennyConfig.task_hints`` directly (no upstream PR for the
    promotion itself — that's Mode C's job to ship). Pass
    ``task_hints_before=None`` on the first iter (no prior state to
    diff against; every hint is "new").

    Returns deduplicated list of candidates. Empty list when no
    overlap reaches ``min_observations``.

    For other rungs (``task_clarification``, ``description_overrides``,
    etc.), write a sibling detector — keep this one focused on the
    text-overlap → ``benchmark_hint_prompt`` rule.
    """
    if not task_hints_after:
        return []
    before = task_hints_before or {}

    # Group tasks by their hint text (post-iter view). Only consider
    # entries that are NEW or CHANGED this iter — unchanged hints were
    # already in steady-state and shouldn't trigger a fresh promotion.
    text_to_tasks: dict[str, list[str]] = {}
    for tid, text in task_hints_after.items():
        clean = (text or "").strip()
        if not clean:
            continue
        if before.get(tid, "").strip() == clean:
            continue  # unchanged → not a fresh candidate
        text_to_tasks.setdefault(clean, []).append(tid)

    candidates: list[PromotionCandidate] = []
    for text, tids in text_to_tasks.items():
        if len(tids) < min_observations:
            continue
        affected = sorted(tids)
        # Affected scope = the tasks that already had this hint applied.
        # Held-out for re-test is picked separately by ``pick_held_out``.
        candidates.append(
            PromotionCandidate(
                candidate_id=stable_id(benchmark, "benchmark_hint_prompt", text),
                rung="benchmark_hint_prompt",
                hint_text=text,
                source_keys=[f"task_hints[{tid!r}]" for tid in affected],
                affected_task_ids=affected,
                rationale=(
                    f"Same hint text appears in {len(tids)} task_hints "
                    f"entries this iter (≥{min_observations} threshold); "
                    f"promote to benchmark_hint_prompt so it applies "
                    f"benchmark-wide instead of per-task fall-through."
                ),
            )
        )

    # Deduplicate by candidate_id (defensive — same text won't hash twice).
    seen: set[str] = set()
    unique: list[PromotionCandidate] = []
    for c in candidates:
        if c.candidate_id in seen:
            continue
        seen.add(c.candidate_id)
        unique.append(c)
    return unique


# ---------------------------------------------------------------------------
# Re-test slice selection
# ---------------------------------------------------------------------------


def pick_held_out(
    *,
    candidate: PromotionCandidate,
    per_task_rewards: dict[str, float],
    n_held_out: int,
    seed: int | None = None,
) -> list[str]:
    """Pick a small held-out task slice for the re-test gate.

    Picked from `per_task_rewards.keys()` EXCLUDING `candidate.affected_task_ids`.
    Seeded RNG for reproducibility — pass `seed=iter_idx` to vary across
    iters while staying deterministic per-iter.

    If the held-out pool is smaller than `n_held_out`, returns the whole
    pool. If the pool is empty (e.g. the candidate affects every task
    in the iter), returns [] — caller should treat empty held-out as
    "skip held-out check, only affected-pass matters."
    """
    affected = set(candidate.affected_task_ids)
    pool = sorted([t for t in per_task_rewards.keys() if t not in affected])
    if not pool:
        return []
    rng = random.Random(seed)
    if len(pool) <= n_held_out:
        return pool
    return rng.sample(pool, n_held_out)


# ---------------------------------------------------------------------------
# Promotion outcome (verdict from the re-test gate)
# ---------------------------------------------------------------------------


PromoteOutcome = Literal["ok", "affected_regressed", "held_out_regressed", "skipped"]


@dataclass(frozen=True)
class PromoteVerdict:
    """Outcome of one `apply_promotion_then_retest` call."""

    candidate_id: str
    outcome: PromoteOutcome
    rung: PromotedTo
    affected_pre_rewards: dict[str, float] = field(default_factory=dict)
    affected_post_rewards: dict[str, float] = field(default_factory=dict)
    held_out_pre_rewards: dict[str, float] = field(default_factory=dict)
    held_out_post_rewards: dict[str, float] = field(default_factory=dict)
    reason: str = ""


def decide_verdict(
    *,
    candidate: PromotionCandidate,
    affected_pre: dict[str, float],
    affected_post: dict[str, float],
    held_out_pre: dict[str, float],
    held_out_post: dict[str, float],
    success_threshold: float = 0.5,
) -> PromoteVerdict:
    """Score the re-test results. Pure function — call after running the
    re-test sub-experiment.

    Pass criteria:
      1. **affected_pass**: every previously-passing affected task still passes
         (`post >= pre` for tasks with `pre >= success_threshold`); plus, at
         least one affected task with `pre < success_threshold` now passes
         (the candidate must actually steer SOMETHING).
      2. **held_out_pass**: no held-out task regresses past `success_threshold`
         (`post >= pre` when `pre >= success_threshold`).

    Returns a `PromoteVerdict` recording all four pre/post score dicts
    + the outcome + a short reason string.
    """
    # 1. Affected: previously-passing tasks must stay passing.
    affected_regressed: list[str] = []
    affected_flipped_to_pass: list[str] = []
    for tid in candidate.affected_task_ids:
        pre = affected_pre.get(tid, 0.0)
        post = affected_post.get(tid, 0.0)
        if pre >= success_threshold and post < success_threshold:
            affected_regressed.append(tid)
        if pre < success_threshold and post >= success_threshold:
            affected_flipped_to_pass.append(tid)

    if affected_regressed:
        return PromoteVerdict(
            candidate_id=candidate.candidate_id,
            outcome="affected_regressed",
            rung=candidate.rung,
            affected_pre_rewards=affected_pre,
            affected_post_rewards=affected_post,
            held_out_pre_rewards=held_out_pre,
            held_out_post_rewards=held_out_post,
            reason=(f"{len(affected_regressed)} previously-passing affected task(s) regressed: {affected_regressed}"),
        )

    # 2. Held-out: previously-passing tasks must stay passing.
    held_out_regressed: list[str] = []
    for tid, pre in held_out_pre.items():
        post = held_out_post.get(tid, 0.0)
        if pre >= success_threshold and post < success_threshold:
            held_out_regressed.append(tid)
    if held_out_regressed:
        return PromoteVerdict(
            candidate_id=candidate.candidate_id,
            outcome="held_out_regressed",
            rung=candidate.rung,
            affected_pre_rewards=affected_pre,
            affected_post_rewards=affected_post,
            held_out_pre_rewards=held_out_pre,
            held_out_post_rewards=held_out_post,
            reason=(f"{len(held_out_regressed)} held-out task(s) regressed under promotion: {held_out_regressed}"),
        )

    # PASS.
    return PromoteVerdict(
        candidate_id=candidate.candidate_id,
        outcome="ok",
        rung=candidate.rung,
        affected_pre_rewards=affected_pre,
        affected_post_rewards=affected_post,
        held_out_pre_rewards=held_out_pre,
        held_out_post_rewards=held_out_post,
        reason=(
            f"affected: no regressions; {len(affected_flipped_to_pass)} flipped to pass. "
            f"held_out ({len(held_out_pre)} tasks): no regressions."
        ),
    )


# ---------------------------------------------------------------------------
# Apply a promotion to an agent config (the SDK-driver re-test pass)
# ---------------------------------------------------------------------------


def apply_promotion_to_config(
    agent_config,  # GennyConfig (typed loosely to avoid the agents import here)
    candidate: PromotionCandidate,
) -> None:
    """Mutate ``agent_config`` IN-PLACE to apply the proposed promotion.

    v1 supports only ``rung == 'benchmark_hint_prompt'`` — sets/appends
    ``GennyConfig.benchmark_hint_prompt``. Other rungs raise
    ``NotImplementedError`` (they need their own wiring per rung; the
    higher rungs like ``task_clarification`` / ``description_overrides``
    /  ``new_action`` / ``system_prompt`` are typically Mode-C-only
    because they ship as PRs to the cube / tool / cube-harness).

    Why in-place: ``GennyConfig`` is a Pydantic model. The caller
    constructs a fresh copy (``.model_copy(deep=True)``) for the
    re-test pass so the pre-test config doesn't change.
    """
    if candidate.rung != "benchmark_hint_prompt":
        raise NotImplementedError(
            f"promotion to rung {candidate.rung!r} not yet wired (v1 supports "
            f"only 'benchmark_hint_prompt'); see docs/PHASE_4_CHUNK_4_DESIGN.md"
        )

    # Set or extend the benchmark_hint_prompt. If one is already set,
    # append the new text on a new paragraph rather than overwriting —
    # multiple universals can coexist on the prompt.
    existing = (getattr(agent_config, "benchmark_hint_prompt", "") or "").strip()
    if existing:
        agent_config.benchmark_hint_prompt = f"{existing}\n\n{candidate.hint_text}"
    else:
        agent_config.benchmark_hint_prompt = candidate.hint_text


__all__ = [
    "PROMOTED_TO_VALUES",  # re-exported for convenience
    "PromoteOutcome",
    "PromoteVerdict",
    "PromotionCandidate",
    "apply_promotion_to_config",
    "candidate_from_task_hints_overlap",
    "decide_verdict",
    "pick_held_out",
    "stable_id",
]
