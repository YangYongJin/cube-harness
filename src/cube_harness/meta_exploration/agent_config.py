"""MetaExplorationGennyConfig — pure-data extension of upstream ``GennyConfig``.

The meta-exploration planner picks per-episode configs (k_candidates,
k_plans, perturbations, refiner) that upstream ``GennyConfig`` doesn't
model. This subclass adds those fields as plain data — nothing more.

**Discipline pin (do NOT add methods or properties here).** This file
must stay a pure-data extension so the B→A migration is trivial: if
Alec eventually accepts an upstream RFC that folds these fields into
``GennyConfig``, the only diff is deleting this file and updating
imports. Any method override (or `@property`, or custom `__init__`)
breaks that property and entangles meta-exploration with Genny
internals. The discipline is enforced by
``tests/test_meta_exploration_agent_config.py``.

See ``auto_cube/use_cases/meta_exploration/DESIGN.md`` §10 Q1 for the
local-vs-upstream architectural decision (Option B).
"""

from __future__ import annotations

from pydantic import Field

from cube_harness.agents.genny import GennyConfig


class MetaExplorationGennyConfig(GennyConfig):
    """Pure-data extension. NO METHOD OVERRIDES — see module docstring."""

    k_candidates: int = 1
    """Number of action candidates the agent samples per step. >1 enables
    the K-candidate exploration knob (mechanical channel; algorithm port
    pending — see HANDOFF.md Tier 3 C1)."""

    k_plans: int = 1
    """Number of strategy/plan candidates the agent considers before
    committing to one. >1 enables plan-candidate exploration (mechanical
    channel; algorithm port pending — HANDOFF.md Tier 3 C2)."""

    enable_refiner: bool = False
    """Mid-rollout refiner pass — when True, inject diagnostic notes
    partway through the trajectory (mechanical channel; algorithm port
    pending — HANDOFF.md Tier 3 C4)."""

    perturbations: list[str] = Field(default_factory=list)
    """Per-step perturbation hooks (e.g. ``["topk_branch"]``,
    ``["plan_swap"]``). Empty list = no perturbations (mechanical channel;
    algorithm port pending — HANDOFF.md Tier 3 C3)."""


__all__ = ["MetaExplorationGennyConfig"]
