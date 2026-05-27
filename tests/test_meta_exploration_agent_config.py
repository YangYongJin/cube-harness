"""Tests for ``MetaExplorationGennyConfig`` — the pure-data extension of
``GennyConfig`` (Tier 1 #2). The headline test is the discipline pin:
no methods may be added in the subclass, so the B→A migration (folding
these fields upstream into ``GennyConfig``) stays trivial.
"""

from __future__ import annotations

from cube_harness.agents.genny import GennyConfig
from cube_harness.llm import LLMConfig
from cube_harness.meta_exploration.agent_config import MetaExplorationGennyConfig

# ---------------------------------------------------------------------------
# DISCIPLINE PIN — no method overrides allowed
# ---------------------------------------------------------------------------


_EXPECTED_NEW_FIELDS: frozenset[str] = frozenset({"k_candidates", "k_plans", "enable_refiner", "perturbations"})


def _own_callable_members(cls: type) -> set[str]:
    """Names callable on the class itself (not inherited) — methods,
    properties, staticmethods, classmethods. Ignores dunders + Pydantic
    plumbing (``model_*`` descriptors that Pydantic generates on every
    subclass)."""
    out: set[str] = set()
    for name, value in vars(cls).items():
        if name.startswith("_") or name.startswith("model_"):
            continue
        if callable(value) or isinstance(value, (property, staticmethod, classmethod)):
            out.add(name)
    return out


def test_no_method_overrides_in_subclass() -> None:
    """The B→A migration relies on this. If the subclass adds a method
    or property, the upstream-RFC fold-back is no longer a trivial
    rename — it becomes a behavior port. Keep methods OUT of this file
    and add them upstream to GennyConfig instead (with Alec sign-off)."""
    parent_callables = _own_callable_members(GennyConfig)
    child_callables = _own_callable_members(MetaExplorationGennyConfig)
    added = child_callables - parent_callables
    assert added == set(), (
        f"MetaExplorationGennyConfig added callable members {added!r}. "
        f"This subclass must stay pure-data — move methods upstream into "
        f"GennyConfig (with RFC) rather than adding them here. See "
        f"src/cube_harness/meta_exploration/agent_config.py docstring."
    )


def test_subclass_adds_exactly_the_expected_fields() -> None:
    """The Pydantic ``model_fields`` diff between parent and child must
    be exactly the four new mechanical-knob fields — no more, no less.
    A surprise field appearing here means someone added a knob without
    updating DESIGN.md / HANDOFF.md; flag it for review."""
    parent_fields = set(GennyConfig.model_fields.keys())
    child_fields = set(MetaExplorationGennyConfig.model_fields.keys())
    added = child_fields - parent_fields
    assert added == set(_EXPECTED_NEW_FIELDS), (
        f"MetaExplorationGennyConfig adds {added!r}, expected "
        f"{set(_EXPECTED_NEW_FIELDS)!r}. Update _EXPECTED_NEW_FIELDS "
        f"AND DESIGN.md §9 if this change is deliberate."
    )


# ---------------------------------------------------------------------------
# Field defaults pin the documented Auto-CUBE Options story
# ---------------------------------------------------------------------------


def _minimal_config(**overrides: object) -> MetaExplorationGennyConfig:
    """A valid instance with the smallest reasonable surface. We need
    an ``LLMConfig`` since ``GennyConfig.llm_config`` is required."""
    return MetaExplorationGennyConfig(
        llm_config=LLMConfig(model_name="azure/gpt-5-mini"),
        **overrides,  # type: ignore[arg-type]
    )


def test_defaults_match_noop_behavior() -> None:
    """Defaults must make the subclass behave like vanilla GennyConfig
    when no mechanical knobs are explicitly set — otherwise plain
    instantiation would silently change agent behavior."""
    cfg = _minimal_config()
    assert cfg.k_candidates == 1
    assert cfg.k_plans == 1
    assert cfg.enable_refiner is False
    assert cfg.perturbations == []


def test_perturbations_default_is_a_fresh_list() -> None:
    """Mutable-default footgun guard: two instances must NOT share the
    same list object, otherwise mutating one config's perturbations
    leaks into the other (classic Pydantic-with-default-factory bug if
    misused)."""
    a = _minimal_config()
    b = _minimal_config()
    a.perturbations.append("topk_branch")
    assert b.perturbations == [], f"perturbations default leaked across instances: b.perturbations={b.perturbations!r}"


# ---------------------------------------------------------------------------
# It is a real subclass — instances pass GennyConfig type checks
# ---------------------------------------------------------------------------


def test_is_instance_of_gennyconfig() -> None:
    """Downstream code that types on ``GennyConfig`` must accept this."""
    cfg = _minimal_config()
    assert isinstance(cfg, GennyConfig)


def test_subclass_can_be_constructed_with_new_knobs() -> None:
    """Smoke: passing the new fields explicitly works end-to-end (model
    validation + dataclass-style attribute access)."""
    cfg = _minimal_config(
        k_candidates=3,
        k_plans=2,
        enable_refiner=True,
        perturbations=["topk_branch", "plan_swap"],
    )
    assert cfg.k_candidates == 3
    assert cfg.k_plans == 2
    assert cfg.enable_refiner is True
    assert cfg.perturbations == ["topk_branch", "plan_swap"]


def test_inherited_gennyconfig_fields_still_accessible() -> None:
    """The subclass must inherit all of GennyConfig's fields — that's
    what makes it a drop-in replacement for upstream consumers."""
    cfg = _minimal_config()
    assert cfg.task_hints == {}
    assert cfg.task_clarification == {}
    assert cfg.benchmark_hint_prompt is None
    assert cfg.flat_history is False
