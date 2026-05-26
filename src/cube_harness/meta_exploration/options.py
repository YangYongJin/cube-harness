"""AutoCubeOptions — the ablation framework.

Different prior-art methods (Meta-Harness, hinter) and our proposal
(meta-exploration) are **named configurations of this single options
struct**, not separate code paths. This is what makes the 6-cell
ablation table from
`auto_cube/use_cases/meta_exploration/DESIGN.md` §2 fall out of the
architecture without per-experiment glue code.

The orchestrator (outer-loop SDK driver, Pivot 7) reads an
``AutoCubeOptions`` instance at iter start and conditionally fires each
stage. Modules consume specific fields:

  Stage A (per-episode config policy) — ``planner.py``  ← reads
    ``enable_config_policy``, ``allow_model_escalation``.
  Stage C (per-trajectory Investigator dispatch) — ``recipe_router.py``
    ← reads ``enable_l1_recipe_routing``.
  Stage D (hint authoring) — orchestrator routes findings into hinter
    when ``enable_hint_authoring``.
  Stage E (Phase 2 promotion gate) — ``promotion.py`` ← reads
    ``enable_text_promotion`` / ``enable_config_promotion``,
    ``held_out_size``, ``min_observations_for_promotion``.
  Stage F (ledger writes) — ``ledger.py`` ← gated by
    ``enable_ledger_writes``.

**Honest training/inference split** (DESIGN.md §1) is code-enforced
here: ``inference_model`` is a separate field; ``assert_inference_model``
runs in the re-test gate to verify that promotion verdicts came from
that model, not from a training-time ``ESCALATE_MODEL`` trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cube_harness.meta_exploration.planner import EPISODE_CONFIG_MENU, EpisodeConfig


# ---------------------------------------------------------------------------
# AutoCubeOptions — the union of all toggleable knobs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AutoCubeOptions:
    """One immutable bundle of ablation toggles + base settings.

    Named recipes below (``WEAK_NOOP``, ``STRONG_NOOP``, ``META_HARNESS``,
    ``HINTER_ONLY``, ``META_EXPLORATION_ONLY``, ``COMBINED``) instantiate
    this struct with specific knob combinations. Add new recipes by
    constructing a new instance; don't subclass.
    """

    # ---- Inference / evaluation model -------------------------------------
    # The fixed baseline executor used for final-eval runs AND for the
    # Phase 2 re-test gate. Promotion verdicts can ONLY be approved when
    # the verdict trajectories were generated under this model.
    inference_model: str = "azure/gpt-5-mini"

    # ---- Stage A — per-episode config policy (meta-exploration) ----------
    enable_config_policy: bool = False
    """Turn on the EpisodeConfig planner per episode (Pivot 4)."""

    allow_model_escalation: bool = False
    """When True, the planner's menu includes ``ESCALATE_MODEL`` (the
    teacher tier). When False, that config is filtered out — keeps the
    sweep cost-bounded AND prevents accidental honest/dishonest mixups.
    Even when True, the re-test gate still uses ``inference_model``."""

    # ---- Stage D — text hint authoring (hinter) --------------------------
    enable_hint_authoring: bool = False
    """Turn on per-trajectory Investigator + hinter recipe to author
    text hints into ``GennyConfig.task_hints``."""

    # ---- Stage E — Phase 2 promotion (shared discipline gate) ------------
    enable_text_promotion: bool = False
    """Phase 2 re-test gate for text-hint promotion candidates."""

    enable_config_promotion: bool = False
    """Phase 2 re-test gate for config-knob promotion candidates."""

    # ---- Stage C / F — shared infrastructure -----------------------------
    enable_l1_recipe_routing: bool = True
    """Per-trajectory Investigator recipe routing
    (``recipe_router.py``). Default on — turning off forces every
    failure to the default recipe."""

    enable_ledger_writes: bool = True
    """``~/auto_cube/hints.json`` updates. Default on — turning off
    skips cross-session record-keeping (useful for one-shot runs
    where you don't want to perturb the shared ledger)."""

    held_out_size: int = 3
    """Held-out task slice size for the Phase 2 re-test gate
    (``promotion.py:pick_held_out``)."""

    min_observations_for_promotion: int = 3
    """Minimum observations before a hint is a promotion candidate
    (``promotion.py:candidate_from_task_hints_overlap``)."""

    # ---- Meta-Harness-specific knobs -------------------------------------
    # Lets us encode Meta-Harness (arXiv 2603.28052) as a named option
    # subset rather than a separate module. Algorithm implementations
    # live alongside the existing hinter machinery and check these flags.
    enable_pareto_verify: bool = False
    """Meta-Harness: verify Pareto improvement before applying any
    hint mutation. Stricter than PGEPA's default mean-reward check;
    stops regressing-some-tasks-while-improving-others updates."""

    propose_multiple_harnesses: bool = False
    """Meta-Harness: propose K candidate harnesses per pass; pick the
    best by validation reward. Cost ≈ K× per-pass."""

    n_proposed_harnesses: int = 4
    """K when ``propose_multiple_harnesses`` is True. Ignored otherwise."""


# ---------------------------------------------------------------------------
# Named recipes — the 6 ablation cells (+ extensions)
# ---------------------------------------------------------------------------


WEAK_NOOP = AutoCubeOptions(
    inference_model="azure/gpt-5-mini",
)
"""Floor baseline — base agent alone, no hints, no exploration.
The 'what does gpt-5-mini do by itself?' measurement."""


STRONG_NOOP = AutoCubeOptions(
    inference_model="azure/gpt-5",
)
"""Naive-scaling ceiling — full GPT-5 with no hints, no exploration.
'Just throw compute at it.' The baseline that tests whether expensive
distillation (via the meta-exploration training loop) beats naive
expensive inference."""


META_HARNESS = AutoCubeOptions(
    inference_model="azure/gpt-5-mini",
    enable_hint_authoring=True,
    enable_text_promotion=True,
    enable_pareto_verify=True,
    propose_multiple_harnesses=True,
)
"""Meta-Harness baseline (arXiv 2603.28052). Hinter discipline plus
their Pareto-verify before commit + multi-harness proposal-and-pick."""


HINTER_ONLY = AutoCubeOptions(
    inference_model="azure/gpt-5-mini",
    enable_hint_authoring=True,
    enable_text_promotion=True,
)
"""Just hinter — author text hints, run them through the re-test
gate. Our hint-authoring contribution in isolation (no
meta-exploration)."""


META_EXPLORATION_ONLY = AutoCubeOptions(
    inference_model="azure/gpt-5-mini",
    enable_config_policy=True,
    allow_model_escalation=True,
    enable_config_promotion=True,
)
"""Just meta-exploration — config policy + config-knob promotion,
no text-hint authoring. Isolates the per-episode config-axis
contribution."""


COMBINED = AutoCubeOptions(
    inference_model="azure/gpt-5-mini",
    enable_config_policy=True,
    allow_model_escalation=True,
    enable_hint_authoring=True,
    enable_text_promotion=True,
    enable_config_promotion=True,
)
"""The full proposed system. Both axes fire; both Phase 2 gates run;
shared ledger + L1 routing. Maps to DESIGN.md §2 ``combined`` row."""


# Registry of named recipes — keys are stable identifiers for sweep
# scripts + analysis tooling.
RUN_RECIPES: dict[str, AutoCubeOptions] = {
    "weak_noop": WEAK_NOOP,
    "strong_noop": STRONG_NOOP,
    "meta_harness": META_HARNESS,
    "hinter_only": HINTER_ONLY,
    "meta_exploration_only": META_EXPLORATION_ONLY,
    "combined": COMBINED,
}


# ---------------------------------------------------------------------------
# Validation + helpers
# ---------------------------------------------------------------------------


class IncoherentOptionsError(ValueError):
    """Raised when an ``AutoCubeOptions`` instance has incompatible flag
    combinations (e.g. ``enable_config_promotion=True`` without
    ``enable_config_policy=True``)."""


def validate_options(opts: AutoCubeOptions) -> None:
    """Run coherence checks; raise ``IncoherentOptionsError`` if any flag
    combination is inconsistent.

    Rules:
      - Phase 2 promotion gates require the matching Phase 1 producer:
          * ``enable_config_promotion`` ⇒ ``enable_config_policy``
          * ``enable_text_promotion`` ⇒ ``enable_hint_authoring``
      - ``allow_model_escalation`` requires ``enable_config_policy``
        (escalation lives in the planner's menu; useless without it).
      - ``enable_pareto_verify`` requires ``enable_hint_authoring``
        (Pareto-verify is over text-hint mutations).
      - ``propose_multiple_harnesses`` requires ``enable_hint_authoring``.
      - ``n_proposed_harnesses >= 2`` when ``propose_multiple_harnesses``.
    """
    if opts.enable_config_promotion and not opts.enable_config_policy:
        raise IncoherentOptionsError(
            "enable_config_promotion=True requires enable_config_policy=True; "
            "Phase 2 needs Phase 1 to produce candidates."
        )
    if opts.enable_text_promotion and not opts.enable_hint_authoring:
        raise IncoherentOptionsError(
            "enable_text_promotion=True requires enable_hint_authoring=True; "
            "Phase 2 needs Phase 1 to produce candidates."
        )
    if opts.allow_model_escalation and not opts.enable_config_policy:
        raise IncoherentOptionsError(
            "allow_model_escalation=True only makes sense with "
            "enable_config_policy=True (the planner is where the menu lives)."
        )
    if opts.enable_pareto_verify and not opts.enable_hint_authoring:
        raise IncoherentOptionsError(
            "enable_pareto_verify=True requires enable_hint_authoring=True; "
            "Pareto-verify guards text-hint mutations."
        )
    if opts.propose_multiple_harnesses and not opts.enable_hint_authoring:
        raise IncoherentOptionsError(
            "propose_multiple_harnesses=True requires enable_hint_authoring=True; "
            "you need the producer to propose multiple harnesses."
        )
    if opts.propose_multiple_harnesses and opts.n_proposed_harnesses < 2:
        raise IncoherentOptionsError(
            f"propose_multiple_harnesses=True requires n_proposed_harnesses>=2; "
            f"got {opts.n_proposed_harnesses}."
        )


def filter_menu_by_options(
    opts: AutoCubeOptions,
    menu: dict[str, EpisodeConfig] | None = None,
) -> dict[str, EpisodeConfig]:
    """Restrict the planner's action menu based on options.

    When ``allow_model_escalation=False``, the ``ESCALATE_MODEL`` entry
    is dropped from the menu so the planner can't accidentally pick it.

    Used by the orchestrator (Pivot 7) when constructing the
    ``PlannerState`` — pass the filtered menu so the planner only
    considers configs the options allow.
    """
    base_menu = menu if menu is not None else EPISODE_CONFIG_MENU
    if opts.allow_model_escalation:
        return dict(base_menu)
    return {name: cfg for name, cfg in base_menu.items() if name != "ESCALATE_MODEL"}


def assert_retest_uses_inference_model(
    opts: AutoCubeOptions,
    episode_config: EpisodeConfig,
) -> None:
    """The honest training/inference split (DESIGN.md §1) is enforced
    here: any episode running under the Phase 2 re-test gate
    (``apply_promotion=True``) MUST use ``opts.inference_model``.

    The orchestrator (Pivot 7) is expected to call this immediately
    before launching a re-test episode. Failure means the promotion
    verdict would have come from a teacher-tier trajectory — invalid.

    Raises ``IncoherentOptionsError`` (the same exception used by
    ``validate_options``) so callers can catch all coherence violations
    with one except clause.
    """
    if not episode_config.apply_promotion:
        return
    if episode_config.model != opts.inference_model:
        raise IncoherentOptionsError(
            f"re-test gate violation: apply_promotion=True with "
            f"model={episode_config.model!r}, but options say "
            f"inference_model={opts.inference_model!r}. Promotion "
            f"verdicts must be generated under the inference model "
            f"only (see DESIGN.md §1)."
        )


__all__ = [
    "AutoCubeOptions",
    "COMBINED",
    "HINTER_ONLY",
    "IncoherentOptionsError",
    "META_EXPLORATION_ONLY",
    "META_HARNESS",
    "RUN_RECIPES",
    "STRONG_NOOP",
    "WEAK_NOOP",
    "assert_retest_uses_inference_model",
    "filter_menu_by_options",
    "validate_options",
]
