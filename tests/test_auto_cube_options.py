"""Tests for the AutoCubeOptions ablation framework (Pivot 5)."""

from __future__ import annotations

import pytest

from cube_harness.auto_cube.options import (
    COMBINED,
    HINTER_ONLY,
    META_EXPLORATION_ONLY,
    META_HARNESS,
    RUN_RECIPES,
    STRONG_NOOP,
    WEAK_NOOP,
    AutoCubeOptions,
    IncoherentOptionsError,
    assert_retest_uses_inference_model,
    filter_menu_by_options,
    validate_options,
)
from cube_harness.meta_exploration.planner import EPISODE_CONFIG_MENU, EpisodeConfig

# ---------------------------------------------------------------------------
# Named recipes — pin the headline shapes
# ---------------------------------------------------------------------------


def test_weak_noop_is_truly_noop() -> None:
    """The floor baseline must have every Phase 1/2 gate OFF."""
    assert WEAK_NOOP.enable_config_policy is False
    assert WEAK_NOOP.enable_hint_authoring is False
    assert WEAK_NOOP.enable_text_promotion is False
    assert WEAK_NOOP.enable_config_promotion is False
    assert WEAK_NOOP.inference_model == "azure/gpt-5-mini"


def test_strong_noop_only_differs_from_weak_by_model() -> None:
    """Naive-scaling baseline: same options as weak, just stronger model."""
    assert STRONG_NOOP.inference_model == "azure/gpt-5"
    assert STRONG_NOOP.enable_config_policy is False
    assert STRONG_NOOP.enable_hint_authoring is False


def test_meta_harness_enables_pareto_verify_and_multi_proposal() -> None:
    """The two Meta-Harness-defining knobs are on."""
    assert META_HARNESS.enable_pareto_verify is True
    assert META_HARNESS.propose_multiple_harnesses is True
    # ...over the standard hinter base
    assert META_HARNESS.enable_hint_authoring is True
    assert META_HARNESS.enable_text_promotion is True
    # ...but no config-axis work
    assert META_HARNESS.enable_config_policy is False
    assert META_HARNESS.enable_config_promotion is False


def test_hinter_only_omits_meta_harness_specific_knobs() -> None:
    """HINTER_ONLY is the 'just our hinter discipline' cell."""
    assert HINTER_ONLY.enable_hint_authoring is True
    assert HINTER_ONLY.enable_text_promotion is True
    assert HINTER_ONLY.enable_pareto_verify is False
    assert HINTER_ONLY.propose_multiple_harnesses is False
    assert HINTER_ONLY.enable_config_policy is False


def test_meta_exploration_only_is_config_axis_only() -> None:
    """No text-hint authoring; only the config-axis machinery."""
    assert META_EXPLORATION_ONLY.enable_config_policy is True
    assert META_EXPLORATION_ONLY.enable_config_promotion is True
    assert META_EXPLORATION_ONLY.allow_model_escalation is True
    assert META_EXPLORATION_ONLY.enable_hint_authoring is False
    assert META_EXPLORATION_ONLY.enable_text_promotion is False


def test_combined_fires_both_axes() -> None:
    assert COMBINED.enable_config_policy is True
    assert COMBINED.enable_config_promotion is True
    assert COMBINED.enable_hint_authoring is True
    assert COMBINED.enable_text_promotion is True
    assert COMBINED.allow_model_escalation is True


def test_run_recipes_registry_has_all_named_recipes() -> None:
    """Registry must contain the 6 ablation cells from DESIGN.md §2."""
    expected = {
        "weak_noop",
        "strong_noop",
        "meta_harness",
        "hinter_only",
        "meta_exploration_only",
        "combined",
    }
    assert set(RUN_RECIPES.keys()) == expected


def test_run_recipes_values_match_module_constants() -> None:
    """The registry must point at the actual constants — not copies."""
    assert RUN_RECIPES["weak_noop"] is WEAK_NOOP
    assert RUN_RECIPES["strong_noop"] is STRONG_NOOP
    assert RUN_RECIPES["meta_harness"] is META_HARNESS
    assert RUN_RECIPES["hinter_only"] is HINTER_ONLY
    assert RUN_RECIPES["meta_exploration_only"] is META_EXPLORATION_ONLY
    assert RUN_RECIPES["combined"] is COMBINED


def test_all_named_recipes_are_valid() -> None:
    """Every shipped recipe must pass validate_options."""
    for name, opts in RUN_RECIPES.items():
        try:
            validate_options(opts)
        except IncoherentOptionsError as e:
            pytest.fail(f"named recipe {name!r} is incoherent: {e}")


# ---------------------------------------------------------------------------
# validate_options — coherence rules
# ---------------------------------------------------------------------------


def test_config_promotion_requires_config_policy() -> None:
    bad = AutoCubeOptions(enable_config_promotion=True)  # no config_policy
    with pytest.raises(IncoherentOptionsError, match="enable_config_policy"):
        validate_options(bad)


def test_text_promotion_requires_hint_authoring() -> None:
    bad = AutoCubeOptions(enable_text_promotion=True)
    with pytest.raises(IncoherentOptionsError, match="enable_hint_authoring"):
        validate_options(bad)


def test_model_escalation_requires_config_policy() -> None:
    bad = AutoCubeOptions(allow_model_escalation=True)
    with pytest.raises(IncoherentOptionsError, match="enable_config_policy"):
        validate_options(bad)


def test_pareto_verify_requires_hint_authoring() -> None:
    bad = AutoCubeOptions(enable_pareto_verify=True)
    with pytest.raises(IncoherentOptionsError, match="enable_hint_authoring"):
        validate_options(bad)


def test_multi_harness_requires_hint_authoring() -> None:
    bad = AutoCubeOptions(propose_multiple_harnesses=True)
    with pytest.raises(IncoherentOptionsError, match="enable_hint_authoring"):
        validate_options(bad)


def test_multi_harness_requires_n_at_least_2() -> None:
    bad = AutoCubeOptions(
        enable_hint_authoring=True,
        propose_multiple_harnesses=True,
        n_proposed_harnesses=1,
    )
    with pytest.raises(IncoherentOptionsError, match="n_proposed_harnesses>=2"):
        validate_options(bad)


def test_default_options_is_coherent() -> None:
    """An empty (all-False) AutoCubeOptions must be valid — that's the
    pure-defaults base everyone overrides from."""
    validate_options(AutoCubeOptions())  # should not raise


# ---------------------------------------------------------------------------
# filter_menu_by_options — escalation gate
# ---------------------------------------------------------------------------


def test_filter_menu_drops_escalate_model_when_disallowed() -> None:
    opts = AutoCubeOptions(allow_model_escalation=False)
    filtered = filter_menu_by_options(opts)
    assert "ESCALATE_MODEL" not in filtered
    # Other entries should still be there.
    assert "BASELINE" in filtered
    assert "WIDEN_SEARCH" in filtered


def test_filter_menu_keeps_escalate_model_when_allowed() -> None:
    opts = AutoCubeOptions(allow_model_escalation=True, enable_config_policy=True)
    filtered = filter_menu_by_options(opts)
    assert "ESCALATE_MODEL" in filtered


def test_filter_menu_size_matches_expectation() -> None:
    opts_disallow = AutoCubeOptions()  # default allow_model_escalation=False
    opts_allow = AutoCubeOptions(allow_model_escalation=True, enable_config_policy=True)
    assert len(filter_menu_by_options(opts_disallow)) == len(EPISODE_CONFIG_MENU) - 1
    assert len(filter_menu_by_options(opts_allow)) == len(EPISODE_CONFIG_MENU)


def test_filter_menu_with_custom_base() -> None:
    """Caller can pass a custom menu (e.g. for sub-ablations)."""
    custom = {"BASELINE": EpisodeConfig(), "ESCALATE_MODEL": EpisodeConfig(model="azure/gpt-5")}
    opts = AutoCubeOptions()  # no escalation
    out = filter_menu_by_options(opts, custom)
    assert set(out.keys()) == {"BASELINE"}


# ---------------------------------------------------------------------------
# assert_retest_uses_inference_model — the honest-promotion-gate guard
# ---------------------------------------------------------------------------


def test_assert_retest_passes_when_not_a_promotion() -> None:
    """Non-promotion episodes can use any model — assertion is no-op."""
    opts = AutoCubeOptions(inference_model="azure/gpt-5-mini")
    cfg = EpisodeConfig(model="azure/gpt-5", apply_promotion=False)
    assert_retest_uses_inference_model(opts, cfg)  # no raise


def test_assert_retest_passes_when_model_matches() -> None:
    opts = AutoCubeOptions(inference_model="azure/gpt-5-mini")
    cfg = EpisodeConfig(model="azure/gpt-5-mini", apply_promotion=True)
    assert_retest_uses_inference_model(opts, cfg)  # no raise


def test_assert_retest_raises_on_promotion_with_wrong_model() -> None:
    """The headline assertion: re-test under teacher model is invalid."""
    opts = AutoCubeOptions(inference_model="azure/gpt-5-mini")
    cfg = EpisodeConfig(model="azure/gpt-5", apply_promotion=True)
    with pytest.raises(IncoherentOptionsError, match="re-test gate violation"):
        assert_retest_uses_inference_model(opts, cfg)


def test_assert_retest_error_mentions_DESIGN_doc() -> None:
    """The error message should point at the design rationale —
    helps the user understand WHY the assertion fires."""
    opts = AutoCubeOptions(inference_model="azure/gpt-5-mini")
    cfg = EpisodeConfig(model="azure/gpt-5", apply_promotion=True)
    with pytest.raises(IncoherentOptionsError, match="DESIGN.md"):
        assert_retest_uses_inference_model(opts, cfg)


# ---------------------------------------------------------------------------
# Frozen / immutable invariants
# ---------------------------------------------------------------------------


def test_autocubeoptions_is_frozen() -> None:
    """Options instances are immutable so they can't be mutated mid-sweep."""
    opts = AutoCubeOptions()
    with pytest.raises((AttributeError, Exception)):
        opts.enable_config_policy = True  # type: ignore[misc]
