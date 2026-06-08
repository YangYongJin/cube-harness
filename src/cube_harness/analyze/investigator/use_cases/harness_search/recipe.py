"""`harness_search` — investigate failures for harness evolution.

This recipe is for meta-exploration runs that evolve the agent harness around a
fixed model. It preserves useful task-specific facts as train-only hints,
surfaces low-reg general hints, and records scaffold changes the harness
proposer can turn into `GennyConfig` deltas.
"""

from __future__ import annotations

from typing import Literal

from cube.core import TypedBaseModel
from pydantic import Field, field_validator

from cube_harness.analyze.investigator.recipe import BaseFindings, InvestigatorRecipe
from cube_harness.analyze.investigator.schema_prompt import model_to_json_example


HintScope = Literal["task_specific", "task_family", "general"]
HARNESS_FIELD_VALUES = (
    "benchmark_hint_prompt",
    "hint",
    "system_prompt",
    "react_prompt",
    "act_prompt",
    "step_prompt",
    "goal_template",
    "obs_format",
    "flat_history",
    "enable_summarize",
    "summarize_prompt",
    "compact_prompt",
    "max_obs_chars",
    "description_overrides",
)
HarnessField = Literal[
    "benchmark_hint_prompt",
    "hint",
    "system_prompt",
    "react_prompt",
    "act_prompt",
    "step_prompt",
    "goal_template",
    "obs_format",
    "flat_history",
    "enable_summarize",
    "summarize_prompt",
    "compact_prompt",
    "max_obs_chars",
    "description_overrides",
]
HARNESS_FIELD_SET = set(HARNESS_FIELD_VALUES)
RecommendationScope = Literal["train_only", "generalizable", "uncertain"]
GeneralHintField = Literal["benchmark_hint_prompt", "hint"]
RECOMMENDATION_SCOPE_ALIASES = {
    "task_specific": "train_only",
    "task-specific": "train_only",
    "task specific": "train_only",
    "task": "train_only",
    "task_family": "uncertain",
    "task-family": "uncertain",
    "task family": "uncertain",
    "family": "uncertain",
    "general": "generalizable",
    "generalizable": "generalizable",
    "generalized": "generalizable",
    "uncertain": "uncertain",
    "unsure": "uncertain",
}


class HarnessTaskHint(TypedBaseModel):
    """A train-only hint candidate extracted from one failed episode."""

    task_id: str = Field(description="The task_id this hint targets.")
    text: str = Field(description="Short instruction ready for `GennyConfig.task_hints[task_id]`.")
    scope: HintScope = Field(
        description="Whether the hint is single-task, family-level, or plausibly general."
    )
    rationale: str = Field(description="Why this hint would have helped, citing transcript evidence.")
    confidence: int = Field(ge=0, le=5, description="0=guess, 5=very likely to fix a re-run.")


class HarnessGeneralHint(TypedBaseModel):
    """A low-reg, task-agnostic hint candidate for the harness proposer."""

    failure_mode_key: str = Field(
        description=(
            "Stable snake_case key for aggregating this signal across episodes, e.g. "
            "`premature_final`, `no_effect_action_loop`, `coordinate_grounding`, "
            "`insufficient_exploration_diversity`."
        )
    )
    target_field: GeneralHintField = Field(
        description="Where this general hint should usually land if promoted by the proposer."
    )
    text: str = Field(
        description=(
            "Short task-agnostic guidance ready for `GennyConfig.hint` or "
            "`GennyConfig.benchmark_hint_prompt`."
        )
    )
    affected_task_ids: list[str] = Field(
        default_factory=list,
        description="Train task IDs this general hint is expected to help.",
    )
    validation_task_ids: list[str] = Field(
        default_factory=list,
        description="Specific tasks that should be re-run to validate this hint and catch regressions.",
    )
    transfer_rationale: str = Field(description="Why this should transfer beyond the exact task.")
    risk: str = Field(default="", description="Likely regression, cost, or overfitting risk.")
    confidence: int = Field(ge=0, le=5, description="0=guess, 5=strongly supported by this episode.")


class HarnessRecommendation(TypedBaseModel):
    """A scaffold-level recommendation for the harness proposer."""

    failure_mode_key: str = Field(
        default="",
        description=(
            "Stable snake_case key for grouping related recommendations across episodes. "
            "Use the same key as a related general_hint_candidate when applicable."
        ),
    )
    target_fields: list[HarnessField] = Field(
        default_factory=list,
        description=(
            "GennyConfig harness fields likely involved. Leave empty if the fix is real "
            "but not representable as a config delta."
        ),
    )
    recommendation: str = Field(
        description="Concrete scaffold change the proposer should consider."
    )
    scope: RecommendationScope = Field(
        description=(
            "train_only = keep as task_hints; generalizable = candidate harness_overrides; "
            "uncertain = useful signal but needs proposer judgment."
        )
    )
    affected_task_ids: list[str] = Field(
        default_factory=list,
        description="Train task IDs this recommendation is expected to affect.",
    )
    validation_task_ids: list[str] = Field(
        default_factory=list,
        description="Tasks that should be included in proposal validation for this recommendation.",
    )
    expected_behavior_change: str = Field(
        default="",
        description="Observable behavior that should change if the harness recommendation works.",
    )
    rationale: str = Field(description="Evidence-grounded reason this is the right layer.")
    risk: str = Field(
        default="",
        description="Likely regression or overfitting risk if converted into a general harness change.",
    )
    confidence: int = Field(ge=0, le=5, description="0=guess, 5=strongly supported by the episode.")

    @field_validator("scope", mode="before")
    @classmethod
    def _normalize_scope_aliases(cls, value):
        if not isinstance(value, str):
            return value
        key = value.strip().lower()
        return RECOMMENDATION_SCOPE_ALIASES.get(key, value)

    @field_validator("target_fields", mode="before")
    @classmethod
    def _drop_non_harness_target_fields(cls, value):
        """Keep bad investigator field guesses from invalidating the episode."""
        if value is None:
            return []
        values = [value] if isinstance(value, str) else list(value)
        return [field for field in values if field in HARNESS_FIELD_SET]


class HarnessSearchOutput(BaseFindings):
    """Investigator output for harness-search subagents."""

    task_hints: list[HarnessTaskHint] = Field(
        default_factory=list,
        description=(
            "Train-only hints worth preserving exactly. Empty when no short task hint "
            "would help or the fix belongs in a general scaffold change."
        ),
    )
    general_hint_candidates: list[HarnessGeneralHint] = Field(
        default_factory=list,
        description=(
            "Low-reg general hints that are task-agnostic enough to test as `hint` or "
            "`benchmark_hint_prompt` before heavier scaffold rewrites."
        ),
    )
    harness_recommendations: list[HarnessRecommendation] = Field(
        default_factory=list,
        description="Potential scaffold changes for the harness proposer to synthesize.",
    )


HARNESS_SEARCH_SYSTEM_PROMPT = f"""You are a harness-search investigator for failed agent episodes.

Your job is to read one failed trajectory and produce evidence-grounded signals
for the next harness-search iteration. The fixed base model is NOT changing.
The harness around it may change what is stored, retrieved, shown, summarized,
or emphasized while the agent works.

This is not a request to design a new meta-harness or new agent architecture.
Prefer signals that help build a clean setting where hinters/general hints are
useful but bottlenecked by exploration scale and diversity.

Think in three channels:

1. `task_hints`: exact train-task facts or one-off quirks that would help this
   task but should NOT be promoted to held-out/OOD.
2. `general_hint_candidates`: short task-agnostic guidance that may fit the
   low-reg `hint` / `benchmark_hint_prompt` channel. Use this before proposing
   heavier scaffold rewrites when a small general hint is enough.
3. `harness_recommendations`: general scaffold changes the proposer might turn
   into `GennyConfig` deltas such as prompt edits, observation formatting,
   summarization policy, max observation size, or tool description overrides.
   `target_fields` must name only scaffold fields. Never put `task_hints` in
   `target_fields`; if the useful fix is a train-only hint, put it in
   `task_hints` instead and set any related recommendation to `scope="train_only"`
   with an empty `target_fields` list.

Do not force generality. Do not force everything into one failure mode. If a
task-specific hint, a low-reg general hint, and a scaffold recommendation are
all useful, emit all three. But be disciplined: a broad harness recommendation
needs evidence that the failure is a scaffold-level pattern, not merely a
per-task fact. Use `failure_mode_key` consistently so the proposer can aggregate
signals across episodes.

Useful scaffold failure patterns include:
- The agent reaches the right state but submits or terminates too early.
- The agent repeats, thrashes, loses context, or forgets prior observations.
- The observation/tool docs hide a crucial state or success condition.
- The prompt encourages a bad protocol, missing verification, or wrong channel.
- The agent lacks an actionable affordance description even though the tool can do it.
- The failure suggests more rollout diversity/exploration would reveal a useful
  hint, but the one observed trajectory is too narrow to justify a broad edit.

Rules:
- Read the transcript end-to-end first; cite step numbers in `evidence`.
- Keep recommendations concrete enough for a proposer to implement.
- If the right fix is a tool bug, eval bug, environment failure, or model ceiling,
  say so in the base findings and keep scaffold recommendations empty unless a
  harness workaround is genuinely useful and low-risk.
- BUT if `primary_blame` is `agent_scaffolding` or `insufficient_observation`,
  the failure lives in the harness layer itself — you MUST emit at least one
  `harness_recommendation` (or a `general_hint_candidate` if a short general hint
  suffices). Do NOT return all three lists empty for a scaffold/observation
  failure: that silently discards the exact signal this recipe exists to capture.
  Prefer naming a concrete `target_field` (e.g. `step_prompt` for a
  plan→execute/verify-before-stop gap, `react_prompt` for a no-pivot loop,
  `obs_format`/`max_obs_chars` for hidden state, `description_overrides` for a
  missing tool affordance).
- Still emit task-specific `task_hints` whenever an exact task fact would help,
  even alongside a scaffold recommendation. Both channels are wanted — a scaffold
  fix does not replace a needed per-task hint.
- `scope="generalizable"` only when the change plausibly helps multiple tasks.
- `scope="train_only"` when the fact should remain a task hint.
- `scope="uncertain"` for task-family-ish scaffold ideas that are useful but
  not clearly train-only or OOD-general.
- Add `validation_task_ids` for any hint/recommendation with nontrivial risk.
- Valid `target_fields` values are: {", ".join(HARNESS_FIELD_VALUES)}.

Reply with a single JSON object inside ```json ... ``` fences, matching the
schema in the user prompt."""


_HARNESS_SEARCH_OUTPUT_JSON = (
    model_to_json_example(HarnessSearchOutput).replace("{", "{{").replace("}", "}}")
)


HARNESS_SEARCH_USER_PROMPT_TEMPLATE = f"""Investigate this failed episode for harness search.

# Episode

trajectory_id: {{trajectory_id}}
task_id: {{task_id}}
reward: {{reward}}
total_steps: {{total_steps}}
agent: {{agent_name}}
benchmark: {{benchmark_name}}

# Files you can read

Transcript:
  {{transcript_dir}}/transcript.txt
  {{transcript_dir}}/steps/

Episode metadata (reward_info, action_schemas, summary_stats):
  {{episode_metadata_path}}

Episode config (agent prompts, model, budget, current task_hints):
  {{episode_config_path}}

Source code (Glob / Grep — useful for understanding tool exposure and scaffold fields):
{{source_paths_block}}

Task description:
  {{task_description}}

# Output schema

Produce a single JSON object matching `HarnessSearchOutput`. Wrap in a ```json
fence. Field order is deliberate: analyze/evidence first, then commit to
task_hints, general_hint_candidates, and harness_recommendations. Each leaf is annotated
`<type — description>`:

```json
{_HARNESS_SEARCH_OUTPUT_JSON}
```

Reminders:
- `task_hints` are training-only. Use them for exact task facts or one-off quirks.
- `general_hint_candidates` are low-reg, task-agnostic hints. Prefer these over
  heavier prompt/tool rewrites when a short general hint is enough.
- `harness_recommendations` are scaffold signals. They are not final deltas; the
  proposer will synthesize and validate them.
- If nothing useful can be changed in hints or harness config, all three lists may be empty."""


RECIPE = InvestigatorRecipe(
    name="harness_search",
    system_prompt=HARNESS_SEARCH_SYSTEM_PROMPT,
    user_prompt_template=HARNESS_SEARCH_USER_PROMPT_TEMPLATE,
    output_model=HarnessSearchOutput,
    model="claude-sonnet-4-6",
)


__all__ = [
    "HARNESS_SEARCH_SYSTEM_PROMPT",
    "HARNESS_SEARCH_USER_PROMPT_TEMPLATE",
    "HarnessGeneralHint",
    "HarnessRecommendation",
    "HarnessSearchOutput",
    "HarnessTaskHint",
    "RECIPE",
]
