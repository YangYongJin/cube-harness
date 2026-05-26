"""Meta-exploration round <N> experiment config — Auto-CUBE meta-exploration use case.

Copied from this template and edited per round. This file IS the config. NO
``# /// script`` header on purpose: run with the repo venv, not standalone uv —

    /path/to/cube-harness/.venv/bin/python <this_dir>/exp_config.py --limit 3

The meta-exploration use case extends `hinter` along a second axis: not only
*what hint text* the agent gets, but *what agent configuration* runs each
episode. The orchestrator (Claude Code in interactive mode, or the outer-loop
SDK driver in unattended mode) picks per-episode config from a structured
menu based on the per-task disposition state in ``~/auto_cube/hints.json``.

This template shows the knobs available to the policy. A single round
fixes a subset (the "config tier" being tested this round); subsequent
rounds vary the subset based on what won.
"""

from terminalbench2_cube import TERMINALBENCH2_CONFIGS

from cube_harness.agents.genny_configs import GENNY_CONFIGS
from cube_harness.experiment import Experiment
from cube_harness.infra import INFRA_CONFIGS
from cube_harness.llm import LLMConfig
from cube_harness.recipe import run

# --- vary per round -------------------------------------------------------

TASK_IDS = [
    # explicit subset for this round — the tasks the policy is exploring
    "build-cython-ext",
    "code-from-image",
    "distribution-search",
]

# Per-round model tier. The policy can pick differently per task by
# overriding agent.llm_config below; this is the default.
MODEL = "azure/gpt-5-mini"
COST_PER_TASK = 0.50

# --- assemble -------------------------------------------------------------

agent = GENNY_CONFIGS["swe"]
agent.llm_config = LLMConfig(model_name=MODEL, temperature=1.0)
agent.budget.cost_limit = COST_PER_TASK

# (1) Fold in any benchmark-wide clarification overlay. Same as hinter —
# this is where curated benchmark prompt + per-task clarifications land
# once they're promoted via the re-test gate.
benchmark = TERMINALBENCH2_CONFIGS["default"].subset_from_list(TASK_IDS)
agent = agent.with_benchmark_clarifications(benchmark)

# (2) Optionally override action descriptions. The meta-exploration policy
# can propose per-action wording tweaks; once they pass the re-test gate,
# they get promoted into the tool's docstring upstream.
agent.description_overrides = {
    # "bash": "Run a shell command. For builds/installs, pass timeout=600 explicitly.",
}

# (3) The exploration knobs the policy controls per episode. The values
# below are the *baseline* / default-config tier; the policy in unattended
# (SDK) mode may override these per task via a separate episode-config
# decision module (see Pivot 4: EpisodeConfig planner). In interactive
# mode, the human edits this file per round to reflect the policy's
# decisions.
#
# k_candidates ∈ {1, 3, 5}
# k_plans ∈ {1, 3}
# enable_refiner ∈ {False, True}
# perturbations: see src/pgepa_v2/perturbations/  (will move into cube-harness)
#
# Document the config choice in this round's notes.md.

exp = Experiment(
    name="auto-cube-meta-exploration-r<N>",
    agent_config=agent,
    benchmark_config=benchmark,
    infra=INFRA_CONFIGS["local"],
    max_steps=agent.budget.max_actions or 60,
)

if __name__ == "__main__":
    run(exp)
