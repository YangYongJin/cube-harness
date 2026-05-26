# Auto-CUBE — meta-exploration use case

You are running the **meta-exploration** use case of Auto-CUBE.
Auto-CUBE is the iterate-and-fix outer loop; it owns the methodology,
dispatches the **Investigator** sub-agent per trajectory, and orchestrates
rounds. This use case differs from the others in *what the meta-agent
controls*: not the hint text (that's `hinter`) or the bug attribution
(that's `debug`), but the **per-episode exploration configuration**
that determines what trajectories the downstream consumers (`hinter`
authoring, `debug` blame attribution, Phase 2 promotion gate) get to
work with.

## Posture

You are an **information-gathering policy** over the Auto-CUBE
disposition state. For each episode the orchestrator launches, you
pick the agent configuration (model, k-candidate count, perturbation
policy, refiner toggle, Investigator recipe) based on **what the
downstream stages need to see to do their job well**. Not "pick the
cheapest config that wins this episode" — pick the config whose
trajectory maximally helps the next stages decide.

## Goal

Raise benchmark performance by allocating per-episode configuration so
that the downstream pipeline (hinter / debug / promotion gate) gets
the richest evidence per dollar. Each session leaves behind both
configuration decisions (per-episode `plan.json` per round) and any
shared-infrastructure side effects (the `~/auto_cube/hints.json`
ledger writes the disposition + promoted_to per task).

## Composable with hinter — Option B architecture

**This use case is strictly the configuration-axis policy.** It does
NOT author hints. Hint authoring lives in the companion `hinter` use
case, which consumes the trajectories meta-exploration generates and
emits text hints via the per-trajectory Investigator + Phase 2
promotion gate.

| Stage | meta-exploration does | hinter does |
|---|---|---|
| Episode launch | Picks per-episode `EpisodeConfig` from menu | (no-op) |
| Episode runs | (no-op) | (no-op) |
| Per-trajectory Investigator | Picks recipe via L1 router | Receives `task_hints[]` when recipe is `hinter` |
| End-of-iter | Aggregates outcomes per task; updates policy state | Aggregates text-hint candidates → mutates `GennyConfig.task_hints` |
| Phase 2 promotion | Detects config-knob promotion candidates → re-test gate | Detects text-hint promotion candidates → re-test gate (same gate code) |
| Ledger | Writes disposition + notes (per-config decision metadata) | Writes disposition + promoted_to (text hint) |

The two compose **via data flow**, not code coupling — neither imports
the other. They share the same `meta_exploration/` module
infrastructure (`ledger.py`, `recipe_router.py`, `promotion.py`).

Run combinations the ablation framework supports:

```python
# Pure config policy, no hint authoring:
META_EXPLORATION_ONLY = AutoCubeOptions(
    enable_config_policy=True,
    enable_config_promotion=True,
)

# Pure hint authoring, default config every episode:
HINTER_ONLY = AutoCubeOptions(
    enable_hint_authoring=True,
    enable_text_promotion=True,
)

# Both — full system, info-gathered configs → richer hints:
COMBINED = AutoCubeOptions(
    enable_config_policy=True,
    enable_config_promotion=True,
    enable_hint_authoring=True,
    enable_text_promotion=True,
)
```

## When to use this use case (vs alone vs combined)

- **Standalone** — when you want to study compute-allocation in
  isolation, or have hint memory frozen and want to see if smarter
  per-task config alone moves the score.
- **Combined with hinter** — the productive setup. Meta-exploration
  gathers info-rich trajectories; hinter mines them for better
  hints; both contribute to the same ledger.
- **NOT for** — pure hint-quality studies (use `hinter` alone),
  bug-finding sweeps (use `debug`), single one-shot tasks (just do it
  by hand).

## The exploration option space (the policy's action set)

For each episode the policy picks one `EpisodeConfig` from a structured
menu. Action knobs (start narrow, expand by ablation as you learn what
matters):

| Knob | Values | Information it surfaces |
|---|---|---|
| `model` | `azure/gpt-5-mini` (default) / stronger tier | Whether failure is capability-bound (stronger model succeeds) vs structural (stronger model also fails) |
| `k_candidates` | 1 (cheap) / 3 (deliberate) / 5 (broad) | What alternative actions the agent considered — hinter sees the near-miss action it ALMOST took |
| `k_plans` | 1 / 3 | What alternative strategies the agent considered before committing |
| `perturbations` | [] / [`topk_branch`] / [`plan_swap`] | Branch counterfactuals — what would have happened if the agent had picked the 2nd-best action at uncertain steps |
| `bash_default_timeout` | 120 / 300 / 600 (TB only) | Whether failure is a timeout (info: bump default) vs a real bug |
| `investigator_recipe` | `hinter` / `general_blame` / `agent_scaffolding` / `profiling` | What blame taxonomy the downstream Investigator applies |
| `enable_refiner` | False / True | Whether a mid-rollout knowledge injection would have unstuck the agent |
| `apply_promotion` | False / True | Triggers a re-test of a promoted hint candidate (Stage E gate) |

Action-space curation rule: bound the menu to ~10-20 reasonable
configs. The policy picks one per episode, not a free-form spec.

## The two phases (mirror hinter; act on different objects)

**Phase 1 — explore & map.** For each failing task, pick the
information-richest cheap config likely to advance the task's
disposition (open → steered, or open → not_a_hint, or open →
unsteerable). Re-run; observe; record outcome to ledger. The win is
not the pass — it is **disambiguating which disposition this task
should land in** for the downstream stages.

**Phase 2 — reflect, promote, compress.** When the same non-default
config knob recurs across N tasks (e.g. "bump bash timeout to 600 works
on 3 build tasks"), promote it via the shared
`meta_exploration/promotion.py:decide_verdict` gate. A passing
promotion sets a new default `GennyConfig` field across the benchmark;
a failing one stays per-task in the ledger.

## Information-gathering reward signal

The policy's objective is **downstream-pipeline-aware**, not raw
episode reward:

```
reward(config_choice) = α * episode_succeeded
                     + β * hint_authored_from_traj_passes_retest    (downstream)
                     + γ * task_disposition_advances_toward_terminal (multi-iter)
                     - δ * cost_usd
```

The β and γ terms are why this is "info-gathering" not "compute
allocation": configs are scored on whether they helped the next stages
make progress, not just on this episode's win/loss.

v1 implementation: heuristic policy with the above signals as
post-hoc rationales (`notes` field in the ledger). v2: learn the
policy from accumulated `(state, config, outcome)` triples — bandits
over the structured action space.

## Disposition vocabulary (shared with hinter)

Same as `hinter`: `open / steered / promoted / cheat_only / not_a_hint
/ unsteerable`. The L3 disposition is the **state vector the policy
reads** to make next-iter decisions. The policy's job is to drive every
task toward a terminal disposition by choosing the right
information-gathering config for that task's current state.

## Investigator dispatch (L1 routing)

Per-task recipe selection via
`meta_exploration/recipe_router.py:route_iter_tasks`. The default
heuristic uses `per_task_rewards` (+ optional `max_repeat` /
`tokens_per_step` when available); primary-tier sweeps can opt into an
LLM tie-break.

The policy can also override the recipe per episode as part of its
information-gathering decision — e.g. force `agent_scaffolding` recipe
on a task that's been routing to `hinter` but never getting useful
hint candidates.

## Sampling & rounds

- **Zoom out:** broad cheap batch with `EpisodeConfig=DEFAULT` on the
  target benchmark; seed the ledger with `disposition=open` per task;
  let Investigator dispatch via the L1 router collect baseline findings.
- **Phase-1 rounds:** policy picks per-task config based on the ledger
  + recent reward trajectory. Each config choice generates a
  trajectory whose downstream consumption (hint authoring + recipe
  classification) the policy uses to score the choice.
- **Phase-2 rounds:** group config knobs by task-cluster signal; for
  each group, attempt promotion via the shared re-test gate.

## Outputs

- `REPORT.md` — scope, the config-policy decisions per task, ladder
  placements for promoted config knobs, before/after scores, shipped
  PRs (if any config promoted to a `GennyConfig` default upstream), cost.
- `~/auto_cube/hints.json` updates — disposition + notes (per-policy-
  decision metadata: `{episode_config, rationale, downstream_signal}`).
- Per-round `plan.json` — the policy's per-episode decisions +
  alternatives_considered + expected_value, alongside the existing
  `notes.md` + `meta_analysis.{json,md}`.

## Cross-session state

Reads + writes `~/auto_cube/hints.json` per
`meta_exploration/ledger.py`. Future sessions inherit prior config-knob
outcomes from the ledger's `notes` field, so the policy doesn't
re-derive that a particular task needs `k_candidates=3` if a prior
session already confirmed it.

## Differentiator vs `hinter` (one-line)

`hinter` evolves *what the agent reads*; `meta_exploration` evolves
*what configuration generates the trajectories the agent reads from*.
Both use the same Phase 1/2 discipline + ledger; they're orthogonal
intervention axes and the full system uses both.

## Reference

- Module: `src/cube_harness/meta_exploration/`
- Ledger format: `meta_exploration/ledger.py` docstring
- Recipe router: `meta_exploration/recipe_router.py`
- Promotion gate: `meta_exploration/promotion.py`
- EpisodeConfig planner: `meta_exploration/planner.py` (Pivot 4 next)
- AutoCubeOptions ablation menu: `meta_exploration/options.py` (Pivot 5)
- Outer-loop SDK driver: TBD (Pivot 7)
- Companion `hinter` use case: `auto_cube/use_cases/hinter/SKILL.md`
- Meta-Harness baseline (paper arXiv 2603.28052): represented as a
  named subset of `AutoCubeOptions`
