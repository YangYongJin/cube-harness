# Auto-CUBE — meta-exploration use case

You are running the **meta-exploration** use case of Auto-CUBE.
Auto-CUBE is the iterate-and-fix outer loop; it owns the methodology,
dispatches the **Investigator** sub-agent per trajectory, and orchestrates
rounds. This use case differs from the others in *what the meta-agent
controls*: not just hint authoring (like `hinter`) or bug attribution
(like `debug`), but the **per-episode exploration configuration itself**.

## Posture

You are a configuration policy over the Auto-CUBE disposition state. For
each episode the orchestrator launches, you pick the agent configuration
(model choice, k-candidate count, perturbation policy, refiner toggle,
Investigator recipe) based on what the per-task ledger tells you about
recent attempts. Cheaper config on tasks where signal is clear; richer
config on tasks where the agent keeps near-missing. The discipline is
the same as `hinter` (Phase 1 / Phase 2 / promote-with-re-test gate),
but the *axis of intervention* is the agent config, not just hint text.

## Goal

Raise benchmark performance by **adaptively allocating per-episode
configuration** across tasks based on disposition state, using the
auto-cube ledger (`~/auto_cube/hints.json`) as both input (prior
sessions' per-task evidence) and output (this session's decisions).
Each session leaves behind both the *promoted hints* (when steers
generalize, same as `hinter`) and the *configuration policy decisions*
recorded in the journal so future sessions can replay or build on them.

## When to use this use case

- You want to **outperform the fixed-config baseline** (uniform K=1,
  fixed model, no perturbations across all tasks).
- You have per-task variance — some tasks always win, some always fail,
  some near-miss. Uniform config wastes compute on the easy + impossible
  tasks and underspends on the near-misses.
- You want a reproducible-from-bash version of the auto-cube hinter
  methodology that also adapts compute allocation.
- **Not** for a single-fix workflow — use `debug`. **Not** for
  pure hint quality (no compute allocation tension) — use `hinter`.

## The exploration option space (the policy's actions)

For each episode the policy picks one `EpisodeConfig` from a structured
menu. Action knobs (start narrow, expand by ablation):

| Knob | Values | What it changes |
|---|---|---|
| `model` | `azure/gpt-5-mini` (default) / `azure/gpt-5` / `azure/o1` | Executor capability + cost |
| `k_candidates` | 1 (cheap) / 3 (deliberate) / 5 (broad) | Action-selection breadth |
| `k_plans` | 1 / 3 | Plan emission count |
| `perturbations` | [] / [`topk_branch`] / [`plan_swap`] | Exploration via branching |
| `bash_default_timeout` | 120 / 300 / 600 (TB only) | Build-task tolerance |
| `investigator_recipe` | `hinter` / `general_blame` / `agent_scaffolding` / `profiling` | L1 dispatch on this episode's blame |
| `apply_promotion` | False / True | Whether this episode is a re-test of a promoted hint |
| `enable_refiner` | False / True | Mid-rollout LLM Refiner |

Action-space curation rule: bound the menu to ~10-20 reasonable configs.
Don't let the policy pick arbitrary combinations — that's combinatorial
explosion + credit-assignment nightmare.

## The two phases (inherited from hinter, generalized)

**Phase 1 — explore & map.** For each failing task, pick the simplest
config likely to flip the episode. Re-run; observe. The win is not the
pass per se — it is **mapping what configuration each task needs**.
Anything that's not the default config is a "steer" in this vocabulary.

**Phase 2 — reflect, promote, compress.** When the same non-default
config (or the same hint produced under it) recurs across several
tasks, promote it up the regularization ladder via the standard auto-cube
discipline gate (`promotion.py:decide_verdict`). A promotion failing
the re-test stays as a per-task cheat in the ledger; a promotion passing
ships either as a `GennyConfig` default change (for config knobs) or as
a benchmark/tool PR (for hint text, via Mode C).

## Disposition vocabulary (read-write on ~/auto_cube/hints.json)

Same as `hinter`: `open / steered / promoted / cheat_only / not_a_hint
/ unsteerable`. The L3 disposition is the **state vector the policy
reads** to make next-iter decisions. The policy's job is to drive every
task toward a terminal disposition (steered / promoted / cheat_only /
not_a_hint / unsteerable) by allocating compute intelligently across
iters.

## Investigator dispatch

Default: per-task L1 recipe routing via
`meta_exploration/recipe_router.py:route_iter_tasks`. The routing
heuristic is cheap (uses `per_task_rewards` + optional `max_repeat` /
`tokens_per_step`); for primary-tier sweeps, the policy can opt into
an LLM tie-break by inspecting the recipe set and prompting a small
classifier — but this is an option, not a default.

## Sampling & rounds

- **Zoom out:** run a broad, cheap batch with the default config on
  the target benchmark; harvest Investigator findings and seed the
  ledger with disposition=`open` for every task.
- **Phase-1 rounds:** policy picks per-task config based on the
  ledger + recent reward trajectory. Confirm steers; record outcomes.
- **Phase-2 rounds:** group steers by their non-default knob; for each
  group, attempt promotion via `apply_promotion_to_config` + re-test
  gate.

## Outputs

- `REPORT.md` — scope, the steer→promote arc per knob, ladder placements,
  before/after scores, shipped vs open PRs, cost.
- `~/auto_cube/hints.json` updates — disposition + promoted_to per task.
- Per-round `plan.json` — the policy's per-episode decisions + rationale,
  alongside the existing `notes.md` + `meta_analysis.{json,md}`.

## Differentiator vs `hinter`

| | `hinter` | `meta_exploration` |
|---|---|---|
| Intervention axis | hint text only | hint text + agent config |
| Compute allocation | uniform per task | adaptive per task |
| Investigator dispatch | always hinter recipe | L1 routing among 4 recipes |
| Policy state | per-task hint history | per-task hint history + reward trajectory + recent configs |
| Outer-loop driver | interactive (Mode C) primarily | interactive **or** SDK (both modes) |
| Mode C role | ship promoted hints upstream | same — plus optionally ship promoted config defaults |
| Comparable baseline | the uniform-config noop run | `hinter` use case + fixed config (apples-to-apples ablation) |

## Cross-session state

Reads + writes `~/auto_cube/hints.json` per
`meta_exploration/ledger.py`. Same shape as `hinter` writes; the
`notes` field carries per-policy-decision metadata
(`{episode_config, expected_value, alternatives_considered}`) so
post-hoc analysis can audit the policy's choices.

## Reference

- Module: `src/cube_harness/meta_exploration/`
- Ledger format: `meta_exploration/ledger.py` docstring
- Recipe router: `meta_exploration/recipe_router.py`
- Promotion gate: `meta_exploration/promotion.py`
- Outer-loop SDK driver: TBD (Pivot 7)
- Companion `hinter` use case: `auto_cube/use_cases/hinter/SKILL.md`
