# investigator-harness-search

**When to pick this recipe**

Pick this when the outer loop is evolving the agent harness rather than only
harvesting task hints. The investigator should identify both:

- train-only task facts worth preserving as `task_hints[task_id]`; and
- low-reg task-agnostic guidance worth testing as `hint` /
  `benchmark_hint_prompt`; and
- general scaffold failures the proposer may convert into `GennyConfig` deltas.

Common triggers:

- The agent reaches the right state but submits or terminates too early.
- The agent repeats, thrashes, loses context, or forgets prior observations.
- Observation formatting, summarization, or truncation hides crucial state.
- Tool documentation or prompt protocol makes the wrong action/channel likely.
- A pattern may affect multiple sibling tasks, but the exact fix still needs
  proposer synthesis and validation.

**Output**

`findings.json` with the standard `BaseFindings` fields PLUS:

- `task_hints[]` — train-only `HarnessTaskHint` entries. Use these for exact
  task facts or one-off quirks.
- `general_hint_candidates[]` — low-reg general hints with a
  `failure_mode_key`, suggested target field, validation tasks, risk, and
  confidence.
- `harness_recommendations[]` — scaffold-level `HarnessRecommendation` entries
  with a `failure_mode_key`, target `GennyConfig` fields, affected task IDs,
  validation tasks, expected behavior change, scope, rationale, risk, and
  confidence.

**When NOT to pick this**

- Pure hint baseline / AutoCube comparison: use `hinter`.
- Pure closed-world blame attribution: use `general_blame`.
- Narrow loop taxonomy only: use `agent_scaffolding`.
- Quantitative budget/token pathology only: use `profiling`.

Harness recommendations are not final edits. The proposer synthesizes them into
candidate `harness_overrides` and validation decides whether to apply them.
