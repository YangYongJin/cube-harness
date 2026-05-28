# Meta-Exploration — Design

Companion to [`SKILL.md`](SKILL.md). Captures the **research framing**
and **architecture decisions** behind the meta-exploration use case so
future sessions don't re-derive them.

> SKILL.md is *what* the use case does. This doc is *why* it does it
> that way. Current state-of-the-work is in [`STATUS.md`](STATUS.md).

---

## §1. Research framing — the training/inference distinction

**The core insight:** the meta-exploration outer loop IS a training
pipeline. The expensive exploration (stronger models on hard tasks,
wider action sampling, mid-rollout refiners, perturbations) is
**training-time compute**, analogous to:

| Standard ML pattern | Meta-exploration equivalent |
|---|---|
| Training-data generation with GPT-4 (expensive teacher) | Meta-exploration loop with `ESCALATE_MODEL` allowed |
| Synthetic / rejection-sampled training set | Trajectories from the 8-config menu × perturbations × replicas |
| Smaller model deployed at inference | Base agent (e.g. gpt-5-mini) + the hints/configs the training loop produced |
| Distillation: teacher knowledge → student | Strong-model trajectories' info → hints that help the weak agent |

As long as the **inference run uses only the fixed baseline model +
the artifacts the training loop produced**, the comparison is honest.

### What makes this honest (code-enforced controls)

1. **`AutoCubeOptions.inference_model`** is a separate field from the
   training-time model menu. The Phase 2 re-test gate uses **only** the
   inference model. Assertion-enforced (`assert_retest_uses_inference_model`),
   not vibes.
2. **The ledger records `inference_model` per task** alongside the
   training-time configs used. Reviewers can audit any per-task model
   choice post-hoc.

### The hints-are-agent-specific claim

The hints aren't generic LLM tips — they're tailored to *what the base
agent gets wrong on a specific task*:

- Generic: *"For slider tasks, use keyboard navigation."*
- Agent-specific: *"gpt-5-mini tends to use ArrowUp (1px/press) on
  sliders; explicitly use PageUp/PageDown for big gaps."*

GPT-5 alone gets the generic version for free from its better
reasoning. But gpt-5-mini + the SPECIFIC hint can be competitive,
because the hint encodes the model-specific failure pattern that even
GPT-5 wouldn't have surfaced. This is **distilled compute**.

---

## §2. The 6-cell ablation table (future paper-grade work)

These are the cells the design supports. **Cell 1 / Cell 2** in
`STATUS.md` are NOT this table — those measure baseline hinter at 1×
vs 4× episodes/iter, to decide whether the full 6-cell ablation is
worth running.

| | Inference model | Training hints | Training exploration | What it measures |
|---|---|---|---|---|
| `weak_noop` | gpt-5-mini | none | none | floor — base agent alone |
| `strong_noop` | gpt-5 | none | none | naive-scaling ceiling |
| `meta_harness` | gpt-5-mini | Meta-Harness | none | prior-art baseline |
| `hinter_only` | gpt-5-mini | PGEPA hinter | default config | hint-authoring contribution alone |
| `meta_exploration_only` | gpt-5-mini | none | meta-exploration policy | config-axis alone |
| `combined` (proposal) | gpt-5-mini | PGEPA hinter | meta-exploration policy | full system |

The 6 recipes are pre-built in
[`auto_cube/options.py`](../../options.py) (`RUN_RECIPES` dict).

### Story we want the table to tell

- `combined > hinter_only > meta_harness > weak_noop` → each piece adds value
- `combined ≥ strong_noop` → distilled compute matches naive scaling
- `combined/cost >> strong_noop/cost` → headline result

---

## §3. Architecture decisions

### §3.1 Option B — composes with hinter, doesn't subsume it

Meta-exploration is **strictly the per-episode configuration policy
axis**. It does NOT author hints — hint authoring lives in the
companion `hinter` use case. The two compose via **data flow**:

```
meta-exploration's Stage A:    hinter's Stage D:
  picks per-episode config  →  consumes resulting trajectory  →
                                 → authors text hint
                                 → enters re-test gate (Stage E)
```

Neither use case imports the other. Both share infrastructure in
[`meta_exploration/`](../../../meta_exploration/) (ledger, recipe_router,
promotion).

### §3.2 The 4-level Auto-CUBE taxonomy

Inherited from upstream [`hinter/SKILL.md`](../hinter/SKILL.md). Used
by both use cases:

- **L1** Investigator recipe (per failed episode): `hinter` /
  `general_blame` / `agent_scaffolding` / `profiling`
- **L2** `hint_type` (per hint): `clarification` / `task_specific` /
  `general_guidance`
- **L3** `disposition` (per task): `open` / `steered` / `promoted` /
  `cheat_only` / `not_a_hint` / `unsteerable`
- **L4** `promoted_to` (per promoted hint): the regularization-ladder
  rung — `task_hints` / `benchmark_hint_prompt` / `task_clarification`
  / `description_overrides` / `new_action` / `system_prompt`

### §3.3 Bounded action menu (8 named EpisodeConfigs)

The planner picks from a **bounded menu** of 8 named configs rather
than composing arbitrary EpisodeConfig field values. Trade-off:

|  | Menu (chosen) | Free composition |
|---|---|---|
| Action space size | 8 | 8 fields × ~3 values ≈ ~3000 |
| Ablation tractable? | ✅ | ❌ |
| Credit assignment | ✅ named-pick → outcome | ❌ which knob mattered? |
| Cold-start sane? | ✅ heuristic picks known-good | ❌ no prior on combos |

v2 expansion path: once we have data on which knobs win, the policy
can move from menu-picking to bandit over the full `EpisodeConfig`
space.

---

## §4. The planner's decision tree

Pure-function, 8-rule, first-match-wins. Implemented in
[`meta_exploration/planner.py:pick_episode_config`](../../../meta_exploration/planner.py).

| # | Trigger | Pick | Info gathered |
|---|---|---|---|
| 1 | `cheat_only` AND ≥2 attempts | `RETEST_PROMOTION` | does the cheat generalize? |
| 2 | brand-new task | `BASELINE` | baseline trajectory |
| 3 | any past reward in (0, 0.5) | `ESCALATE_MODEL` | capability- vs structural-bound |
| 4 | ≥3 attempts, all 0 reward | `DIAGNOSE_PROFILING` | cheap confirmation before unsteerable |
| 5a | one+ fails + loop note | `DIAGNOSE_SCAFFOLDING` | route to scaffolding recipe |
| 5b | one+ fails + timeout note | `EXTEND_TIMEOUT` | bump bash to 600s |
| 5c | one+ fails + no notes | `WIDEN_SEARCH` | k=3 + topk_branch perturbation |
| 8 | fallthrough | `BASELINE` | safe fallback |

After any pick, a **budget check**: if the picked config's approx cost
exceeds remaining budget, downgrade to `DIAGNOSE_PROFILING` with the
original pick logged in `alternatives_considered`.

### Verb-first naming convention

| Menu name | Underlying EpisodeConfig |
|---|---|
| `BASELINE` | defaults |
| `WIDEN_SEARCH` | `k_candidates=3, perturbations=("topk_branch",)` |
| `ESCALATE_MODEL` | `model="azure/gpt-5"` |
| `ENABLE_REFINER` | `enable_refiner=True` |
| `RETEST_PROMOTION` | `apply_promotion=True` |
| `EXTEND_TIMEOUT` | `bash_default_timeout=600` |
| `DIAGNOSE_SCAFFOLDING` | `investigator_recipe="agent_scaffolding"` |
| `DIAGNOSE_PROFILING` | `investigator_recipe="profiling"` |

---

## §5. Mechanical knobs vs textual exploration notes

Two ways meta-exploration can intervene; we ship both as independently
toggleable on `AutoCubeOptions` so we can ablate which channel matters.

### Channel 1 — mechanical knobs (the original design)

Fields on `EpisodeConfig` mechanically force the agent to behave
differently:

- `model`, `bash_default_timeout`, `investigator_recipe`,
  `apply_promotion` — directly supported by upstream `GennyConfig` /
  `TerminalToolConfig`; **no porting needed**.
- `k_candidates`, `k_plans`, `perturbations`, `enable_refiner` —
  implemented by PGEPA's algorithm modules; **requires porting** into
  `meta_exploration/` or as upstream RFC.

**Pro:** strong guarantee — the agent IS forced to explore differently.
**Con:** algorithm-porting cost; tight coupling to Genny internals.

### Channel 2 — textual exploration notes (the lightweight alternative)

Meta-exploration's policy authors short text strings ("exploration
notes") per task, injected into the agent's prompt as a fourth
category of hint alongside `benchmark_hint_prompt` /
`task_clarification` / `task_hints`.

Examples:
- *"This task has timed out twice — consider running pip install with
  longer timeouts."*
- *"On the prior attempt the agent used ArrowUp on a slider with 50
  steps to go — consider PageUp/PageDown for big gaps."*

Storage: `HintLedgerEntry.notes["exploration_notes"]` per task.

**Pro:** zero algorithm porting; cleanly composes with hinter (both
are text authors); cheap to iterate.
**Con:** weak guarantee — the agent MIGHT ignore the note.

### Both-as-options

New `AutoCubeOptions` fields:
- `enable_mechanical_exploration_knobs: bool = True`
- `enable_exploration_notes: bool = False`

Ablation extension (4-cell sub-table):

| Variant | mechanical | notes | What it isolates |
|---|---|---|---|
| `meta_exploration_only` | True | False | Original framing — mechanical channel only |
| `meta_exploration_notes_only` | False | True | Lightweight text channel only |
| `meta_exploration_both` | True | True | Overdetermined — maximum exploration signal |
| `meta_exploration_off` (= weak_noop) | False | False | Baseline |

If `notes_only` is competitive with `mechanical_only`, the algorithm
ports become low-value work and we ship just the notes channel.

---

## §6. Open architectural calls

These are not blockers — defaults are in place — but worth revisiting
when scaling experiments (Cell 1 / Cell 2 in STATUS.md) reveal what
actually matters.

### Q1 — Mechanical knobs upstream vs local
**Current default:** local subclass (Option B) —
`MetaExplorationGennyConfig` in `meta_exploration/`. **Trigger to
revisit:** if the mechanical knobs (k_candidates / perturbations /
refiner) prove broadly valuable across cube-harness agents, propose
upstream RFC against `GennyConfig`.

### Q2 — Mechanical vs textual exploration channel
**Current default:** ship the notes channel first (low cost; covers
the pilot phase). Mechanical channel deferred until scaling
experiments confirm exploration is the bottleneck.

### Q3 — When to surface RFCs upstream
**Current:** deferred. To be discussed with Alec when scaling
experiments inform what's most-broadly-useful.

---

## §7. References

- Methodology: [`SKILL.md`](SKILL.md)
- Status + how to run: [`STATUS.md`](STATUS.md)
- Investigator biasing: [`investigator_extra.md`](investigator_extra.md)
- Per-round template: [`templates/exp_config.py`](templates/exp_config.py)
- Ledger module: [`../../../meta_exploration/ledger.py`](../../../meta_exploration/ledger.py)
- Recipe router: [`../../../meta_exploration/recipe_router.py`](../../../meta_exploration/recipe_router.py)
- Promotion gate: [`../../../meta_exploration/promotion.py`](../../../meta_exploration/promotion.py)
- EpisodeConfig planner: [`../../../meta_exploration/planner.py`](../../../meta_exploration/planner.py)
- `AutoCubeOptions`: [`../../options.py`](../../options.py)
- Python-SDK driver: [`../../python_driver.py`](../../python_driver.py)
- Companion `hinter` use case: [`../hinter/SKILL.md`](../hinter/SKILL.md)
- Upstream `auto_cube/driver.py` (the LLM-driven driver we use today):
  [`../../driver.py`](../../driver.py)
- Meta-Harness paper: [arXiv 2603.28052](https://arxiv.org/abs/2603.28052)
