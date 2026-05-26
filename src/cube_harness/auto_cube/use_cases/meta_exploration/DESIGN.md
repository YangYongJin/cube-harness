# Meta-Exploration — Design Doc

Companion to [`SKILL.md`](SKILL.md). Captures the **design decisions
+ research framing** behind the meta-exploration use case so future
sessions (and reviewers) don't have to re-derive them.

SKILL.md is *what* the use case does. This doc is *why* it does it
that way.

---

## §1. Research framing — the training/inference distinction

**The core insight:** the meta-exploration outer loop IS a training
pipeline. The expensive exploration (stronger models on hard tasks,
wider action sampling, mid-rollout refiners, perturbations) is
training-time compute, analogous to:

| Standard ML pattern | PGEPA equivalent |
|---|---|
| Training-data generation with GPT-4 (expensive teacher) | Meta-exploration loop with `ESCALATE_MODEL` allowed |
| Synthetic / rejection-sampled training set | Trajectories from the 8-config menu × perturbations × replicas |
| Smaller model deployed at inference | Base agent (e.g. gpt-5-mini) + PGEPA-generated hints |
| Distillation: teacher knowledge → student | Strong-model trajectories' info → hints that help weak agent |

As long as the **inference run uses only the fixed baseline model +
the artifacts the training loop produced**, the comparison is honest.
Same as nobody complains that "GPT-4 generated the SFT data" when the
deployed model is a small open-source one.

### What makes this honest (code-enforced controls)

1. **`AutoCubeOptions.inference_model`** is a separate field from the
   training-time model menu. The re-test gate (Phase 2) uses **only**
   the inference model. Assertion-enforced, not vibes.
2. **Dedicated `inference_eval` mode** in the outer-loop SDK driver —
   runs with `inference_model` only, no menu, no exploration. This is
   the final-eval pass that produces the headline number.
3. **Ledger records `inference_model` per task** alongside the
   training-time configs used. Reviewers can audit any per-task model
   choice post-hoc.

### The hints-are-agent-specific claim

PGEPA's hints aren't generic LLM tips — they're tailored to *what the
base agent gets wrong on a specific task*. Compare:

- Generic hint: *"For slider tasks, use keyboard navigation."*
- PGEPA hint: *"gpt-5-mini tends to use ArrowUp (1px/press) on sliders;
  explicitly use PageUp/PageDown for big gaps."*

GPT-5 alone gets the generic version for free from its better reasoning.
But gpt-5-mini + the SPECIFIC hint can be competitive, because the hint
encodes the model-specific failure pattern that even GPT-5 wouldn't have
surfaced. This is *distilled compute*: spend the GPT-5 budget during
training to discover the gotchas, then run the cheap agent with those
gotchas pre-encoded at inference.

---

## §2. The baseline ablation table

These are the cells the design supports without extra glue code:

| | Inference model | Training hints | Training exploration | What it measures |
|---|---|---|---|---|
| `weak-noop` | gpt-5-mini | none | none | floor — base agent alone |
| `strong-noop` | gpt-5 | none | none | naive-scaling ceiling — "just throw compute" |
| `meta-harness` | gpt-5-mini | Meta-Harness | none | prior-art baseline on hint authoring |
| `hinter` | gpt-5-mini | PGEPA hinter | default config | our hint authoring contribution |
| `meta-exploration-only` | gpt-5-mini | none | meta-exploration policy | config-axis contribution in isolation |
| `combined` (PROPOSAL) | gpt-5-mini | PGEPA hinter | meta-exploration policy | full system |

### Story we want the table to tell

- **`combined > hinter > meta-harness > weak-noop`** → each piece adds
  value
- **`combined ≥ strong-noop`** → distilled compute matches naive scaling
  at a fraction of inference cost
- **`combined/cost >> strong-noop/cost`** → headline result
  (stronger-effective agent for cheap-model price at inference)

If `combined ≥ strong-noop`: excellent result.
If `combined < strong-noop but combined/cost >> strong-noop/cost`:
still excellent, different framing but still compelling.

---

## §3. Architecture decisions

### §3.1 Option B — composes with hinter, doesn't subsume it

Meta-exploration is **strictly the per-episode configuration policy
axis**. It does NOT author hints — hint authoring lives in the
companion `hinter` use case. The two compose via **data flow**, not
code coupling:

```
meta-exploration's Stage A:    hinter's Stage D:
  picks per-episode config  →  consumes resulting trajectory  →
                                 → authors text hint
                                 → enters re-test gate (Stage E)
```

Neither use case imports the other. They share infrastructure
(`meta_exploration/ledger.py`, `recipe_router.py`, `promotion.py`).
Run combinations via `AutoCubeOptions` (Pivot 5) — `HINTER_ONLY`,
`META_EXPLORATION_ONLY`, `COMBINED` are all named subsets.

**Why not the superset version?** Superset blurs the contribution —
"what specifically does meta-exploration add over hinter?" is hard to
ablate cleanly. Option B's composition lets us run the 5-cell ablation
above without entangling the two axes.

### §3.2 The 4-level Auto-CUBE taxonomy

Inherited from upstream `auto_cube/use_cases/hinter/SKILL.md`. Used by
both `hinter` and `meta_exploration`:

- **L1** Investigator recipe (per failed episode): `hinter` /
  `general_blame` / `agent_scaffolding` / `profiling`
- **L2** `hint_type` (per hint): `clarification` / `task_specific` /
  `general_guidance`
- **L3** `disposition` (per task): `open` / `steered` / `promoted` /
  `cheat_only` / `not_a_hint` / `unsteerable`
- **L4** `promoted_to` (per promoted hint): the regularization ladder
  rung — `task_hints` / `benchmark_hint_prompt` / `task_clarification`
  / `description_overrides` / `new_action` / `system_prompt`

### §3.3 Bounded action menu (8 named EpisodeConfigs)

The planner picks from a **bounded menu** of 8 named configs rather
than composing arbitrary EpisodeConfig field values. Trade-off:

|  | Menu (chosen) | Free composition |
|---|---|---|
| Action space size | 8 | 8 fields × ~3 values = ~3000 combos |
| Ablation tractable? | ✅ | ❌ |
| Credit assignment | ✅ named-pick → outcome | ❌ which knob mattered? |
| Cold-start sane? | ✅ heuristic picks known-good | ❌ no prior on combos |
| Expressiveness | Limited to 8 | Unlimited |

**v2 expansion path:** once we have data on which knobs win, the policy
can move from menu-picking to bandit over the full `EpisodeConfig`
space.

---

## §4. The planner's decision tree (Pivot 4)

Pure-function, 8-rule, first-match-wins. Implemented in
[`meta_exploration/planner.py:pick_episode_config`](../../../../meta_exploration/planner.py).

| # | Trigger | Pick | Info gathered |
|---|---|---|---|
| 1 | `cheat_only` AND ≥2 attempts | `RETEST_PROMOTION` | does the cheat generalize? (Phase 2 gate fires) |
| 2 | brand-new task | `DEFAULT` | baseline trajectory |
| 3 | any past reward in (0, 0.5) | `ESCALATE_MODEL` | capability-bound vs structural |
| 4 | ≥3 attempts, all 0 reward | `PROFILING_DIAGNOSIS` | cheap confirmation before unsteerable |
| 5a | one+ fails + loop note | `DIAGNOSE_SCAFFOLDING` | route to scaffolding recipe |
| 5b | one+ fails + timeout note | `EXTEND_TIMEOUT` | bump bash to 600s |
| 5c | one+ fails + no notes | `WIDEN_SEARCH` | k=3 + topk_branch perturbation |
| 8 | fallthrough | `DEFAULT` | safe fallback |

After any pick, a **budget check**: if the picked config's approx cost
exceeds remaining budget, downgrade to `PROFILING_DIAGNOSIS` with the
original pick logged in `alternatives_considered`.

### Tunable thresholds (open to revision)

- Near-miss threshold = `(0, 0.5)` exclusive
- Stuck-streak threshold = 3 attempts
- Cheat-only retest threshold = 2 attempts
- Approx episode costs in `_APPROX_EPISODE_COST_USD` dict — empirical,
  refined as we collect real cost data

---

## §5. The naming convention

Menu entries follow **verb-first, action-oriented** naming after the
Pivot 4 rename. Each name describes what the policy is *doing* when it
picks this config, not what the underlying fields look like:

| Verb-first name | Underlying EpisodeConfig |
|---|---|
| `BASELINE` (was `DEFAULT`) | defaults — establishes baseline prior |
| `WIDEN_SEARCH` (was `EXPLORATORY_BREADTH`) | k_candidates=3, perturbations=("topk_branch",) |
| `ESCALATE_MODEL` (was `STRONG_MODEL`) | model="azure/gpt-5" |
| `ENABLE_REFINER` (was `REFINER_ENABLED`) | enable_refiner=True |
| `RETEST_PROMOTION` (unchanged) | apply_promotion=True |
| `EXTEND_TIMEOUT` (was `LONG_BUILD_TASK`) | bash_default_timeout=600 |
| `DIAGNOSE_SCAFFOLDING` (was `SCAFFOLDING_DIAGNOSIS`) | investigator_recipe="agent_scaffolding" |
| `DIAGNOSE_PROFILING` (was `PROFILING_DIAGNOSIS`) | investigator_recipe="profiling" |

(Rename happens before Pivot 5's ablation framework wraps around the menu.)

---

## §6. Conversation log — design decisions in chronological order

This section captures the actual back-and-forth that produced the
above. Future sessions can audit *why* the design ended up this way.

### 2026-05-26 — Phase 4 of pgepa-v2 (predecessor work)

- Built per-iter ledger writes, Investigator-recipe routing (chunk 3),
  promotion + re-test gate (chunk 4) — all opt-in inside pgepa-v2.
- Hit OOD null result on TB-2 noop reproduction (3/16 wins vs handoff's
  7/16; within N=1 noise but motivated a structural pivot).

### 2026-05-26 — Pivot to cube-harness (after Alec meeting)

- Decision: stop building inside pgepa-v2, fork cube-harness directly,
  contribute back. Use case framing: meta-exploration as a sibling
  use case to debug / hinter in `auto_cube/use_cases/`.
- Drop: scripted ace / gepa / jef_hinter (no longer needed).
- Add: Meta-Harness (arXiv 2603.28052) as baseline; outer-loop SDK
  driver as the unattended runner.
- Architectural framing (Alec-inspired): treat method comparisons as
  **option subsets of AutoCube** — Meta-Harness = enable-these-knobs,
  PGEPA meta-exploration = enable-those-knobs.

### Pivots 1-4 (this session)

- Stood up `YangYongJin/cube-harness:feat/meta-exploration`.
- Copied ledger / recipe_router / promotion modules from pgepa-v2 into
  `cube_harness/meta_exploration/`, adapted shape to upstream
  `GennyConfig` / `BenchmarkClarifications`.
- Scaffolded use case dir with SKILL.md / investigator_extra.md /
  templates/exp_config.py.
- Built EpisodeConfig planner (8-rule heuristic over PlannerState).

### Design discussion (2026-05-27)

**Q: should meta-exploration ⊃ hinter?** (Option A vs B vs C)
**A:** Option B — compose via data flow. Cleaner ablation,
contribution boundary explicit. Locked into SKILL.md.

**Q: does meta-exploration also need to control exploration to feed
hinter better info?**
**A:** Yes — that's the *real* purpose. Reframed meta-exploration as
"information-gathering policy" not "compute allocator". Reward
signal becomes downstream-aware:
`α * episode_success + β * hint_passes_retest + γ * disposition_advances - δ * cost`

**Q: is ESCALATE_MODEL cheating?**
**A:** Only under "PGEPA wins more" framing. Under
**"PGEPA-as-training-loop"** framing it's legitimate — strong model
is a teacher used at training time; deployment uses fixed baseline.
This unlocks the standard distillation / synthetic-data pattern. The
re-test gate uses only `inference_model`, so strong-model hints
that don't generalize get filtered out.

**Q: what's the baseline that exposes the value?**
**A:** Full-GPT-5-no-hints (`strong-noop`). Hypothesis: it won't beat
`combined` because GPT-5 alone has no info about *the base agent's*
failure patterns; PGEPA hints encode that agent-specific info.

**Q: naming is inconsistent?**
**A:** Yes — renaming to verb-first action-oriented form (see §5).

---

## §7. Open questions / future work

### v2 — learned policy
Replace the heuristic decision tree with a bandit over the menu (or
the full EpisodeConfig space). Same API, learned internals. Inputs:
accumulated `(PlannerState, EpisodeConfig, outcome)` triples.

### Menu expansion
Once we observe which knobs matter, expand the menu — e.g.,
`WIDEN_SEARCH_PLUS_REFINER`, `ESCALATE_AND_DIAGNOSE`, etc. Keep
bounded for ablation.

### Per-task multi-replica scheduling
v1 plans 1 episode per task per iter. v2 could plan
`{config_A: 1 replica, config_B: 2 replicas}` per task to do
within-iter ablations cheaply.

### Cross-session policy transfer
The ledger already carries per-task `disposition` across sessions.
The planner could learn task-class → config priors from accumulated
sessions, transferring policy across runs.

### Investigator-driven note authoring
Ledger notes (`loop_pattern_suspected`, `build_timeout_suspected`)
are hand-flagged in v1. The Investigator could populate these
automatically from trajectory analysis — closes the loop between
L1 dispatch and the planner.

---

## §8. References

- Use case methodology: [`SKILL.md`](SKILL.md)
- Investigator biasing: [`investigator_extra.md`](investigator_extra.md)
- Per-round template: [`templates/exp_config.py`](templates/exp_config.py)
- Ledger module: [`../../../meta_exploration/ledger.py`](../../../meta_exploration/ledger.py)
- Recipe router: [`../../../meta_exploration/recipe_router.py`](../../../meta_exploration/recipe_router.py)
- Promotion gate: [`../../../meta_exploration/promotion.py`](../../../meta_exploration/promotion.py)
- EpisodeConfig planner: [`../../../meta_exploration/planner.py`](../../../meta_exploration/planner.py)
- AutoCubeOptions (Pivot 5): TBD
- Outer-loop SDK driver (Pivot 7): TBD
- Companion `hinter` use case: [`../hinter/SKILL.md`](../hinter/SKILL.md)
- Auto-CUBE README: [`../../README.md`](../../README.md)
- Meta-Harness paper: [arXiv 2603.28052](https://arxiv.org/abs/2603.28052)
- Meta-Harness codebase reference: [stanford-iris-lab/meta-harness](https://github.com/stanford-iris-lab/meta-harness)
