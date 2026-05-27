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
| 2 | brand-new task | `BASELINE` | baseline trajectory |
| 3 | any past reward in (0, 0.5) | `ESCALATE_MODEL` | capability-bound vs structural |
| 4 | ≥3 attempts, all 0 reward | `DIAGNOSE_PROFILING` | cheap confirmation before unsteerable |
| 5a | one+ fails + loop note | `DIAGNOSE_SCAFFOLDING` | route to scaffolding recipe |
| 5b | one+ fails + timeout note | `EXTEND_TIMEOUT` | bump bash to 600s |
| 5c | one+ fails + no notes | `WIDEN_SEARCH` | k=3 + topk_branch perturbation |
| 8 | fallthrough | `BASELINE` | safe fallback |

After any pick, a **budget check**: if the picked config's approx cost
exceeds remaining budget, downgrade to `DIAGNOSE_PROFILING` with the
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
- AutoCubeOptions: [`../../options.py`](../../options.py) (moved here from `meta_exploration/` 2026-05-27)
- Python-SDK outer-loop driver: [`../../python_driver.py`](../../python_driver.py) (moved here 2026-05-27; pairs with upstream PR #441's LLM-driven `../../driver.py`)
- Companion `hinter` use case: [`../hinter/SKILL.md`](../hinter/SKILL.md)
- Auto-CUBE README: [`../../README.md`](../../README.md)
- Meta-Harness paper: [arXiv 2603.28052](https://arxiv.org/abs/2603.28052)
- Meta-Harness codebase reference: [stanford-iris-lab/meta-harness](https://github.com/stanford-iris-lab/meta-harness)

---

## §9. Mechanical knobs vs textual notes — both as options

Update 2026-05-27: there are TWO complementary channels meta-exploration
can use to intervene. We support both as independently-togglable options
on ``AutoCubeOptions`` so we can ablate which channel actually matters.

### Channel 1 — mechanical knobs (the original design)

Fields on ``EpisodeConfig`` mechanically force the agent to behave
differently:

- ``model``, ``bash_default_timeout``, ``investigator_recipe``,
  ``apply_promotion`` — directly supported by upstream
  ``GennyConfig`` / ``TerminalToolConfig``; **no porting needed**.
- ``k_candidates``, ``k_plans``, ``perturbations``, ``enable_refiner``
  — implemented by PGEPA's modules
  (``pgepa_v2/agent/k_candidate.py``, ``plan_candidate.py``,
  ``perturbations/``, ``agent/refiner.py``); **requires porting into**
  ``meta_exploration/`` or as upstream RFC.

**Pro:** strong guarantee — the agent IS forced to explore differently
(no way for it to "skip" a perturbation).
**Con:** algorithm-porting cost (~half a day per module × 4 modules);
tight coupling between meta_exploration and Genny internals.

### Channel 2 — textual exploration notes (the lightweight alternative)

Meta-exploration's policy authors short text strings ("exploration
notes") per task, injected into the agent's prompt as a fourth
category of hint alongside ``benchmark_hint_prompt`` /
``task_clarification`` / ``task_hints``.

Examples:
- ``"This task has timed out twice — consider running pip install with
  longer timeouts."``
- ``"On the prior attempt the agent used ArrowUp on a slider with 50
  steps to go — consider PageUp/PageDown for big gaps."``
- ``"This task class has had recurring issues with file-path
  case-sensitivity — verify exact casing before edits."``

Storage: ``HintLedgerEntry.notes["exploration_notes"]`` per task. The
orchestrator injects them into the agent prompt alongside other text
hints (Stage D's authoring path is reused).

**Pro:** zero algorithm porting; cleanly composes with hinter (both
are text authors); cheap to iterate on what guidance works.
**Con:** weak guarantee — the agent MIGHT ignore the note; the
intervention is a *prompt change* not a *behavior change*.

### Both-as-options

New ``AutoCubeOptions`` fields:

- ``enable_mechanical_exploration_knobs: bool = True`` — when True, the
  planner's menu includes ``WIDEN_SEARCH`` (k_candidates=3 +
  perturbations) and ``ENABLE_REFINER`` (refiner=True). When False,
  those mechanical-only entries are filtered out (same pattern as
  ``allow_model_escalation``).
- ``enable_exploration_notes: bool = False`` — when True, the planner
  ALSO authors a textual ``exploration_note`` per task per iter (in
  addition to picking a config). The orchestrator injects these into
  the agent's prompt next iter.

Ablation table (extends DESIGN.md §2):

| Recipe variant | mechanical | notes | What it isolates |
|---|---|---|---|
| `meta_exploration_only` (current) | True | False | Original framing — mechanical channel only |
| `meta_exploration_notes_only` (NEW) | False | True | Lightweight text channel only |
| `meta_exploration_both` (NEW) | True | True | Overdetermined — maximum exploration signal |
| `meta_exploration_off` (= weak_noop) | False | False | Baseline |

The 2×2 sub-ablation lets us measure: does the **mechanical channel**
add value over the **textual channel** at the same level of policy
sophistication? If `notes_only` is competitive with `mechanical_only`,
then C1-C4 algorithm ports become low-value work and we can ship just
the notes channel.

### Implementation order (per HANDOFF.md priority shift)

Per the staged plan (pilot → seed → scaling → meta-exploration tests):

1. **Pilot phase:** wire **notes channel only** (it requires no
   porting). Stage D's text-hint authoring path is reused; planner
   gains a "draft note for this task" output alongside its
   ``EpisodeConfig`` pick.
2. **Scaling-experiments phase:** if exploration is confirmed as
   the bottleneck via uniform-config scaling sweeps, then **port the
   mechanical channel** (C1-C4). Without confirmation, this work is
   speculative.
3. **Comparison phase:** run the 2×2 notes/mechanical ablation to
   decide which channel ships in the final paper.

---

## §10. Open architectural questions (record for future sessions)

These are not blockers — defaults are in place — but they're worth
revisiting when scaling experiments reveal what actually matters.

### Q1 — Mechanical knobs upstream vs local (was H1)
**Current default:** local subclass (Option B) — `MetaExplorationGennyConfig`
in `meta_exploration/`. **Trigger to revisit:** if scaling experiments
show the mechanical knobs (k_candidates / perturbations / refiner) are
broadly valuable across cube-harness agents, propose upstream as an
RFC against `GennyConfig`. Alec sign-off would mark the migration.

### Q2 — Held-out tier strategy (was H2)
**Current default:** rotate per iter via `pick_held_out(seed=iter_idx)`
(implemented in `promotion.py`). Gives more held-out task coverage
across iters at the cost of comparability of verdicts.
**Alternative:** pin same N tasks for the whole sweep; makes
per-promotion verdicts directly comparable across iters but tests
fewer distinct held-out tasks. **No change planned** unless cross-iter
verdict comparability becomes the bottleneck.

### Q3 — When to surface RFCs upstream (was H3)
**Current:** deferred. To be discussed with Alec when scaling
experiments inform what's most-broadly-useful. Two candidates:
- The mechanical-knob extension to `GennyConfig` (Q1)
- The `outer_loop_driver` → `auto_cube/orchestrator.py` move (it's
  a general primitive, not meta-exploration-specific — see HANDOFF.md
  §3 refactor item)

### Q4 — Mechanical vs textual exploration channel (NEW, §9)
**Current default:** ship the notes channel first (low cost; covers
the pilot phase). Mechanical channel deferred until scaling
experiments confirm exploration is the bottleneck.
