# Meta-Exploration — Session Handoff

**Status:** branch in good shape, foundational architecture landed,
specific algorithm wiring remaining. Read this top-to-bottom to pick up.

**Date of handoff:** 2026-05-27
**Branch:** `feat/meta-exploration` on `YangYongJin/cube-harness` (fork
of `The-AI-Alliance/cube-harness`)
**Tests:** 1068 passed, 7 skipped, 14 deselected — clean baseline.

---

## §1. What's done

7 commits on `feat/meta-exploration` (newest first):

```
63499be8 feat(meta-exploration): outer-loop SDK driver — v1 (A + B + F)
2e09ae31 feat(meta-exploration): AutoCubeOptions — ablation framework
a3943ff8 refactor(meta-exploration): verb-first menu names
8a13b846 docs(meta-exploration): DESIGN.md — research framing + decision log
553e4caa feat(meta-exploration): EpisodeConfig planner + Option B framing
47ac9f3f feat(auto_cube): add meta-exploration use case skeleton
cb722430 feat(meta-exploration): scaffold module + port ledger/router/promotion
```

### File layout

```
src/cube_harness/meta_exploration/                # the module
├── __init__.py
├── ledger.py               # ~/auto_cube/hints.json — L2/L3/L4 taxonomy
├── recipe_router.py        # L1 dispatch heuristic
├── promotion.py            # Phase 1/2 candidate detection + re-test gate
├── planner.py              # EpisodeConfig + per-task policy
├── options.py              # AutoCubeOptions — ablation framework
└── outer_loop_driver.py    # The unattended runner (v1: A+B+F wired)

src/cube_harness/auto_cube/use_cases/meta_exploration/  # the use case
├── SKILL.md                # methodology spec (loaded as system prompt)
├── investigator_extra.md   # Investigator biasing for L1 routing
├── DESIGN.md               # research framing + decision log
├── HANDOFF.md              # THIS FILE
└── templates/exp_config.py # per-round Python config template

tests/test_meta_exploration_{ledger,recipe_router,promotion,
                             planner,options,outer_loop_driver}.py
```

### Test counts per module

| Module | Tests | Passing |
|---|---|---|
| ledger | 24 | ✅ |
| recipe_router | 21 | ✅ |
| promotion | 22 | ✅ |
| planner | 22 | ✅ |
| options | 25 | ✅ |
| outer_loop_driver | 18 | ✅ |
| **Total new** | **132** | ✅ |
| Pre-existing cube-harness baseline | 936 | ✅ |
| **Grand total** | **1068** | ✅ |

---

## §2. The research story (don't lose this framing)

### Two-line claim

> *We turn the auto-cube hinter loop into a training pipeline: a
> per-task policy picks training-time exploration configs (possibly
> using strong teacher models) to gather information for downstream
> hinter authoring. Inference uses a fixed weak base agent + the
> hints/configs the training loop produced — same distillation pattern
> as RLHF or synthetic-data SFT.*

### Why this is honest

`ESCALATE_MODEL` in the planner's menu uses `azure/gpt-5` (teacher).
This would be cheating if it leaked into the headline number. The
**honest-split guard** lives in [`options.py:assert_retest_uses_inference_model`](../../../meta_exploration/options.py)
and is called from [`outer_loop_driver.py:stage_b_launch_episode`](../../../meta_exploration/outer_loop_driver.py)
before any re-test episode launches. The Phase 2 promotion gate's
verdicts are produced under `opts.inference_model` only — guaranteed
by assertion, not by vibes.

### The 6-cell ablation table (memorize this)

| Recipe | Inference model | What ablates |
|---|---|---|
| `weak_noop` | gpt-5-mini | floor — base agent alone |
| `strong_noop` | gpt-5 | naive-scaling ceiling ("just throw compute") |
| `meta_harness` | gpt-5-mini | prior-art (Pareto-verify + multi-harness) |
| `hinter_only` | gpt-5-mini | our hint-authoring contribution alone |
| `meta_exploration_only` | gpt-5-mini | our config-axis contribution alone |
| `combined` | gpt-5-mini | **THE PROPOSAL** — full system |

Story we want: `combined > hinter_only > meta_harness > weak_noop`,
AND `combined ≥ strong_noop` (or `combined/cost >> strong_noop/cost`).

See [`DESIGN.md`](DESIGN.md) §1 + §2 for the full framing.

---

## §3. What's remaining (in priority order)

**Staged plan** (updated 2026-05-27): we're NOT going straight at the
6-cell paper-grade ablation. Order is:

1. **Pilot tests** — minimum-viable end-to-end (Stage B + smoke); does
   the architecture work?
2. **Few-seed tests** — multi-seed runs to establish noise floor
3. **Scaling experiments** — *the load-bearing scientific question:
   is exploration actually the bottleneck?* If uniform extra compute
   on more episodes / longer rollouts doesn't help, then
   meta-exploration won't help either and the meta-exploration work
   is speculative.
4. **Meta-exploration tests** — only after #3 confirms exploration is
   the bottleneck.

This shifts what's Tier 1 vs Tier 2 vs Tier 3.

### Tier 1 — load-bearing for the pilot phase

#### 1. **Architectural refactor — move orchestrator + options to `auto_cube/`** (~1.5 hrs)
   - `meta_exploration/outer_loop_driver.py` → `auto_cube/orchestrator.py`
     (it's not meta-exploration-specific; debug + hinter can use it too)
   - `meta_exploration/options.py` → `auto_cube/options.py` (same — universal)
   - Only `planner.py` stays in `meta_exploration/` — that's the actual
     meta-exploration contribution
   - Update imports + tests + this HANDOFF doc + DESIGN.md
   - **Do early** — every follow-up file lands in the right home then.

#### 2. **MetaExplorationGennyConfig subclass** (Option B — see DESIGN.md §10 Q1) (~30 min)
   - File: `meta_exploration/agent_config.py` (new)
   - Pure data extension over upstream `GennyConfig`. Adds:
     `k_candidates: int = 1`, `k_plans: int = 1`,
     `enable_refiner: bool = False`, `perturbations: list[str] = []`
   - **DISCIPLINE PIN:** no method overrides — pinned via a comment AND
     a test asserting `set(MetaExplorationGennyConfig.__dict__).difference(set(GennyConfig.__dict__)) == {set of field names}`.
     This keeps the B→A migration trivial (~5 min) if Alec eventually
     accepts the upstream RFC.

#### 3. **Stage B episode runner — minimum viable** (~2-3 hrs)
   - File: `outer_loop_driver.py:stage_b_launch_episode` (or wherever
     after refactor)
   - Translate `EpisodeConfig` → `MetaExplorationGennyConfig` (use
     the subclass; mechanical fields just pass through) →
     `Experiment(benchmark_config=, agent_config=).run()`.
   - **Minimum viable:** only need `model`, `bash_default_timeout`,
     `investigator_recipe`, `apply_promotion` translation. The
     `k_candidates` / `k_plans` / `perturbations` / `enable_refiner`
     fields can stay no-op until Stage B v2 (Tier 3 if needed).
   - The honest-split assertion already fires (committed in v1).

#### 4. **Plan.json writer + minimal Stage F polish** (~30 min)
   - End of each iter: dump `dict[task_id, PlannerDecision]` to
     `output_dir/iter_<k>/plan.json`. Already designed; just needs
     code in `orchestrator.py`. Critical for debugging the pilot runs.

#### 5. **Smoke runs** (~$2-5, half hour wall)
   - `weak_noop` and `combined` on 3-5 tasks, 1 iter, TB-2
   - Confirms end-to-end plumbing of A + B + F + the honest-gate
     assertion.

### Tier 2 — scaling experiments (the load-bearing scientific question)

#### 6. **Uniform-config scaling sweep** (~$20-50, depending on scope)
   - Goal: answer "is exploration the bottleneck?"
   - Hold config fixed (e.g. `weak_noop` recipe); vary compute via
     `experiences_per_task` (1 / 3 / 5 / 10) and `max_steps`
     (100 / 200 / 400) — both knobs already in cube-harness's
     `Experiment` config.
   - Measure: per-task win-rate as a function of compute budget.
   - **Decision criterion for meta-exploration**:
     - If win-rate plateaus quickly (e.g. no improvement from 3→10
       replicas) → exploration ISN'T the bottleneck → meta-exploration
       deprioritized; investigate hint quality (Tier 4) instead.
     - If win-rate keeps improving with more replicas/steps → exploration
       IS the bottleneck → proceed to Tier 3 meta-exploration tests.

#### 7. **Few-seed runs of the same scaling sweep** (~$50-100)
   - 3-5 seeds to bracket the noise band.
   - Same axes (replicas / max_steps) but with seed variance.
   - Outputs the actual signal-to-noise ratio meta-exploration needs
     to beat.

### Tier 3 — meta-exploration tests (conditional on Tier 2 outcome)

#### 8. **Investigator-emitted ledger notes** (~2-3 hrs)
   - File: `outer_loop_driver.py:stage_c_dispatch_investigator`
   - Wire to `cube_harness.analyze.investigator` (verify the
     callable interface first per Stage C contract docstring).
   - Populate ledger notes (`loop_pattern_suspected`,
     `build_timeout_suspected`) from trajectory analysis.
   - Without this, the planner's notes-based rules (5a, 5b) never fire.

#### 9. **Stage D vanilla — text-hint authoring + exploration notes** (~3-4 hrs)
   - File: `outer_loop_driver.py:stage_d_author_hints`
   - Aggregate text-hint candidates from Stage C findings; mutate
     `GennyConfig.task_hints`.
   - **NEW: also author exploration notes** (DESIGN.md §9) when
     `opts.enable_exploration_notes=True`. Lightweight text channel
     for meta-exploration that avoids the algorithm-port cost.
   - Both kinds of text flow into the agent's prompt via the same
     hint-injection mechanism.

#### 10. **Stage E — Phase 2 promotion + re-test gate** (~4 hrs)
   - File: `outer_loop_driver.py:stage_e_phase2_promotion`
   - Detect candidates via
     `promotion.candidate_from_task_hints_overlap` (text path);
     similar function for config-knob candidates if needed.
   - Run re-test sub-experiment on (affected + held-out) tasks.
   - The honest-split assertion fires inside Stage B during the
     re-test episodes.

#### 11. **Algorithm ports — CONDITIONAL on Tier 2 outcome + DESIGN.md §9 ablation** (~half-day each)
   - **C1** Port `pgepa-v2/src/pgepa_v2/agent/k_candidate.py`
   - **C2** Port `pgepa-v2/src/pgepa_v2/agent/plan_candidate.py`
   - **C3** Port `pgepa-v2/src/pgepa_v2/perturbations/`
   - **C4** Port `pgepa-v2/src/pgepa_v2/agent/refiner.py`
   - **Only do these if:** scaling experiments confirm exploration is
     the bottleneck AND DESIGN.md §9 ablation shows the mechanical
     channel beats the notes channel.
   - If notes channel alone is competitive → skip C1-C4 entirely.

#### 12. **Promotion-rung extensions** (~1-2 hrs each)
   - **E1** Extend `promotion.apply_promotion_to_config` to handle
     `task_clarification` rung
   - **E2** Extend it to handle `description_overrides` rung
   - `new_action` + `system_prompt` stay Mode-C-only per design.

### Tier 4 — paper-grade ablation (only if Tier 3 produces a signal)

#### 13. **Meta-Harness branches in Stage D** (~4-6 hrs)
   - `enable_pareto_verify` — gate hint mutations on per-task Pareto improvement
   - `propose_multiple_harnesses` — propose K candidate hint sets, pick best
   - Reference: arXiv 2603.28052 + stanford-iris-lab/meta-harness

#### 14. **Typer CLI entry point** (~1 hr)
   - `python -m cube_harness.auto_cube.orchestrator --recipe combined ...`
   - Thin wrapper following `scripts/experiments_report.py` conventions.

#### 15. **Full 6-cell ablation sweep** (~$100-150, overnight)
   - 6 recipes × 16+16 tasks × 3 iters × 2 seeds = ~1100 episodes
   - Output: 6-cell table with mean ± SD wins, total $, $/win, time
   - **Add the 2×2 notes/mechanical sub-ablation** from DESIGN.md §9
     if both channels were implemented in Tier 3.

---

## §4. Open design questions — answered (record for future sessions)

### Q1 — Where do the algorithm modules live? (formerly H1)
**Decision (2026-05-27):** Option B — port locally into
`meta_exploration/` (not upstream). Trigger to revisit: if scaling
experiments + DESIGN.md §9 ablation show the mechanical knobs are
broadly valuable across cube-harness agents, propose upstream RFC.
See DESIGN.md §10 Q1.

### Q2 — Held-out tier rotation strategy (formerly H2)
**Decision (2026-05-27):** Sweep by each iter — keep
`pick_held_out(seed=iter_idx)` as the default (already implemented
in `promotion.py`). Gives more held-out task coverage at the cost
of per-iter verdict comparability. No change needed.
See DESIGN.md §10 Q2.

### Q3 — When to surface upstream RFCs to Alec (formerly H3)
**Deferred:** to be discussed with Alec async; tentative timing is
after scaling experiments inform what's most-broadly-useful.
See DESIGN.md §10 Q3.

### Q4 — Mechanical knobs vs textual exploration notes (NEW 2026-05-27)
**Decision:** ship BOTH as independently-togglable options on
`AutoCubeOptions`. Pilot phase wires the **notes channel only**
(zero algorithm-porting cost); mechanical channel deferred until
scaling experiments confirm exploration is the bottleneck. The
DESIGN.md §9 2×2 sub-ablation tells us which channel ships in the
paper. See DESIGN.md §9 + §10 Q4.

### Inline TODOs (low-priority, easy to forget)
- Audit `recipe_router.py`'s heuristic thresholds against the actual
  signals in upstream `analyze/investigator/use_cases/agent_scaffolding/SKILL.md`
  (currently my guesses: `_MIN_REPEAT_FOR_SCAFFOLDING = 3`,
  `_HIGH_TOKENS_PER_STEP = 50_000`)

---

## §5. How to resume

```bash
# 1. Set up environment
cd ~/Research/cube-harness   # the YangYongJin fork
git checkout feat/meta-exploration
git pull --rebase upstream dev   # if upstream has new commits
uv sync --all-extras

# 2. Set up cube-standard (cross-repo dep — see Cube-Harness CLAUDE.md
# §Cross-repo PRs). The cube-standard dev branch has the
# ValidatedConfig + IncompatibleInfraError symbols cube-harness needs.
# Cloned at ~/Research/cube-harness/cube-standard (gitignored).
cd cube-standard && git pull origin dev && cd ..

# 3. The local pyproject override is UNCOMMITTED and must stay that
# way. It points cube-standard at the local clone:
#
#   [tool.uv.sources]
#   cube-standard = { path = "cube-standard", editable = true }
#
# Verify it's still there:
grep -A 2 "cube-standard" pyproject.toml
# If missing, re-add (see commit 8a13b846 in pgepa-v2's CUBE_HARNESS_MIGRATION.md)

# 4. Verify clean state
uv run pytest tests/ -q --no-header \
    -m "not slow and not live_api and not integration"
# Expect: 1068 passed, 7 skipped, 14 deselected

# 5. Start at Tier-1 item #1 (Stage B episode runner). Reference:
cat src/cube_harness/meta_exploration/outer_loop_driver.py | \
    grep -A 25 "def stage_b_launch_episode"
```

---

## §6. Companion docs

- [`SKILL.md`](SKILL.md) — user-facing methodology spec; loaded as
  Claude Code system prompt by the `/auto-cube-meta-exploration` skill
- [`DESIGN.md`](DESIGN.md) — research framing + architecture decisions
  + decision log
- [`investigator_extra.md`](investigator_extra.md) — Investigator
  biasing fragment for L1 routing
- [`templates/exp_config.py`](templates/exp_config.py) — per-round
  Python config template (for interactive Mode-C rounds)
- Upstream `auto_cube/README.md` — the methodology framework these
  use cases plug into
- Upstream `auto_cube/use_cases/hinter/SKILL.md` — the companion
  use case meta-exploration composes with

---

## §7. Quick-reference: how the modules connect

```
                          AutoCubeOptions (recipe)
                                   │
                                   ▼
                       run_outer_loop (driver)
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
   Stage A: plan          Stage B: run                Stage F: ledger
   ──────────────         ──────────────              ──────────────
   PlannerState ←─── ledger.py                       upsert_entry
        │              ↑                                  ↑
        ▼              │                                  │
   plan_iter      assert_retest_                          │
   (from planner)  uses_inference_model                   │
        │              │                                  │
        │              ▼                                  │
        │         runner(task_id, cfg)                    │
        │              │                                  │
        │              ▼                                  │
        │         per_task_rewards ───────────────────────┘
        │
        ▼ (Stages C/D/E stubbed; iterate here next)
   findings → text hints → promotion gate
                          │
                          ▼
                    decide_verdict
                    (promotion.py)
                          │
                          ▼
                    apply_promotion_to_config
                    (promotion.py)
                          │
                          ▼
                    ledger.disposition=promoted | cheat_only
```
