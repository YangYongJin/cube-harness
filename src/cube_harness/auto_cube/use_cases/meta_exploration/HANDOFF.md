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

### Tier 1 — load-bearing for any sweep to actually run

These 3 stages are stubbed in `outer_loop_driver.py` with explicit
`§Contract` docstrings. Each one is ~3-5 hrs.

#### 1. **Stage B episode runner** — make the driver actually launch
   episodes through cube-harness's `Experiment` / `exp_runner`.

   File: [`outer_loop_driver.py:stage_b_launch_episode`](../../../meta_exploration/outer_loop_driver.py)
   Currently: takes a `runner=` callable; default raises `NotImplementedError`.
   Needed: a default runner that translates `EpisodeConfig` →
   `GennyConfig` (+ `LLMConfig`, perturbations, etc.) and launches one
   episode via `Experiment(benchmark_config=..., agent_config=...).run()`.

   Sticky bits:
   - `EpisodeConfig.perturbations` is a tuple of strings — needs a
     registry mapping names to actual perturbation modules.
     PGEPA's `src/pgepa_v2/perturbations/` is the reference; needs
     porting into cube-harness or wired as a separate registry.
   - `EpisodeConfig.enable_refiner` needs PGEPA's refiner module
     similarly ported.
   - `bash_default_timeout` needs to be threaded into
     `cube.tools.terminal.TerminalToolConfig` for TB-style cubes.

#### 2. **Stage C — Investigator dispatch via cube_harness SDK**

   File: [`outer_loop_driver.py:stage_c_dispatch_investigator`](../../../meta_exploration/outer_loop_driver.py)
   Contract: per trajectory, pick a recipe (via `recipe_router` or
   planner override) and invoke the matching
   `cube_harness.analyze.investigator.use_cases.<recipe>.recipe`
   (which uses `claude_agent_sdk` internally per the `investigator`
   optional dep).

   Verify before wiring: that upstream's investigator module exposes
   a callable interface (not just a `recipe.py` with constants).
   `cube_harness/analyze/investigator/use_cases/hinter/recipe.py` has
   `RECIPE = InvestigatorRecipe(...)` — needs a dispatch function that
   takes a trajectory + recipe and returns `BaseFindings`.

#### 3. **Stage D — text-hint authoring + Meta-Harness branches**

   File: [`outer_loop_driver.py:stage_d_author_hints`](../../../meta_exploration/outer_loop_driver.py)
   Two paths:
   - Vanilla: aggregate `task_hints[]` from Stage C findings; mutate
     `GennyConfig.task_hints`.
   - Meta-Harness: if `opts.propose_multiple_harnesses=True`, do step
     K times and pick best by validation reward; if
     `opts.enable_pareto_verify=True`, gate the mutation on Pareto
     improvement over per-task rewards.

   Reference for the Meta-Harness algorithm: arXiv 2603.28052 +
   [stanford-iris-lab/meta-harness](https://github.com/stanford-iris-lab/meta-harness).

#### 4. **Stage E — Phase 2 promotion + re-test gate**

   File: [`outer_loop_driver.py:stage_e_phase2_promotion`](../../../meta_exploration/outer_loop_driver.py)
   Contract: detect candidates (text via
   `promotion.candidate_from_task_hints_overlap`; config via a sibling
   `candidate_from_episode_config_overlap` that needs to be written);
   per candidate, run a sub-experiment on (affected + held-out)
   tasks; verdict via `promotion.decide_verdict`; mutate ledger
   disposition accordingly.

   The honest-split assertion fires inside Stage B during the re-test
   episodes — no extra work needed beyond ensuring re-test episodes
   set `EpisodeConfig.model = opts.inference_model`.

### Tier 2 — observability + ergonomics

#### 5. **Plan.json writer** — at end of each iter, serialize the
   `PlannerDecision` per task to `output_dir/iter_<k>/plan.json`.
   Already named in `DESIGN.md`; just needs to be written. ~30 min.

#### 6. **CLI entry point** — `python -m
   cube_harness.meta_exploration.outer_loop_driver --recipe combined
   --benchmark terminalbench2 ...`. The driver's `run_outer_loop`
   takes positional args; the CLI is a thin Typer wrapper following
   cube-harness's existing CLI conventions
   ([`scripts/experiments_report.py`](../../../../scripts/experiments_report.py)
   is the canonical example). ~1 hr.

#### 7. **Smoke runs against real TB-2 cube** — once Stages B/C/D/E
   land, run `weak_noop` and `combined` on a small subset
   (3-5 tasks, 1 iter) to confirm end-to-end plumbing.

### Tier 3 — paper-grade ablation

#### 8. **Full 6-cell sweep** on TB-2 (or MW or both). Plan:
   - 16 train + 16 held-out tasks per benchmark
   - 3 iters per recipe
   - 2 seeds per cell (for noise band)
   - 6 recipes × 32 tasks × 3 iters × 2 seeds = ~1100 episodes
   - At $0.10 per episode (mid estimate): **~$110 budget**
   - Wall: 3-6 hours per cell × 6 cells = potentially overnight
   - Output: 6-cell table with mean ± SD wins, total $, $/win, time

---

## §4. Open design questions for the next session

### 1. Where do the perturbations + refiner modules live?

PGEPA has `src/pgepa_v2/perturbations/` and `src/pgepa_v2/agent/refiner.py`.
These need to either (a) be ported into `cube_harness/meta_exploration/`
as part of the contribution, or (b) be PR'd upstream into a more
appropriate location (cube-harness's `agents/` for refiner; new
top-level `perturbations/` for the others). Alec's call.

### 2. How does `EpisodeConfig → GennyConfig` translation work?

The driver's Stage B needs to build a `GennyConfig` per episode that
reflects the planner's `EpisodeConfig`. Some mappings are direct
(`model` → `LLMConfig.model_name`), some need wiring
(`k_candidates` → PGEPA's `PgepaGennyConfig.k_candidates`, which
doesn't exist yet in cube-harness's `GennyConfig`).

Two options:
- (a) Extend upstream `GennyConfig` with `k_candidates` /
  `k_plans` / `enable_refiner` / `perturbations` fields — needs Alec
  RFC since `GennyConfig` is core.
- (b) Subclass `GennyConfig` inside `meta_exploration/` as
  `MetaExplorationGennyConfig`, like PGEPA's `PgepaGennyConfig`.
  Lighter touch, doesn't require Alec sign-off.

### 3. Should Investigator-derived notes (`loop_pattern_suspected`,
`build_timeout_suspected`) be code-emitted or hand-flagged?

Currently the planner's decision tree reads these notes from the
ledger but nothing populates them. Two options:
- (a) Add a pass that walks each iter's trajectories, computes loop
  detection + timeout detection from the trace, and writes to the
  ledger. Pure-function over trajectory data.
- (b) Make the Investigator do this as part of Stage C dispatch.
  Cleaner architecturally but ties planner to Stage C being wired.

### 4. Held-out tier — same set every iter or rotated?

`promotion.pick_held_out` is seeded by iter index → varies across
iters. Is that what we want? Alternative: pin the held-out set for
the whole sweep so verdicts are comparable across iters.

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
