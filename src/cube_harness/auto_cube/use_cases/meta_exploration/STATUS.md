# Meta-Exploration — Status

**Date:** 2026-05-28
**Branch:** `feat/meta-exploration` on `YangYongJin/cube-harness`
**Tests:** all passing (last full run: 1103 / 7 skipped / 14 deselected)

This doc replaces the earlier `HANDOFF.md` + `NEXT_SESSION_PROMPT.md`,
which were based on a wrong framing. Read this top to bottom.

---

## TL;DR

- **Basic AutoCUBE (hinter use case) already works upstream.** PR #441
  (rebased into this branch 2026-05-27 evening) added
  [`auto_cube/driver.py`](../../driver.py) — a headless Claude Agent
  SDK orchestrator. Run it via:

  ```
  uv run ch-auto-cube --use-case hinter --model claude-haiku-4-5 \
      --max-turns 800 --objective "<spec>"
  ```

  Claude reads [`hinter/SKILL.md`](../hinter/SKILL.md) as its system
  prompt, runs the methodology, dispatches `ch-investigate` per round,
  writes `REPORT.md` at the end.

- **Meta-exploration is a *future* AutoCUBE use case** we are
  building on this branch. It is NOT what `ch-auto-cube --use-case
  hinter` runs today. The Cell 1 / Cell 2 measurement experiments
  (basic hinter on TB-2 + miniwob, baseline vs 4× episodes) are about
  deciding whether the meta-exploration extension is worth the work.

---

## The two-layer story

| Layer | What it is | Status |
|---|---|---|
| **Basic AutoCUBE** (driver.py + use_cases/{debug,hinter}/SKILL.md) | The shipped infrastructure. Claude Agent SDK orchestrator + per-use-case SKILL methodology. Adds new use cases by writing a SKILL.md. | ✅ Shipped upstream |
| **Meta-exploration use case** (this branch's contribution) | A new SKILL.md + supporting modules that add a per-episode configuration policy on top of basic AutoCUBE. Composes with hinter (Option B). | 🟡 Built but not yet wired into the LLM-driven driver |

---

## What this branch actually built

All of the following live under
[`src/cube_harness/meta_exploration/`](../../../meta_exploration/)
and [`src/cube_harness/auto_cube/`](../..). They are NOT used by
`ch-auto-cube --use-case hinter` today — they are the meta-exploration
extension waiting to be exercised:

| File | Purpose | Status |
|---|---|---|
| `meta_exploration/ledger.py` | `~/auto_cube/hints.json` shared ledger (L2/L3/L4 taxonomy) | ✅ Tested |
| `meta_exploration/recipe_router.py` | L1 dispatch heuristic (which Investigator recipe per trajectory) | ✅ Tested |
| `meta_exploration/promotion.py` | Phase 1/2 candidate detection + re-test gate | ✅ Tested |
| `meta_exploration/planner.py` | EpisodeConfig 8-rule decision tree + per-task policy | ✅ Tested |
| `meta_exploration/agent_config.py` | `MetaExplorationGennyConfig` (pure-data subclass adding k_candidates / k_plans / perturbations / enable_refiner) | ✅ Tested |
| `auto_cube/python_driver.py` | A **Python-as-orchestrator** alternative to driver.py — programmatic Stages A → B → F. Useful for tests + future programmatic ablations. Not the path we use for the current experiments. | ✅ Tested |
| `auto_cube/options.py` | `AutoCubeOptions` ablation framework for meta-exploration cells (6 named recipes) | ✅ Tested |
| `auto_cube/use_cases/meta_exploration/SKILL.md` | **The methodology** — gets loaded as Claude's system prompt by `ch-auto-cube --use-case meta_exploration` once we wire it | ✅ Written |
| `auto_cube/use_cases/meta_exploration/investigator_extra.md` | Biasing fragment for per-trajectory Investigator | ✅ Written |
| `auto_cube/use_cases/meta_exploration/templates/exp_config.py` | Per-round Python config template | ✅ Written |
| `scripts/smoke/auto_cube_meta_exploration.py` | Smoke that exercises Stages A + B + F + the honest-split assertion end-to-end (Python driver path) | ✅ Passes on TB-2 fix-git |

---

## Current experiment in flight

Goal: *Does AutoCUBE (basic hinter) get meaningfully better when we
give it more episodes per task per iteration?* If yes → meta-exploration
has headroom. If no → ship hint quality instead.

| Cell | Benchmark | Episodes/task/iter | Iters | Driver | Status |
|---|---|---|---|---|---|
| 1 | miniwob 16/8/8 | 1 | 3 | `ch-auto-cube --use-case hinter` (Claude Agent SDK) | 🟡 Running (as of 13:40) |
| 1 | TB-2 16/8/8 | 1 | 3 | same | ⏸ Deferred until miniwob lands |
| 2 | miniwob 16/8/8 | 4 | 3 | same | ⏸ Pending Cell 1 |
| 2 | TB-2 16/8/8 | 4 | 3 | same | ⏸ Pending Cell 1 |

The headline number per cell = **OOD test win-rate** on 8 held-out
tasks the agent never saw during Phase 1 / Phase 2 — added on top of
the vanilla hinter methodology to give cleaner cross-cell comparison.

---

## What's outstanding (in priority order)

1. **Finish miniwob Cell 1** — let the running `ch-auto-cube` session
   complete; pull `REPORT.md` out
2. **Run TB-2 Cell 1** — same invocation, swap benchmark + task lists
3. **Run Cell 2 (4× episodes) for both benchmarks** — same objective with
   the "episodes per task per iter" constraint changed to 4 and a Python
   loop around 4 successive Experiment.run() calls per task per iter
4. **Compare**: if 4× cell ≥ baseline + 2σ on OOD → meta-exploration
   has headroom; otherwise shift to hint quality / Tier 4
5. **If go on meta-exploration:** wire the existing `SKILL.md` into
   `ch-auto-cube --use-case meta_exploration` and verify end-to-end
6. **If no-go:** the meta-exploration code still has value as the
   structured-action-space scaffolding for future work, but the cell
   ablation experiment becomes lower-priority

---

## How to launch new experiments (copy-paste)

### Basic hinter on miniwob
```bash
uv run ch-auto-cube --use-case hinter --model claude-haiku-4-5 \
    --max-turns 800 \
    --objective "...spec text — see this STATUS.md's adjacent
    objective_templates/ once we move them out of here..."
```

### Basic hinter on TB-2
Same invocation; just substitute the task lists + `max_steps=150`
(TB-2 needs more steps than miniwob's 30) + `INFRA_CONFIGS['local']`
in the exp_config.

### Future: meta-exploration use case
Once wired:
```bash
uv run ch-auto-cube --use-case meta_exploration --model claude-haiku-4-5 \
    --max-turns 1000 --objective "..."
```

---

## Key gotchas that bit us (so next session doesn't repeat)

1. **`uv run` breaks Ray + editable cube-* dependencies.** Use
   `.venv/bin/python` and `.venv/bin/ch-investigate` directly.
2. **The upstream hinter `exp_config.py` template hardcodes
   `max_steps=10`.** That's right for miniwob but FAR too small for
   TB-2 (needs 150). The `--objective` string must override.
3. **Claude's investigator confidence is integer 0–5**, NOT float
   0–1. Threshold should be ≥ 3 if you ever pick one in code; in the
   LLM-driven path Claude uses judgment, no threshold needed.
4. **Hint findings land in `episode_record.json:findings.task_hints`**
   (merged in by `ch-investigate`), not in a separate findings file.
5. **macOS sleep kills background runs.** Use `caffeinate -i -m -s`
   for overnight runs. AC power required for `-s`.
6. **Anthropic API keys in `.env`** need `load_dotenv(override=True)`
   because the shell often exports an empty `ANTHROPIC_API_KEY`.
