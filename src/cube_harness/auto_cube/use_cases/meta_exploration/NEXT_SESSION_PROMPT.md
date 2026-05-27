# Next-Session Prompt

**Paste the section below into a fresh Claude Code session in
`~/Research/cube-harness/` to resume the meta-exploration work.**

---

# Resume meta-exploration on cube-harness

You're picking up research-grade work on the `feat/meta-exploration`
branch of a `cube-harness` fork. The work adds a **meta-exploration**
use case to Auto-CUBE — a per-episode configuration policy that
composes with the existing `hinter` use case. The branch is in a
known-good state (1068 tests passing); you're continuing the
implementation per a staged plan.

## Repo and branch

- **Upstream:** [`The-AI-Alliance/cube-harness`](https://github.com/The-AI-Alliance/cube-harness)
- **Fork (push target):** [`YangYongJin/cube-harness`](https://github.com/YangYongJin/cube-harness)
- **Local:** `~/Research/cube-harness/`
- **Branch:** `feat/meta-exploration`
- **HEAD when handed off:** `6c30b34b`

## First, read these three docs in order (~15 min)

1. **[`HANDOFF.md`](HANDOFF.md)** — start here. The §3 "What's remaining"
   section has a 4-tier priority list. The §4 "Open design questions"
   has decided answers — DO NOT re-derive them.
2. **[`DESIGN.md`](DESIGN.md)** — the *why*: research framing
   (training/inference distinction §1), the 6-cell ablation table (§2),
   architecture decisions (§3), the planner's decision tree (§4), the
   notes-vs-mechanical option (§9), open architectural questions (§10).
3. **[`SKILL.md`](SKILL.md)** — the *what*: methodology spec, what the
   use case does end-to-end.

After reading, your mental model should include:
- Option B compose-don't-subsume framing (meta-exploration ⊥ hinter)
- 4-level taxonomy (L1 Investigator recipe / L2 hint_type / L3 disposition / L4 promoted_to)
- The honest training/inference split (DESIGN.md §1 — why ESCALATE_MODEL is OK)
- The 6 named recipes for ablation (weak_noop / strong_noop / meta_harness / hinter_only / meta_exploration_only / combined)
- The user's staged plan: pilot → seed → SCALING (is exploration the bottleneck?) → meta-exploration

## Environment setup

```bash
cd ~/Research/cube-harness
git checkout feat/meta-exploration
git fetch upstream dev      # check for new upstream commits
git status                   # should be clean except for the local pyproject override

# Verify the local-dev pyproject override is still in place. If grep
# returns nothing, the override was lost — re-add per HANDOFF.md §5
# (it points cube-standard at the local clone).
grep -A 2 "cube-standard = { path" pyproject.toml

# Cube-standard cross-repo dep (cube-harness/dev needs symbols not on PyPI rc8).
cd cube-standard && git pull origin dev && cd ..

# Sync + verify clean test baseline
uv sync --all-extras
uv run pytest tests/ -q --no-header \
    -m "not slow and not live_api and not integration"
# Expect: 1068 passed, 7 skipped, 14 deselected
```

If the test count differs from 1068, something drifted — investigate
before proceeding.

## Immediate task

**Start at HANDOFF.md §3 Tier 1 item #1** — the architectural refactor
to move `outer_loop_driver.py` and `options.py` out of
`meta_exploration/` and into `auto_cube/` where they belong (they're
universal across use cases, not meta-exploration-specific). ~1.5 hrs.

Then Tier 1 #2 (`MetaExplorationGennyConfig` subclass with the no-methods
discipline test), then #3 (Stage B minimum viable runner), then #4 (plan.json
writer), then #5 (smoke run on 3-5 TB-2 tasks). Total Tier 1 ≈ 1 day.

Do NOT skip ahead to Tier 3 (meta-exploration tests) before Tier 2
(scaling experiments) tells us whether exploration is actually the
bottleneck. If scaling experiments show uniform extra compute already
saturates performance, meta-exploration is speculative and the
priorities should shift toward hint-quality work instead.

## Critical constraints (do not violate)

1. **DCO signing required.** Every commit needs `git commit -s` (cube-harness
   constitution enforces it via CI).
2. **The pyproject.toml `[tool.uv.sources]` cube-standard override is
   LOCAL-DEV-ONLY** — see comment in the file. Never commit it. Before
   PR-prep, revert it and add `Depends-on: cube-standard/dev` to the PR
   description body per cube-harness/CLAUDE.md §Cross-repo PRs.
3. **Algorithm ports stay LOCAL to `meta_exploration/`** per the
   answered Q1 (DESIGN.md §10). Don't push them into upstream
   `cube_harness/agents/` without explicit user sign-off + Alec RFC.
4. **The honest training/inference assertion is load-bearing.**
   `assert_retest_uses_inference_model` in `options.py` must fire
   before any re-test episode launches. If you ever weaken or skip
   it, the entire research claim becomes invalid. See DESIGN.md §1.
5. **NEW design questions need user sign-off, not silent decisions.**
   If you hit a branching architectural call not covered in DESIGN.md
   §10, surface it and wait.

## What to surface to the user

- Tier 1 progress at each item completion
- Any tests dropping below 1068
- Any need for an architectural call not covered in DESIGN.md §10
- Any cost spend over $5 for smoke runs

## What you should NOT do

- Start TaskCreate/TaskUpdate tracking unless the user asks (they
  explicitly preferred docs-only handoff)
- Run Tier 2/3/4 work before Tier 1 lands cleanly
- Modify any commits already on origin (the 9 commits on the branch
  are signed + pushed; rewrite history requires user sign-off)
- Push to `upstream` (The-AI-Alliance) directly; always push to
  `origin` (YangYongJin) and PR
- Create new use cases or modules without surfacing first

## User context

- The user is YangYongJin (dyyjkd@kaist.ac.kr) at KAIST
- They've been coordinating with Alec (the upstream cube-harness
  author) on this contribution; Alec authored the auto-cube hinter use case (PR #438)
- The user's preferred style: brief progress reports, no jargon
  walls, ask before making non-obvious calls

## When you're done with your session

Update `HANDOFF.md` with what landed + what remains, write a
companion `NEXT_SESSION_PROMPT.md` if structure changed, commit and
push. Tell the user the branch HEAD + test count + what they should
read first.
