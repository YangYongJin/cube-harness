## Auto-CUBE meta-exploration use case — Investigator biasing

You are being dispatched by Auto-CUBE running its **meta-exploration**
use case. The orchestrator above you is not just authoring hints — it
is also adapting the per-episode agent configuration (model choice,
k-candidate count, perturbation policy, refiner toggle) based on the
disposition history of each task. Your findings feed both the hint
authoring AND the configuration policy.

### What the orchestrator does with your output

- The standard `BaseFindings` + `task_hints[]` (when you're dispatched
  as the `hinter` recipe) flow into the auto-cube ledger
  (`~/auto_cube/hints.json`) and drive the existing Phase 1 / Phase 2
  hinter discipline.
- **Additionally**, signals you record drive the configuration policy:
  - `primary_blame == "agent_scaffolding"` → orchestrator considers
    bumping `k_candidates` / enabling `topk_branch` perturbation /
    enabling refiner for this task's next attempts.
  - `primary_blame == "model_capability"` → orchestrator considers
    bumping `model` to a stronger tier for this task.
  - `primary_blame == "task_setup"` (build/install timeout, missing
    image dep) → orchestrator considers bumping
    `bash_default_timeout` or switching to a heavier infra.
  - Persistent loops / response-action mismatch → orchestrator considers
    routing this task to `agent_scaffolding` recipe next iter instead
    of `hinter`.

So **be specific about which configuration knob would help**. Don't
just say "the agent ran out of context" — say "tokens-per-step ≈ 70k
× 200 steps; agent thrashed; suggest reducing context via
`enable_summarize=True` OR bumping `max_steps` to give more headroom."

### Calibration discipline

- Reserve `primary_blame_confidence >= 4` for cases where you have
  verbatim transcript evidence + a clean alternative hypothesis is
  excluded. The orchestrator weights your confidence when picking the
  next config — overconfident dispatch wastes compute on the wrong knob.
- An empty `task_hints[]` is still a strong signal — it routes the task
  away from `hinter` to `general_blame` / `agent_scaffolding` /
  `profiling`, which then drives a different *configuration* response
  rather than no response.

### What helps the orchestrator most

- **Per-task disposition reasoning.** When you observe a task that has
  failed across multiple iters with different configs, note which
  configs have already been tried and what new direction is worth trying.
- **Cross-task pattern recognition.** When you see the same failure
  shape across N tasks, flag it — the orchestrator will consider a
  global config bump rather than per-task tweaks (Phase 2 promotion
  candidates apply to configuration knobs as well as to hint text).
- **Cost-aware suggestions.** When suggesting a config bump, qualify
  the cost: `"k_candidates=5 on this task ≈ 5× cost; only justified if
  the agent shows real action-selection confusion (not just bad
  context)"`. The policy applies cost-quality trade-offs explicitly.

### What this use case is NOT for

- Pure hint quality without configuration adaptation: use `hinter`.
- Bug-finding sweeps: use `debug`.
- A single one-shot fix: use direct human iteration.
