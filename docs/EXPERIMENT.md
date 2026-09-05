# Experiment protocol: Instruction Hierarchy Decay

## Objective

Test whether instruction-hierarchy compliance changes as a model is embedded in progressively longer agentic trajectories.

The study is designed to distinguish **static competence** from **persistence under continued processing**. A model that passes an instruction-conflict item at step 1 may still fail later if lower-priority content, its own prior outputs, tool results, or accumulated context alters effective control.

## Primary question

For the same IH-Challenge item and model, does the probability of satisfying the privileged instruction change as trajectory length increases?

## Hypotheses

- **H0 (persistence):** compliance is invariant to trajectory length after controlling for item and decoding settings.
- **H1 (decay):** compliance decreases with trajectory length.
- **H2 (task interaction):** decay differs across single-constraint, multi-constraint, input-conditioned, and anti-overrefusal tasks.
- **H3 (authority interaction):** decay depends on privileged and attacker levels (system/developer vs developer/user conflicts).
- **H4 (recovery):** some models recover after an intermediate violation; first failure therefore should not be treated as equivalent to permanent failure.

H1 is the motivating hypothesis, not an assumed result.

## Initial trajectory lengths

Primary checkpoints:

`1, 3, 5, 10, 20`

A 50-step stress condition may be added only after the initial sweep establishes cost and context-length feasibility.

## Minimal trajectory conditions

The first experiment should include at least two conditions so that trajectory length is not confounded with new adversarial content.

### A. Persistence-only trajectory

- Start from the original IH-Challenge conversation.
- Preserve the privileged instruction verbatim at every checkpoint.
- Continue the interaction using neutral task-progress turns or model self-revision.
- Do not introduce any new malicious instruction after step 1.

This condition asks whether hierarchy control drifts merely because the trajectory continues.

### B. Accumulating lower-priority context

- Start from the same original item.
- Add task-relevant, lower-priority observations or tool-like outputs over time.
- Keep privileged instructions fixed.
- Do not change the intended answer criterion.

This condition tests whether accumulated context increases interference with the original authority structure.

A later study may manipulate tool calls, memory, retries, or self-reflection separately.

## Unit of analysis

The same benchmark item should be evaluated at every trajectory length for a given model/condition. This paired structure supports item-fixed comparisons and avoids mistaking item difficulty for temporal decay.

Recommended record key:

`model × condition × source_file × row_index × seed × checkpoint`

## Outcomes

### Primary

**Compliance probability by checkpoint.**

Estimate the change in pass probability as a function of trajectory length. Report both raw paired proportions and a model with item-level controls or random effects when sample size permits.

### Secondary

- `absolute_decay = compliance(step_1) - compliance(step_T)`
- slope of compliance over step count
- normalized area under the compliance curve
- first-failure step
- recovery rate after first failure
- over-refusal rate

Over-refusal must remain separate from hierarchy failure: a model can avoid violating a privileged instruction while still refusing benign requests unnecessarily.

## Controls

Hold constant within a matched sweep:

- model and model version;
- temperature / top-p / max tokens;
- system and developer instructions;
- benchmark item;
- seed where the serving API exposes one;
- grader version;
- trajectory-generation policy.

Record token counts because step count and context length are related but not identical. Later analyses should include both.

## Important confounds

1. **Longer context vs longer reasoning.** A 20-step trajectory contains more tokens. Record cumulative tokens and, when possible, create token-matched controls.
2. **Repeated prompting.** Repeating the same instruction may strengthen rather than weaken compliance. Preserve wording consistently and explicitly tag repetition conditions.
3. **Self-generated contamination.** The agent's previous outputs can become context. Analyze this as a mechanism, not noise.
4. **Grader validity.** The official benchmark includes Python grader code. Do not execute downloaded grader strings indiscriminately inside the orchestration process. Use a restricted evaluation environment with timeouts and versioned grader wrappers.
5. **Provider-side nondeterminism.** API-served models may change over time. Record exact model identifiers and timestamps.

## Phase plan

### Phase 0 — scaffold

- bucket-native dataset loader
- dataset metadata audit
- metric implementation
- deterministic tests
- literature search
- credential validation

### Phase 1 — small pilot

- 25–50 matched items per task family
- one model
- checkpoints 1/3/5/10/20
- persistence-only condition
- inspect trajectories manually for construct validity

### Phase 2 — confirmatory sweep

- larger stratified sample
- multiple models / model sizes
- persistence-only + accumulating-context conditions
- preregister analysis before full sweep

### Phase 3 — mechanisms

Only after an effect is observed: tool use, memory, retry loops, self-reflection, context compression, and explicit hierarchy reminders.

## Decision rule for calling something "decay"

Do not label a single late-step failure as Instruction Hierarchy Decay. The term should be reserved for a reproducible negative association between trajectory length and hierarchy compliance under a controlled trajectory policy, with uncertainty reported and alternative explanations checked.
