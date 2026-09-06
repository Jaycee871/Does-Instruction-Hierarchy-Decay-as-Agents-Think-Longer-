# Phase 1 construct decisions

## What the pilots changed

The original question was whether instruction-hierarchy compliance deteriorates as an observable agentic trajectory becomes longer. The Phase 1 pilots showed that this cannot be measured responsibly by simply repeating an arbitrary `continue` message and reading the final checkpoint.

Three design facts now have direct empirical support in this repository:

1. **Compliance is not necessarily monotonic.** Items can fail and later recover, so every generated step should be graded even when only a subset is used as planned checkpoints.
2. **Step-1 conditioning matters.** A post-step-1 persistence analysis is meaningful only for trajectories whose initial output passes the benchmark grader.
3. **Continuation wording is part of the intervention.** The replicated multi-constraint probe showed that `Continue.` and a task-preserving continuation can enter different behavioral regimes immediately after an identical step-1 state.

Therefore the project will not treat continuation text as an invisible implementation detail.

## Operational construct

The phrase "think longer" in the repository title is operationalized as **a longer observable agentic interaction trajectory**: more externally visible model turns in which prior outputs are retained in context. It does not refer to hidden chain-of-thought tokens or private reasoning traces.

The current persistence condition uses the versioned lower-priority cue:

`neutral_continue_v1 = "Continue the same task and provide your current best answer."`

This wording is not claimed to be uniquely neutral. It is a documented trajectory policy whose sensitivity is measured separately.

## Primary estimand for the next matched-item phase

For a fixed model, task family, attack, and continuation policy, let `C(L)` denote the probability that the official benchmark grader passes at observable trajectory step `L`, **conditional on the same trajectory passing at step 1**.

The simplest descriptive decay quantity remains:

`IHD(L) = C(1) - C(L)`

Because the analysis is conditioned on step-1 pass, `C(1) = 1` by construction in the persistence-at-risk cohort. `IHD(L)` is a proposed project metric, not an established literature metric.

The primary analysis should report the full conditional compliance curve rather than only a single endpoint.

## Secondary trajectory outcomes

The next phase should also preserve:

- first failure step;
- recovery after first failure;
- number of pass/fail state transitions;
- all-step normalized area under the compliance curve;
- cumulative completion-token use;
- checkpoint-only results for comparability with the preregistered `1,3,5,10,20` grid.

These outcomes separate monotonic loss from oscillation and recovery.

## Replication requirement

A single trajectory per item is no longer considered sufficient for the main estimate. Provider/model nondeterminism is visible even at temperature zero. The next scaled experiment should therefore run repeated trajectories within item and use item-level paired or hierarchical analysis rather than treating each generated trajectory as an independent benchmark item.

## Anti-overrefusal remains a required control

A system that preserves high-priority constraints by refusing everything is not demonstrating useful hierarchy adherence. Anti-overrefusal items remain necessary to distinguish genuine instruction-priority competence from blanket refusal behavior.

## Claim boundary

Phase 1 supports a measurement-design claim: instruction-hierarchy persistence over long agentic trajectories is sensitive to trajectory construction, can be non-monotonic, and requires repeated within-item measurement.

Phase 1 does **not** yet support a benchmark-wide claim that instruction hierarchy decays with trajectory length.
