# Literature map

This project sits at the intersection of instruction hierarchy, long-horizon agent reliability, multi-turn constraint following, and prompt-injection robustness.

## What the literature already establishes

Existing work strongly supports **static** instruction hierarchy as a measurable capability, and several adjacent literatures report longitudinal degradation phenomena. What is still comparatively underdeveloped is a direct, matched test of whether *instruction hierarchy itself* decays as an agent trajectory grows longer.

Key neighboring findings identified in the initial Undermind review:

- **Instruction hierarchy evaluation:** IHEval formalizes conflict resolution across privileged and lower-priority instructions.
- **Instruction instability over dialogue:** prior work reports instruction drift within repeated self-chat / multi-turn settings.
- **Long-horizon goal drift:** agent goals and prohibitions can erode under extended interaction and environmental pressure.
- **Constraint-following degradation:** newer multi-turn benchmarks report worsening constraint satisfaction as conversations accumulate constraints.
- **Long-horizon reliability:** errors propagate and later decisions become conditionally more fragile after earlier off-canonical actions.
- **Indirect prompt injection:** tool outputs and external content create persistent lower-authority pressure that static prompt-injection evaluations can miss.

## Research gap used here

The motivating gap is narrower than "agents get worse when tasks are long." We ask:

> Holding the benchmark item and privileged instruction fixed, does the probability of respecting the instruction hierarchy change as trajectory length increases?

That formulation matters because generic long-horizon failure can arise from planning difficulty, tool errors, memory overload, or task complexity without any failure of authority resolution.

## Constructs to borrow

The review suggests several useful measurement ideas:

- paired short-vs-long trajectories;
- survival / hazard analysis for first hierarchy violation;
- per-step authority labels;
- recovery after violation rather than pass@1 only;
- separating prohibitions from positive requirements;
- delayed-trigger and persistent-memory conditions;
- process annotations to distinguish hierarchy failure from ordinary task failure.

## Planned synthesis

Before the confirmatory sweep, this file will be expanded into a citation-complete related-work matrix covering:

1. static instruction hierarchy benchmarks;
2. multi-turn instruction stability;
3. goal drift and long-context agent reliability;
4. prompt injection in tool-using agents;
5. constraint persistence and recovery metrics.

The full literature search is maintained in the project Undermind workspace so that paper selection and follow-up reading remain auditable.
