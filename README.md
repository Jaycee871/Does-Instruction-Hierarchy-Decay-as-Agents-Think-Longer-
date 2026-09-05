# Does Instruction Hierarchy Decay as Agents Think Longer?

An empirical study of whether instruction-hierarchy compliance remains stable across long-horizon agentic trajectories.

## Research question

A model may correctly prioritize higher-authority instructions at the beginning of a task. Does that priority remain stable after repeated reasoning, revision, retries, tool use, memory accumulation, and self-correction?

This repository treats **trajectory length** as an experimental variable rather than assuming that one-shot instruction-following performance transfers unchanged to multi-step agents.

## Core hypothesis

We define **Instruction Hierarchy Decay (IHD)** as a systematic reduction in compliance with higher-priority instructions as an agentic trajectory grows longer.

The first preregistered-style sweep uses trajectory checkpoints:

`1 -> 3 -> 5 -> 10 -> 20 steps`

Primary outcomes:

- hierarchy compliance at each checkpoint;
- decay slope across trajectory length;
- first-failure step;
- recovery after a violation;
- area under the compliance curve;
- over-refusal rate, analyzed separately from hierarchy failure.

## Data

The initial benchmark is OpenAI's **IH-Challenge** dataset. A working copy is available in the public Hugging Face bucket:

`hf://buckets/Jaycee766/ih-challenge-bucket/`

Expected source files:

- `single-constraint.jsonl`
- `multi-constraint.jsonl`
- `input-conditioned.jsonl`
- `anti-overrefusal.jsonl`

The loader is bucket-native and uses Hugging Face's `HfFileSystem` interface, so the 157 MB dataset does not need to be committed to GitHub.

## Design principle

The project separates three things that are often conflated:

1. **Static hierarchy competence** — can the model resolve a conflict at step 1?
2. **Trajectory persistence** — does the same constraint remain respected after continued agentic processing?
3. **Recovery dynamics** — after a violation, can the agent return to the privileged instruction without being explicitly reset?

The main analysis is paired: the same benchmark item is evaluated at multiple trajectory lengths to reduce item-level confounding.

## Repository layout

```text
configs/                 experiment schedules
src/ih_decay/            loaders, trajectory utilities, metrics
scripts/                  inspection and experiment entry points
tests/                    deterministic unit tests
docs/EXPERIMENT.md        study protocol and hypotheses
.github/workflows/        CI and credential checks
```

## Quick start

```bash
python -m pip install -e .[dev]
python scripts/inspect_dataset.py --limit 20
pytest -q
```

For authenticated access set `HF_TOKEN`; GitHub Actions maps the repository secret `HF_TOKEN1` to that environment variable when needed.

## Status

**Phase 0 — research scaffold.** Dataset access, metrics, trajectory schedule, CI, and literature review are being established before any expensive model sweep.

The goal is not to prove that longer reasoning is inherently worse. The null hypothesis — stable hierarchy compliance across trajectory length — is treated as a real possibility.

## Reproducibility rule

Raw benchmark data remain external. Every experiment should record dataset source, file name, item identifier, model identifier, decoding parameters, trajectory length, seed, model outputs, grader result, and timestamp. Generated traces belong in artifact storage rather than Git history.
