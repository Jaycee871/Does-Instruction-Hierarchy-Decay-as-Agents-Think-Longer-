# IH-Challenge schema audit

A prompt-free audit of all **27,570** released skeletons confirms two useful implementation facts:

1. every row declares an `attack_placeholder`, and the declared placeholder occurs verbatim in its defender template;
2. every row includes `grader_code_python` as a string, but this project does **not** execute that code inside the orchestration process.

## Template structure

The single-constraint, multi-constraint, and anti-overrefusal splits all use two-message defender templates. The input-conditioned split is more structurally diverse: **17,150** rows have two messages and **1,888** have three.

Importantly, message order must not be "normalized" by sorting roles. Input-conditioned templates include sequences such as:

- `developer -> system`
- `user -> system`
- `user -> developer`
- `system -> user -> system`
- `developer -> user -> developer`

Those sequences are part of the released benchmark representation. The materialization layer therefore preserves message order exactly and only substitutes the declared attack placeholder.

## Grader complexity

Programmatic grader strings vary substantially in size:

| split | min chars | mean chars | max chars |
| --- | ---: | ---: | ---: |
| single-constraint | 120 | 220.32 | 431 |
| multi-constraint | 457 | 1090.37 | 1858 |
| input-conditioned | 990 | 2835.27 | 6354 |
| anti-overrefusal | 151 | 448.45 | 776 |

This strengthens the decision to build a separate restricted grader service rather than using `exec()` in the experiment runner.

## Consequence for the longitudinal design

IH-Challenge releases task skeletons rather than a single canonical instantiated attack. For an Instruction Hierarchy Decay experiment, attacker adaptation cannot be allowed to vary with trajectory length: otherwise a later checkpoint could simply have a stronger attack.

The longitudinal protocol therefore uses a **frozen-attack design**:

1. generate one attack candidate per selected benchmark item using a fixed attacker configuration;
2. hash and freeze that candidate;
3. substitute the identical candidate into the template at every trajectory-length condition;
4. vary only the trajectory policy / checkpoint;
5. grade defender outputs in a separate evaluation layer.

This turns the attack into a paired item-level control rather than an uncontrolled time-varying intervention.

Machine-readable counts are stored in `reports/schema_audit_summary.json`.
