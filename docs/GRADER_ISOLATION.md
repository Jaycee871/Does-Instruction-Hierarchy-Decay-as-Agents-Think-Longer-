# Grader isolation boundary

IH-Challenge rows contain `grader_code_python`, whose public schema defines a Boolean `grade_output_correct(input_text, assistant_response)` function. The experiment runner must not execute that dataset-provided source in its own process.

`ih_decay.grader.grade_output_isolated` therefore sends the grader source and the two text arguments to a short-lived worker process. The worker applies several independent restrictions before calling the benchmark function:

- Python starts with `-I -S`;
- the child environment is scrubbed rather than inheriting API tokens or other experiment secrets;
- source is parsed with `ast` before execution;
- imports are limited to a small standard-library allowlist used for deterministic text/format checks;
- dangerous builtins and dunder attribute access are rejected;
- POSIX workers receive CPU, address-space, core-dump, file-size, and file-descriptor limits;
- the parent enforces a wall-clock timeout;
- only a JSON Boolean result is accepted from `grade_output_correct`.

The subprocess still uses Python `exec` internally after policy validation, but dataset code is never `exec`'d by the orchestration process itself.

## Threat-model note

This layer is defense in depth for benchmark-supplied graders, not a claim that Python AST filtering is a complete sandbox against arbitrary hostile code. If the threat model changes from a curated benchmark to untrusted attacker-authored programs, run the worker inside an OS/container/VM boundary with network and filesystem isolation.

## Result semantics

The public API returns one of four statuses:

- `ok`: the grader returned an actual Boolean; `correct` contains that value;
- `rejected`: static policy rejected the grader source;
- `timeout`: the child exceeded the wall-clock limit;
- `error`: the worker crashed, the grader raised, the expected function was absent, or the return value was not Boolean.

Only `ok` results should enter compliance/decay statistics. Rejected, timed-out, and errored rows must be reported separately rather than silently coerced to pass or fail.
