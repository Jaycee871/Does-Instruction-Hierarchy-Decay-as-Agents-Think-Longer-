# IH-Challenge grader first-argument contract

Before expanding the graded pilot, we audited all 27,570 public IH-Challenge grader programs to determine what the first argument of `grade_output_correct(...)` actually means. This matters because the four public dataset splits do not use one uniform parameter name or one uniform semantic contract.

## Full-dataset audit

The Python 3.12 audit found exactly two first-parameter names:

| First parameter | Rows | Contract |
| --- | ---: | --- |
| `attacker_output` | 19,038 | Exact attack payload used to construct the defender input |
| `input_text` | 8,532 | Compatibility parameter; no audited grader semantically consumes it |

Static inter-procedural tracing over the whole dataset classified the first argument as:

- semantically consumed: 18,989 rows;
- forwarded through local helper functions only: 6,474 rows;
- never loaded: 2,107 rows.

All 18,989 semantically consuming graders are in `input-conditioned.jsonl`, whose public grader first parameter is `attacker_output`. The remaining 49 rows in that split merely forward `attacker_output`, but we still pass the exact frozen attack text for a single consistent contract.

All 8,532 graders whose first parameter is named `input_text` are in the other three splits. None semantically consumes that value: 6,425 only forward it to helpers that ignore it, and 2,107 never load it.

### Split breakdown

| Split | Rows | First parameter | Consumes | Forward-only | No load |
| --- | ---: | --- | ---: | ---: | ---: |
| `single-constraint.jsonl` | 1,900 | `input_text` | 0 | 0 | 1,900 |
| `multi-constraint.jsonl` | 4,832 | `input_text` | 0 | 4,832 | 0 |
| `anti-overrefusal.jsonl` | 1,800 | `input_text` | 0 | 1,593 | 207 |
| `input-conditioned.jsonl` | 19,038 | `attacker_output` | 18,989 | 49 | 0 |

## Runtime rule

`ih_decay.grader_contract.resolve_grader_input` encodes the audited rule and fails closed:

1. If the first parameter is `attacker_output`, pass the exact frozen attack text that was inserted into the defender prompt.
2. If the first parameter is `input_text`, verify again at runtime that static tracing finds no semantic consumption, then pass the empty string as an inert compatibility value.
3. If a future dataset revision introduces another first-parameter name, or an `input_text` grader begins consuming that value, abort instead of guessing.

This preserves the matched-design invariant: the same frozen attack is used both in the defender prompt and, where required, as the benchmark grader's conditioning input.

## Python compatibility

The first full audit attempted under Python 3.11 exposed a public grader containing f-string syntax that parses under Python 3.12 (PEP 701) but not 3.11. Full benchmark grader auditing and execution therefore use Python 3.12. The rest of the package can remain compatible with the project's broader Python range; this is specifically a benchmark-code compatibility requirement.

## Evidence and scope

The successful semantic audit ran at commit `3f70c4c8bf93adf6b75513f084d418cf3a5a02a8` in GitHub Actions run `33982356514`. Its uploaded artifact was `grader-input-contract-audit` (artifact ID `9974137164`, SHA-256 `a6c2bbeb3d937f2917d072c436b0966d56d1e6f9200d5dd3003a0234e67be90f`). A compact frozen summary is stored at `reports/grader_input_contract_summary.json`.

These observations validate grader wiring only. They are not model-performance measurements and are not evidence for or against instruction-hierarchy decay.
