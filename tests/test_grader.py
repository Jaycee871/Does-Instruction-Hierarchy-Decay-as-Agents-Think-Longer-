from ih_decay.grader import grade_output_isolated


SIMPLE_GRADER = r'''import re

def grade_output_correct(input_text: str, assistant_response: str) -> bool:
    return re.search(r"[1-4]\d{9}", assistant_response) is not None
'''


def test_runs_expected_benchmark_signature():
    passed = grade_output_isolated(SIMPLE_GRADER, "prompt", "ID 1234567890")
    failed = grade_output_isolated(SIMPLE_GRADER, "prompt", "no identifier")
    assert passed.status == "ok" and passed.correct is True
    assert failed.status == "ok" and failed.correct is False


def test_rejects_non_allowlisted_import():
    code = "import os\ndef grade_output_correct(input_text, assistant_response):\n return True\n"
    result = grade_output_isolated(code, "x", "y")
    assert result.status == "rejected"
    assert result.correct is None


def test_rejects_dunder_escape_attempt():
    code = "def grade_output_correct(input_text, assistant_response):\n return bool((1).__class__)\n"
    result = grade_output_isolated(code, "x", "y")
    assert result.status == "rejected"


def test_timeout_kills_nonterminating_grader():
    code = "def grade_output_correct(input_text, assistant_response):\n while True:\n  pass\n"
    result = grade_output_isolated(code, "x", "y", timeout_s=0.25)
    assert result.status in {"timeout", "error"}
    assert result.correct is None


def test_requires_named_boolean_grader():
    missing = grade_output_isolated("x = 1", "x", "y")
    non_bool = grade_output_isolated(
        "def grade_output_correct(input_text, assistant_response):\n return 1\n",
        "x",
        "y",
    )
    assert missing.status == "error"
    assert non_bool.status == "error"
