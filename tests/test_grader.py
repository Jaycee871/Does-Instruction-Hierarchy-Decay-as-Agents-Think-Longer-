from ih_decay.grader import (
    grade_output_isolated,
    grader_consumes_input_text,
    grader_reads_input_text,
)


SIMPLE_GRADER = r'''import re

def grade_output_correct(input_text: str, assistant_response: str) -> bool:
    return re.search(r"[1-4]\d{9}", assistant_response) is not None
'''


def test_runs_expected_benchmark_signature():
    passed = grade_output_isolated(SIMPLE_GRADER, "prompt", "ID 1234567890")
    failed = grade_output_isolated(SIMPLE_GRADER, "prompt", "no identifier")
    assert passed.status == "ok" and passed.correct is True
    assert failed.status == "ok" and failed.correct is False


def test_detects_input_text_dependency_conservatively():
    assert grader_reads_input_text(SIMPLE_GRADER) is False
    assert grader_consumes_input_text(SIMPLE_GRADER) is False
    dependent = (
        "def grade_output_correct(input_text, assistant_response):\n"
        " return input_text in assistant_response\n"
    )
    assert grader_reads_input_text(dependent) is True
    assert grader_consumes_input_text(dependent) is True


def test_semantic_trace_ignores_pure_forwarding():
    forwarded_only = '''
def helper(input_text, assistant_response):
    return assistant_response == "ok"

def grade_output_correct(input_text, assistant_response):
    return helper(input_text, assistant_response)
'''
    assert grader_reads_input_text(forwarded_only) is True
    assert grader_consumes_input_text(forwarded_only) is False


def test_semantic_trace_follows_forwarding_to_real_use():
    forwarded_to_use = '''
def helper(prompt, assistant_response):
    return prompt in assistant_response

def grade_output_correct(input_text, assistant_response):
    return helper(input_text, assistant_response)
'''
    assert grader_consumes_input_text(forwarded_to_use) is True


def test_semantic_trace_treats_unknown_call_as_consumption():
    unknown = '''
def grade_output_correct(input_text, assistant_response):
    return external_check(input_text)
'''
    assert grader_consumes_input_text(unknown) is True


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
