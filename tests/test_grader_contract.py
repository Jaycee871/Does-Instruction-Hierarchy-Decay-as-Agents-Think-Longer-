import pytest

from ih_decay.grader_contract import (
    UnsupportedGraderContract,
    grader_input_parameter_name,
    resolve_grader_input,
)


def test_passes_frozen_attack_to_attacker_output_grader():
    code = '''
def grade_output_correct(attacker_output, defender_output):
    return attacker_output in defender_output
'''
    assert grader_input_parameter_name(code) == "attacker_output"
    value, mode = resolve_grader_input(code, frozen_attack_text="PAYLOAD")
    assert value == "PAYLOAD"
    assert mode == "frozen_attack_text"


def test_uses_empty_compatibility_value_when_input_text_is_unused():
    code = '''
def grade_output_correct(input_text, assistant_response):
    return assistant_response == "ok"
'''
    value, mode = resolve_grader_input(code, frozen_attack_text="PAYLOAD")
    assert value == ""
    assert mode == "empty_string_verified_unused"


def test_uses_empty_value_when_input_text_is_only_forwarded_to_unused_helper():
    code = '''
def helper(input_text, assistant_response):
    return assistant_response == "ok"

def grade_output_correct(input_text, assistant_response):
    return helper(input_text, assistant_response)
'''
    value, mode = resolve_grader_input(code, frozen_attack_text="PAYLOAD")
    assert value == ""
    assert mode == "empty_string_verified_unused"


def test_fails_closed_if_input_text_becomes_semantically_consumed():
    code = '''
def grade_output_correct(input_text, assistant_response):
    return input_text in assistant_response
'''
    with pytest.raises(UnsupportedGraderContract):
        resolve_grader_input(code, frozen_attack_text="PAYLOAD")


def test_fails_closed_for_unknown_parameter_name():
    code = '''
def grade_output_correct(prompt, assistant_response):
    return assistant_response == "ok"
'''
    with pytest.raises(UnsupportedGraderContract):
        resolve_grader_input(code, frozen_attack_text="PAYLOAD")
