import pytest

from ih_decay.materialize import materialize_attack


def test_materialize_replaces_exactly_one_without_mutating_source():
    source = [
        {"role": "system", "content": "Keep the rule."},
        {"role": "user", "content": "Before <<<ATTACK>>> after."},
    ]
    result = materialize_attack(
        source,
        placeholder="<<<ATTACK>>>",
        attack_text="candidate",
    )
    assert result.replacements == 1
    assert result.messages[1]["content"] == "Before candidate after."
    assert source[1]["content"] == "Before <<<ATTACK>>> after."


def test_materialize_rejects_missing_placeholder():
    with pytest.raises(ValueError, match="exactly one"):
        materialize_attack(
            [{"role": "user", "content": "No placeholder"}],
            placeholder="<<<ATTACK>>>",
            attack_text="candidate",
        )


def test_materialize_rejects_duplicate_placeholder_by_default():
    with pytest.raises(ValueError, match="found 2"):
        materialize_attack(
            [{"role": "user", "content": "<<<A>>> then <<<A>>>"}],
            placeholder="<<<A>>>",
            attack_text="candidate",
        )
