import pytest

from ih_decay.trajectory import (
    NEUTRAL_CONTINUATION_V1,
    TrajectoryPlan,
    append_neutral_continuation,
    checkpoint_events,
)


def test_default_plan_is_strictly_increasing():
    plan = TrajectoryPlan()
    assert plan.checkpoints == (1, 3, 5, 10, 20)
    assert plan.continuation_policy == "neutral_continue_v1"


def test_checkpoint_events_marks_only_requested_steps():
    events = dict(checkpoint_events(5, checkpoints=(1, 3, 5)))
    assert events == {1: True, 2: False, 3: True, 4: False, 5: True}


def test_invalid_checkpoint_order_is_rejected():
    with pytest.raises(ValueError):
        TrajectoryPlan(checkpoints=(1, 5, 3))


def test_neutral_continuation_preserves_original_messages():
    original = [
        {"role": "system", "content": "Keep this exactly."},
        {"role": "user", "content": "Do the task."},
    ]
    extended = append_neutral_continuation(original, assistant_output="First answer")

    assert original == [
        {"role": "system", "content": "Keep this exactly."},
        {"role": "user", "content": "Do the task."},
    ]
    assert extended[:2] == original
    assert extended[-2] == {"role": "assistant", "content": "First answer"}
    assert extended[-1] == {"role": "user", "content": NEUTRAL_CONTINUATION_V1}


def test_neutral_continuation_rejects_empty_output():
    with pytest.raises(ValueError):
        append_neutral_continuation([], assistant_output="   ")
