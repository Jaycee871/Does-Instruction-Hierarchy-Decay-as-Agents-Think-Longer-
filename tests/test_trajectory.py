import pytest

from ih_decay.trajectory import TrajectoryPlan, checkpoint_events


def test_default_plan_is_strictly_increasing():
    plan = TrajectoryPlan()
    assert plan.checkpoints == (1, 3, 5, 10, 20)


def test_checkpoint_events_marks_only_requested_steps():
    events = dict(checkpoint_events(5, checkpoints=(1, 3, 5)))
    assert events == {1: True, 2: False, 3: True, 4: False, 5: True}


def test_invalid_checkpoint_order_is_rejected():
    with pytest.raises(ValueError):
        TrajectoryPlan(checkpoints=(1, 5, 3))
