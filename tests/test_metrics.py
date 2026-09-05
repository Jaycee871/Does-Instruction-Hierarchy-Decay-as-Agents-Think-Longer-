from ih_decay.metrics import trajectory_metrics


def test_stable_compliance_has_zero_decay():
    m = trajectory_metrics([1, 3, 5, 10, 20], [1, 1, 1, 1, 1])
    assert m.absolute_decay == 0.0
    assert m.slope_per_step == 0.0
    assert m.normalized_auc == 1.0
    assert m.first_failure_step is None
    assert m.recovery_rate is None


def test_failure_and_recovery_are_distinguished():
    m = trajectory_metrics([1, 3, 5, 10, 20], [1, 1, 0, 1, 0])
    assert m.first_failure_step == 5
    assert m.recovery_rate == 0.5
    assert m.absolute_decay == 1.0


def test_fractional_group_compliance_is_supported():
    m = trajectory_metrics([1, 3, 5], [0.9, 0.8, 0.7])
    assert round(m.absolute_decay, 6) == 0.2
    assert m.slope_per_step < 0
    assert 0.0 <= m.normalized_auc <= 1.0
