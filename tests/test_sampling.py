from ih_decay.sampling import PilotCandidate, select_stratified


def _c(row, privileged, attack, task):
    return PilotCandidate("toy.jsonl", row, task, attack, privileged)


def test_selection_is_deterministic():
    rows = [_c(i, "system", "user", f"task-{i % 4}") for i in range(20)]
    a = select_stratified(rows, n=8, seed=7)
    b = select_stratified(rows, n=8, seed=7)
    assert [x.example_id for x in a] == [x.example_id for x in b]


def test_authority_pairs_are_balanced_when_capacity_allows():
    rows = []
    row = 0
    for privileged, attack in [("system", "user"), ("system", "developer"), ("developer", "user")]:
        for i in range(12):
            rows.append(_c(row, privileged, attack, f"task-{i % 3}"))
            row += 1
    selected = select_stratified(rows, n=12, seed=11)
    counts = {}
    for item in selected:
        counts[item.authority_pair] = counts.get(item.authority_pair, 0) + 1
    assert sorted(counts.values()) == [4, 4, 4]


def test_task_types_are_covered_before_repeats_within_pair():
    rows = [
        _c(0, "system", "user", "a"),
        _c(1, "system", "user", "a"),
        _c(2, "system", "user", "b"),
        _c(3, "system", "user", "b"),
        _c(4, "system", "user", "c"),
        _c(5, "system", "user", "c"),
    ]
    selected = select_stratified(rows, n=3, seed=5)
    assert {x.task_type for x in selected} == {"a", "b", "c"}
