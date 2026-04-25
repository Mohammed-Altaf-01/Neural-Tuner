from training_utils import split_scenarios


def test_split_scenarios_has_no_overlap():
    rows = [{"id": f"s{i}", "model_id": "inception_v3", "difficulty": "easy"} for i in range(10)]
    train_rows, eval_rows = split_scenarios(rows, train_fraction=0.8, seed=7)

    train_ids = {r["id"] for r in train_rows}
    eval_ids = {r["id"] for r in eval_rows}
    assert train_ids
    assert eval_ids
    assert train_ids.isdisjoint(eval_ids)


def test_split_scenarios_is_deterministic_for_same_seed():
    rows = [{"id": f"s{i}", "model_id": "resnet50", "difficulty": "medium"} for i in range(12)]
    a_train, a_eval = split_scenarios(rows, train_fraction=0.75, seed=123)
    b_train, b_eval = split_scenarios(rows, train_fraction=0.75, seed=123)
    assert [x["id"] for x in a_train] == [x["id"] for x in b_train]
    assert [x["id"] for x in a_eval] == [x["id"] for x in b_eval]
