import pytest

from training_utils import ensure_metric_keys


def test_metric_schema_validation_passes_with_required_keys():
    metrics = {
        "pre_training_random_mean": 0.4,
        "pre_training_random_std": 0.1,
        "training_reward_mean": 0.5,
        "oracle_ceiling": 0.8,
    }
    ensure_metric_keys(
        metrics,
        [
            "pre_training_random_mean",
            "pre_training_random_std",
            "training_reward_mean",
            "oracle_ceiling",
        ],
    )


def test_metric_schema_validation_fails_when_missing():
    with pytest.raises(ValueError):
        ensure_metric_keys({"pre_training_random_mean": 0.4}, ["oracle_ceiling"])
