from models import NeuralTunerAction
from server.neural_tuner_env_environment import NeuralTunerEnvironment


def _first_layer_id(obs_text: str) -> str:
    for line in obs_text.splitlines():
        if "HIDDEN" in line and line.strip():
            return line.split()[0]
    raise AssertionError("No layer row found in reset observation.")


def test_benchmark_budget_is_enforced():
    env = NeuralTunerEnvironment()
    env.reset(model_id="resnet50", difficulty="easy", seed=42)

    for _ in range(env.MAX_BENCHMARKS):
        result = env.step(NeuralTunerAction(action_type="benchmark"))
        assert result.success is True
        assert result.done is False

    exhausted = env.step(NeuralTunerAction(action_type="benchmark"))
    assert exhausted.success is False
    assert exhausted.error == "benchmark_limit_reached"


def test_invalid_layer_returns_error():
    env = NeuralTunerEnvironment()
    env.reset(model_id="inception_v3", difficulty="easy", seed=42)

    result = env.step(
        NeuralTunerAction(action_type="quantize_layer", layer_id="not_a_layer", dtype="INT8")
    )
    assert result.success is False
    assert result.error == "layer_not_found"


def test_submit_ends_episode_and_next_step_is_rejected():
    env = NeuralTunerEnvironment()
    env.reset(model_id="mobilenet_v3", difficulty="easy", seed=42)

    submitted = env.step(NeuralTunerAction(action_type="submit"))
    assert submitted.done is True
    assert submitted.success is True

    after = env.step(NeuralTunerAction(action_type="benchmark"))
    assert after.success is False
    assert after.error == "episode_complete"
    assert after.done is True


def test_quantize_then_revert_changes_state_safely():
    env = NeuralTunerEnvironment()
    reset_obs = env.reset(model_id="resnet50", difficulty="easy", seed=123)
    layer_id = _first_layer_id(reset_obs.output)

    quantized = env.step(
        NeuralTunerAction(action_type="quantize_layer", layer_id=layer_id, dtype="INT8")
    )
    assert quantized.success is True

    reverted = env.step(NeuralTunerAction(action_type="revert_layer", layer_id=layer_id))
    assert reverted.success is True
    assert "FP32" in reverted.output
