from server.model_zoo import get_layers
from server.scenarios import sample_scenario
from server.simulator import HardwareSimulator


def test_reward_prefers_int8_over_fp32_for_easy_constraints():
    scenario = sample_scenario(model_id="inception_v3", difficulty="easy", seed=1)
    layers = get_layers("inception_v3")
    sim = HardwareSimulator(layers, scenario.constraints)

    all_fp32 = {layer.layer_id: "FP32" for layer in layers}
    all_int8 = {layer.layer_id: "INT8" for layer in layers}

    fp32_reward = sim.get_benchmark_report(all_fp32)["reward"]
    int8_reward = sim.get_benchmark_report(all_int8)["reward"]
    assert int8_reward > fp32_reward


def test_reward_penalizes_over_aggressive_int4_on_hard_case():
    scenario = sample_scenario(model_id="bmw_drive_net", difficulty="hard", seed=1)
    layers = get_layers("bmw_drive_net")
    sim = HardwareSimulator(layers, scenario.constraints)

    safe_mix = {}
    for layer in layers:
        if layer.sensitivity < 0.12:
            safe_mix[layer.layer_id] = "INT4"
        elif layer.sensitivity < 0.22:
            safe_mix[layer.layer_id] = "INT8"
        elif layer.sensitivity < 0.28:
            safe_mix[layer.layer_id] = "FP16"
        else:
            safe_mix[layer.layer_id] = "FP32"

    all_int4 = {layer.layer_id: "INT4" for layer in layers}

    safe_reward = sim.get_benchmark_report(safe_mix)["reward"]
    all_int4_reward = sim.get_benchmark_report(all_int4)["reward"]
    assert safe_reward > all_int4_reward
