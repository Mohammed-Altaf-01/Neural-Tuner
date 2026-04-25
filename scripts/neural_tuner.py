from server.neural_tuner_env_environment import NeuralTunerEnvironment
from models import NeuralTunerAction
from typing import Optional


class NeuralTunerOpenEnv:
    """OpenEnv wrapper compatible with TRL environment_factory."""

    scenario_schedule: list[dict] = []
    schedule_idx: int = 0

    def __init__(self):
        self._env = NeuralTunerEnvironment()
        self.reward = 0.0
        self.done = False

    def reset(self, **kwargs) -> str:
        scenario = None
        if kwargs.get("model_id") or kwargs.get("difficulty"):
            scenario = {
                "model_id": kwargs.get("model_id", "inception_v3"),
                "difficulty": kwargs.get("difficulty", "medium"),
            }
        elif self.scenario_schedule:
            scenario = self.scenario_schedule[self.schedule_idx % len(self.scenario_schedule)]
            NeuralTunerOpenEnv.schedule_idx += 1
        else:
            scenario = {"model_id": "inception_v3", "difficulty": "medium"}

        obs = self._env.reset(
            difficulty=scenario["difficulty"],
            model_id=scenario["model_id"],
            seed=kwargs.get("seed", 42),
        )
        self.reward = 0.0
        self.done = False
        return obs.output

    def _step(self, action_type: str, layer_id: Optional[str] = None, dtype: Optional[str] = None, sparsity: Optional[str] = None) -> str:
        result = self._env.step(
            NeuralTunerAction(action_type=action_type, layer_id=layer_id, dtype=dtype, sparsity=sparsity)
        )
        self.reward = float(result.reward)
        self.done = bool(result.done)
        return result.output

    def profile_layer(self, layer_id: str) -> str:
        """Reveal sensitivity and hardware risk for a specific layer.

        Args:
            layer_id: Layer identifier from the environment layer table.

        Returns:
            Text report containing sensitivity score and optimization hints.
        """
        return self._step("profile_layer", layer_id=layer_id)

    def quantize_layer(self, layer_id: str, dtype: str) -> str:
        """Apply a quantization dtype to one layer.

        Args:
            layer_id: Layer identifier from the environment layer table.
            dtype: Quantization target, one of FP32, FP16, INT8, INT4.

        Returns:
            Text summary of the quantization change.
        """
        return self._step("quantize_layer", layer_id=layer_id, dtype=dtype)

    def prune_layer(self, layer_id: str, sparsity: str) -> str:
        """Apply structured pruning to one layer for Snapdragon sparse-acceleration.

        Pruning removes channels/filters, reducing compute and memory. The Snapdragon
        HTP has dedicated hardware for sparse workloads — combine with quantization
        for maximum compression. Profile first to gauge accuracy risk.

        Args:
            layer_id: Layer identifier from the environment layer table.
            sparsity: Pruning level — LOW (25% removed), MEDIUM (50%), or HIGH (75%).

        Returns:
            Text summary of the pruning change and expected impact.
        """
        return self._step("prune_layer", layer_id=layer_id, sparsity=sparsity)

    def revert_layer(self, layer_id: str) -> str:
        """Reset one layer back to FP32 with no pruning.

        Args:
            layer_id: Layer identifier from the environment layer table.

        Returns:
            Text summary confirming the revert action.
        """
        return self._step("revert_layer", layer_id=layer_id)

    def benchmark(self) -> str:
        """Run hardware simulation for the current quantization and pruning plan.

        Returns:
            Benchmark report with latency, memory, accuracy, and projected reward.
        """
        return self._step("benchmark")

    def submit(self) -> str:
        """Finalize the episode and compute the final reward.

        Returns:
            Final submission summary including constraint pass/fail and reward.
        """
        return self._step("submit")



