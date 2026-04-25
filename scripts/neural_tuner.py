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
        self._last_action_signature = None
        self._last_profiled_layer = None
        self._state_revision = 0
        self._last_benchmark_revision = -1
        self._last_benchmark = None
        self._pending_benchmark_delta = 0.0
        self._pending_action_quality = 0.0

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
        self._last_action_signature = None
        self._last_profiled_layer = None
        self._state_revision = 0
        self._last_benchmark_revision = -1
        self._last_benchmark = None
        self._pending_benchmark_delta = 0.0
        self._pending_action_quality = 0.0
        return obs.output

    def _step(self, action_type: str, layer_id: Optional[str] = None, dtype: Optional[str] = None, sparsity: Optional[str] = None) -> str:
        action_signature = (action_type, layer_id, dtype, sparsity)
        prev_action_signature = self._last_action_signature
        if self._last_action_signature == action_signature:
            # Penalize repeatedly issuing the exact same action.
            self._pending_action_quality -= 0.01

        if action_type == "profile_layer":
            if self._last_profiled_layer == layer_id:
                self._pending_action_quality -= 0.005
            else:
                self._pending_action_quality += 0.005
            self._last_profiled_layer = layer_id

        if action_type in {"quantize_layer", "prune_layer", "revert_layer"}:
            self._state_revision += 1
            if layer_id is not None and layer_id == self._last_profiled_layer:
                # Reward profile->decision progression on the same layer.
                self._pending_action_quality += 0.008
            else:
                self._pending_action_quality += 0.002

        result = self._env.step(
            NeuralTunerAction(action_type=action_type, layer_id=layer_id, dtype=dtype, sparsity=sparsity)
        )
        self.reward = float(result.reward)
        self.done = bool(result.done)
        self._last_action_signature = action_signature

        if action_type == "benchmark":
            report = result.metadata or {}
            latency = float(report.get("quantized_latency_ms", 0.0))
            memory = float(report.get("quantized_memory_mb", 0.0))
            accuracy = float(report.get("estimated_accuracy_retention", 0.0))
            current = {"latency": latency, "memory": memory, "accuracy": accuracy}

            if self._last_benchmark is not None:
                prev = self._last_benchmark
                latency_gain = (prev["latency"] - current["latency"]) / max(prev["latency"], 1.0)
                memory_gain = (prev["memory"] - current["memory"]) / max(prev["memory"], 1.0)
                accuracy_term = 0.002 if current["accuracy"] >= prev["accuracy"] else -0.004
                delta_reward = 0.05 * latency_gain + 0.05 * memory_gain + accuracy_term
                if self._state_revision == self._last_benchmark_revision:
                    # Penalize benchmark spam without state changes.
                    delta_reward -= 0.01
            else:
                delta_reward = 0.0

            self._pending_benchmark_delta += max(-0.03, min(0.03, delta_reward))
            self._last_benchmark = current
            self._last_benchmark_revision = self._state_revision

            if prev_action_signature and prev_action_signature[0] in {"quantize_layer", "prune_layer", "revert_layer"}:
                self._pending_action_quality += 0.004

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

    def _consume_reward_components(self) -> dict:
        """Internal helper: return and reset pending shaping components."""
        components = {
            "benchmark_delta_reward": float(self._pending_benchmark_delta),
            "action_quality_reward": float(self._pending_action_quality),
        }
        self._pending_benchmark_delta = 0.0
        self._pending_action_quality = 0.0
        return components



