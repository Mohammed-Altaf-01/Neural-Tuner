"""NeuralTuner Environment — main environment implementation.

The agent acts as a hardware optimization engineer: it receives a neural network
model and hardware deployment constraints, then uses profiling and quantization
tools to reduce latency and memory while preserving accuracy.

Episode lifecycle:
  reset()                  → agent receives model overview + hardware constraints
  step(profile_layer)      → reveals sensitivity score for a specific layer
  step(apply_quantization) → quantizes a layer to FP16, INT8, or INT4
  step(revert_layer)       → reverts a layer back to FP32
  step(benchmark)          → shows current latency/memory/accuracy vs targets
  step(submit)             → ends episode, computes final reward
"""

from typing import Any, Dict, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import NeuralTunerAction, NeuralTunerObservation, NeuralTunerState
    from .model_zoo import get_layers, get_metadata
    from .scenarios import Scenario, sample_scenario
    from .simulator import DTYPE_CONFIGS, VALID_DTYPES, HardwareSimulator
except ImportError:
    from models import NeuralTunerAction, NeuralTunerObservation, NeuralTunerState
    from server.model_zoo import get_layers, get_metadata
    from server.scenarios import Scenario, sample_scenario
    from server.simulator import DTYPE_CONFIGS, VALID_DTYPES, HardwareSimulator


MAX_STEPS = 20
MAX_BENCHMARKS = 5


class NeuralTunerEnvironment(Environment):
    """RL environment for training LLMs to optimize neural networks for edge hardware.

    Each episode presents the agent with a neural network model and hardware
    deployment constraints (memory budget, latency target, accuracy threshold).
    The agent must use profiling tools to investigate layer sensitivity and apply
    quantization decisions to meet the hardware targets.

    Sensitivity scores are hidden until profiled, forcing the agent to learn
    to investigate large layers before quantizing them.

    Attributes:
        SUPPORTS_CONCURRENT_SESSIONS: Each WebSocket client gets its own instance.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        """Initialize the NeuralTuner environment."""
        self._state = NeuralTunerState(episode_id=str(uuid4()), step_count=0)
        self._scenario: Optional[Scenario] = None
        self._simulator: Optional[HardwareSimulator] = None
        self._applied_quantization: Dict[str, str] = {}
        self._profiled_layers: set = set()

    def reset(
        self,
        difficulty: str = "easy",
        model_id: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> NeuralTunerObservation:
        """Reset the environment and start a new episode.

        Args:
            difficulty: Episode difficulty — 'easy', 'medium', 'hard', or 'all'.
            model_id: If provided, locks the episode to this specific model.
            seed: Optional random seed for reproducible scenario sampling.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Initial observation with model overview and hardware constraints.
        """
        self._scenario = sample_scenario(difficulty=difficulty, model_id=model_id, seed=seed)
        layers = get_layers(self._scenario.model_id)
        self._simulator = HardwareSimulator(layers, self._scenario.constraints)
        self._applied_quantization = {}
        self._profiled_layers = set()

        self._state = NeuralTunerState(
            episode_id=str(uuid4()),
            step_count=0,
            model_id=self._scenario.model_id,
            difficulty=self._scenario.difficulty,
        )

        return NeuralTunerObservation(
            output=self._build_initial_observation(),
            success=True,
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: NeuralTunerAction,
        **kwargs: Any,
    ) -> NeuralTunerObservation:
        """Execute one action in the environment.

        Args:
            action: The agent's action with a valid action_type.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Observation reflecting the result of the action.
        """
        if self._scenario is None or self._simulator is None:
            return NeuralTunerObservation(
                output="",
                error="Environment not initialised. Call reset() first.",
                success=False,
                done=True,
                reward=0.0,
            )

        if self._state.submitted:
            return NeuralTunerObservation(
                output="Episode already finished. Call reset() to start a new one.",
                success=False,
                done=True,
                reward=self._state.final_reward,
            )

        self._state.step_count += 1

        if self._state.step_count > MAX_STEPS:
            self._state.submitted = True
            result = self._simulator.simulate(self._applied_quantization)
            reward = self._simulator.compute_reward(result, self._state.step_count)
            self._state.final_reward = reward
            return NeuralTunerObservation(
                output=(
                    f"Step limit ({MAX_STEPS}) reached. Auto-submitting current config.\n\n"
                    + self._simulator.get_benchmark_report(result, self._state.step_count, MAX_STEPS)
                ),
                success=True,
                done=True,
                reward=reward,
            )

        atype = action.action_type

        if atype == "profile_layer":
            return self._handle_profile(action)
        elif atype == "apply_quantization":
            return self._handle_quantize(action)
        elif atype == "revert_layer":
            return self._handle_revert(action)
        elif atype == "benchmark":
            return self._handle_benchmark()
        elif atype == "submit":
            return self._handle_submit()
        else:
            return NeuralTunerObservation(
                output="",
                error=(
                    f"Unknown action_type '{atype}'. "
                    "Valid: profile_layer, apply_quantization, revert_layer, benchmark, submit"
                ),
                success=False,
                done=False,
                reward=-0.02,
            )

    # ── Action handlers ────────────────────────────────────────────────────

    def _handle_profile(self, action: NeuralTunerAction) -> NeuralTunerObservation:
        """Reveal sensitivity and quantization tradeoffs for one layer.

        Args:
            action: Action with layer_id set.

        Returns:
            Observation with the layer profile report.
        """
        if not action.layer_id:
            return NeuralTunerObservation(
                output="", error="profile_layer requires layer_id.",
                success=False, done=False, reward=-0.02,
            )

        current_dtype = self._applied_quantization.get(action.layer_id, "FP32")
        report = self._simulator.get_profile_report(action.layer_id, current_dtype)

        if "ERROR" not in report:
            self._profiled_layers.add(action.layer_id)

        steps_left = MAX_STEPS - self._state.step_count
        return NeuralTunerObservation(
            output=report + f"\n\nSteps remaining: {steps_left} / {MAX_STEPS}",
            success="ERROR" not in report,
            done=False,
            reward=0.0,
        )

    def _handle_quantize(self, action: NeuralTunerAction) -> NeuralTunerObservation:
        """Apply a quantization dtype to a layer.

        Args:
            action: Action with layer_id and dtype set.

        Returns:
            Observation confirming the change or reporting an error.
        """
        if not action.layer_id:
            return NeuralTunerObservation(
                output="", error="apply_quantization requires layer_id.",
                success=False, done=False, reward=-0.02,
            )

        if action.layer_id not in self._simulator.layers:
            return NeuralTunerObservation(
                output="",
                error=f"Layer '{action.layer_id}' not found.",
                success=False, done=False, reward=-0.02,
            )

        dtype = (action.dtype or "").upper()
        if dtype not in VALID_DTYPES or dtype == "FP32":
            return NeuralTunerObservation(
                output="",
                error=f"Invalid dtype '{action.dtype}'. Use: FP16, INT8, INT4.",
                success=False, done=False, reward=-0.02,
            )

        prev_dtype = self._applied_quantization.get(action.layer_id, "FP32")
        self._applied_quantization[action.layer_id] = dtype

        layer = self._simulator.layers[action.layer_id]
        cfg = DTYPE_CONFIGS[dtype]
        new_lat = layer.baseline_latency_ms * cfg["latency_factor"]
        new_mem = layer.weight_memory_mb * cfg["memory_factor"]
        acc_drop = layer.sensitivity * cfg["accuracy_penalty_per_sensitivity"]

        warn = ""
        if action.layer_id not in self._profiled_layers and layer.sensitivity > 0.15:
            warn = (
                "\n⚠️  WARNING: This layer has HIGH sensitivity but was not profiled first. "
                "Risk of accuracy degradation."
            )

        steps_left = MAX_STEPS - self._state.step_count
        return NeuralTunerObservation(
            output=(
                f"✓ Applied {dtype} to '{action.layer_id}' (was {prev_dtype})\n"
                f"  Latency: {layer.baseline_latency_ms:.1f}ms → {new_lat:.1f}ms "
                f"(−{layer.baseline_latency_ms - new_lat:.1f}ms)\n"
                f"  Memory:  {layer.weight_memory_mb:.1f}MB → {new_mem:.1f}MB "
                f"(−{layer.weight_memory_mb - new_mem:.1f}MB)\n"
                f"  Accuracy impact: −{acc_drop:.3f}%{warn}\n\n"
                f"Steps remaining: {steps_left} / {MAX_STEPS}"
            ),
            success=True,
            done=False,
            reward=0.0,
        )

    def _handle_revert(self, action: NeuralTunerAction) -> NeuralTunerObservation:
        """Revert a layer back to FP32.

        Args:
            action: Action with layer_id set.

        Returns:
            Observation confirming the revert.
        """
        if not action.layer_id:
            return NeuralTunerObservation(
                output="", error="revert_layer requires layer_id.",
                success=False, done=False, reward=-0.02,
            )

        if action.layer_id not in self._simulator.layers:
            return NeuralTunerObservation(
                output="",
                error=f"Layer '{action.layer_id}' not found.",
                success=False, done=False, reward=-0.02,
            )

        prev = self._applied_quantization.pop(action.layer_id, "FP32")
        steps_left = MAX_STEPS - self._state.step_count
        return NeuralTunerObservation(
            output=(
                f"✓ Reverted '{action.layer_id}' from {prev} → FP32\n"
                f"Steps remaining: {steps_left} / {MAX_STEPS}"
            ),
            success=True,
            done=False,
            reward=0.0,
        )

    def _handle_benchmark(self) -> NeuralTunerObservation:
        """Run a full model benchmark and show performance vs targets.

        Returns:
            Observation with current latency, memory, accuracy, and estimated reward.
        """
        self._state.benchmark_count += 1
        result = self._simulator.simulate(self._applied_quantization)
        report = self._simulator.get_benchmark_report(
            result, self._state.step_count, MAX_STEPS
        )

        extra = ""
        if self._state.benchmark_count > MAX_BENCHMARKS:
            extra = (
                f"\n⚠️  You have used {self._state.benchmark_count} benchmark calls "
                f"(recommended max: {MAX_BENCHMARKS}). Excessive benchmarking reduces efficiency score."
            )

        return NeuralTunerObservation(
            output=report + extra,
            success=True,
            done=False,
            reward=0.0,
        )

    def _handle_submit(self) -> NeuralTunerObservation:
        """End the episode and compute the final reward.

        Returns:
            Final observation with reward breakdown and performance summary.
        """
        self._state.submitted = True
        result = self._simulator.simulate(self._applied_quantization)
        reward = self._simulator.compute_reward(result, self._state.step_count)
        self._state.final_reward = reward

        report = self._simulator.get_benchmark_report(
            result, self._state.step_count, MAX_STEPS
        )
        summary = self._build_result_summary(result, reward)

        return NeuralTunerObservation(
            output=report + "\n\n" + summary,
            success=True,
            done=True,
            reward=reward,
        )

    # ── Observation builders ───────────────────────────────────────────────

    def _build_initial_observation(self) -> str:
        """Build the initial observation shown to the agent after reset.

        Returns:
            Formatted string with model overview, constraints, and action reference.
        """
        sc = self._scenario
        meta = get_metadata(sc.model_id)
        c = sc.constraints
        sim = self._simulator

        mem_diff = sim.baseline_memory - c.memory_budget_mb
        lat_diff = sim.baseline_latency - c.latency_target_ms

        mem_note = (
            f"{sim.baseline_memory:.1f}MB — {mem_diff:.1f}MB over budget ❌"
            if mem_diff > 0
            else f"{sim.baseline_memory:.1f}MB — already within budget ✓"
        )
        lat_note = (
            f"{sim.baseline_latency:.1f}ms — {lat_diff:.1f}ms over target ❌"
            if lat_diff > 0
            else f"{sim.baseline_latency:.1f}ms — already within target ✓"
        )

        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║              NeuralTuner — Edge AI Optimizer                ║",
            "╚══════════════════════════════════════════════════════════════╝",
            "",
            f"MISSION: {sc.description}",
            "",
            "━━━ HARDWARE CONSTRAINTS ━━━",
            f"  Memory budget:     {c.memory_budget_mb:.1f}MB  ({mem_note})",
            f"  Latency target:    {c.latency_target_ms:.1f}ms  ({lat_note})",
            f"  Max accuracy drop: {c.max_accuracy_drop_pct:.1f}%  "
            f"(baseline: {c.baseline_accuracy:.1f}%)",
            f"  Hardware:          {c.hardware_name}",
            "",
            f"━━━ MODEL: {meta['display_name']} ━━━",
            f"Task: {meta['task']}",
            f"Domain: {meta['domain']}  |  Parameters: {meta['params_m']:.1f}M",
            "",
            "Layers (sensitivity hidden — use profile_layer to reveal before quantizing):",
            f"  {'Layer ID':<22s} {'Type':<16s} {'Latency':>8s}  {'Memory':>8s}",
            "  " + "─" * 62,
        ]

        for lid, layer in sim.layers.items():
            lines.append(
                f"  {lid:<22s} {layer.layer_type:<16s} "
                f"{layer.baseline_latency_ms:>6.1f}ms  {layer.weight_memory_mb:>6.1f}MB"
            )

        lines += [
            "",
            "━━━ AVAILABLE ACTIONS ━━━",
            '  profile_layer      → {"action_type": "profile_layer", "layer_id": "<id>"}',
            '  apply_quantization → {"action_type": "apply_quantization", "layer_id": "<id>", "dtype": "FP16|INT8|INT4"}',
            '  revert_layer       → {"action_type": "revert_layer", "layer_id": "<id>"}',
            '  benchmark          → {"action_type": "benchmark"}',
            '  submit             → {"action_type": "submit"}',
            "",
            f"Steps: 0 / {MAX_STEPS}",
        ]

        return "\n".join(lines)

    def _build_result_summary(self, result, reward: float) -> str:
        """Build a concise final result summary.

        Args:
            result: SimulationResult from the submitted configuration.
            reward: Final reward value.

        Returns:
            Formatted result summary string.
        """
        if reward >= 0.75:
            verdict = "EXCELLENT — all constraints met with room to spare"
        elif reward >= 0.55:
            verdict = "GOOD — most constraints met"
        elif reward >= 0.35:
            verdict = "PARTIAL — some constraints missed"
        else:
            verdict = "POOR — constraints significantly violated"

        all_met = result.meets_memory and result.meets_latency and result.meets_accuracy

        return "\n".join([
            "━━━ EPISODE COMPLETE ━━━",
            f"Final reward:       {reward:.3f}  ({verdict})",
            f"Constraints met:    {'ALL ✓' if all_met else 'SOME MISSED ✗'}",
            f"Latency saved:      {result.latency_improvement_pct:.1f}%",
            f"Memory saved:       {result.memory_saved_pct:.1f}%",
            f"Accuracy retained:  {result.accuracy:.2f}%  (drop: {result.accuracy_drop:.3f}%)",
            f"Layers optimized:   {len(result.layer_details)} / {len(self._simulator.layers)}",
            f"Steps used:         {self._state.step_count} / {MAX_STEPS}",
        ])

    @property
    def state(self) -> NeuralTunerState:
        """Return the current episode state.

        Returns:
            NeuralTunerState with episode metadata and progress tracking.
        """
        return self._state
