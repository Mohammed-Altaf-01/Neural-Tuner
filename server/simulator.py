"""Hardware performance simulator for the NeuralTuner environment.

Simulates neural network inference on Qualcomm Snapdragon edge hardware under
different quantization configurations. Latency/memory multipliers are calibrated
to real Snapdragon 8 Gen 3 HTP (Hexagon Tensor Processor) behavior.
"""

from dataclasses import dataclass, field
from typing import Dict, List


# Per-dtype performance multipliers for Snapdragon 8 Gen 3 HTP
DTYPE_CONFIGS: Dict[str, Dict] = {
    "FP32": {
        "latency_factor": 1.00,
        "memory_factor": 1.00,
        "accuracy_penalty_per_sensitivity": 0.0,
    },
    "BF16": {
        "latency_factor": 0.68,
        "memory_factor": 0.50,
        "accuracy_penalty_per_sensitivity": 0.4,
    },
    "FP16": {
        "latency_factor": 0.62,
        "memory_factor": 0.50,
        "accuracy_penalty_per_sensitivity": 0.3,
    },
    "INT8": {
        "latency_factor": 0.42,
        "memory_factor": 0.25,
        "accuracy_penalty_per_sensitivity": 2.0,
    },
    "INT4": {
        "latency_factor": 0.28,
        "memory_factor": 0.125,
        "accuracy_penalty_per_sensitivity": 7.0,
    },
}

VALID_DTYPES = list(DTYPE_CONFIGS.keys())


@dataclass
class LayerProfile:
    """One layer in a neural network model.

    Args:
        layer_id: Unique identifier for this layer.
        layer_type: Architecture type string (Conv2d, Linear, InceptionBlock, etc.).
        gflops: Computational cost in giga-FLOPs.
        params_m: Trainable parameter count in millions.
        baseline_latency_ms: Inference latency in FP32 on target hardware (ms).
        weight_memory_mb: Weight memory footprint in FP32 (MB).
        sensitivity: Quantization sensitivity in [0, 1]. Multiplied by each dtype's
            accuracy_penalty_per_sensitivity to compute accuracy drop percentage.
            Low values (< 0.10) are safe for INT8. High values (> 0.20) need FP16 max.
    """

    layer_id: str
    layer_type: str
    gflops: float
    params_m: float
    baseline_latency_ms: float
    weight_memory_mb: float
    sensitivity: float


@dataclass
class HardwareConstraints:
    """Deployment constraints for an optimization episode.

    Args:
        memory_budget_mb: Maximum allowed weight memory (MB).
        latency_target_ms: Maximum allowed inference latency (ms).
        max_accuracy_drop_pct: Maximum allowed accuracy degradation (percentage points).
        baseline_accuracy: Model accuracy at FP32 (percent).
        hardware_name: Human-readable hardware platform name.
    """

    memory_budget_mb: float
    latency_target_ms: float
    max_accuracy_drop_pct: float
    baseline_accuracy: float
    hardware_name: str = "Snapdragon 8 Gen 3 (Mobile)"


@dataclass
class SimulationResult:
    """Performance metrics after applying a quantization configuration.

    Args:
        total_latency_ms: Total inference latency after optimization.
        total_memory_mb: Total weight memory after optimization.
        accuracy: Model accuracy after optimization.
        accuracy_drop: Accuracy reduction vs FP32 baseline (percentage points).
        latency_improvement_pct: Percent latency reduction vs baseline.
        memory_saved_pct: Percent memory reduction vs baseline.
        meets_memory: Whether memory is within budget.
        meets_latency: Whether latency is within target.
        meets_accuracy: Whether accuracy drop is within threshold.
        layer_details: Per-layer dict with dtype and savings info.
    """

    total_latency_ms: float
    total_memory_mb: float
    accuracy: float
    accuracy_drop: float
    latency_improvement_pct: float
    memory_saved_pct: float
    meets_memory: bool
    meets_latency: bool
    meets_accuracy: bool
    layer_details: Dict[str, dict] = field(default_factory=dict)


class HardwareSimulator:
    """Simulates neural network performance on Snapdragon edge hardware.

    Given layer profiles and hardware constraints, computes the performance
    impact of quantization configs and calculates a multi-component reward.

    Args:
        layers: Layer profiles describing the model architecture.
        constraints: Hardware deployment constraints for this episode.
    """

    def __init__(self, layers: List[LayerProfile], constraints: HardwareConstraints) -> None:
        self.layers: Dict[str, LayerProfile] = {l.layer_id: l for l in layers}
        self.constraints = constraints
        self.baseline_latency: float = sum(l.baseline_latency_ms for l in layers)
        self.baseline_memory: float = sum(l.weight_memory_mb for l in layers)

    def get_profile_report(self, layer_id: str, current_dtype: str) -> str:
        """Generate a human-readable profile report for a specific layer.

        Args:
            layer_id: ID of the layer to profile.
            current_dtype: The layer's current quantization dtype.

        Returns:
            Formatted string with layer metrics, sensitivity, and quantization options.
        """
        if layer_id not in self.layers:
            return f"ERROR: Layer '{layer_id}' not found. Check available layer IDs."

        layer = self.layers[layer_id]
        latency_pct = layer.baseline_latency_ms / self.baseline_latency * 100
        memory_pct = layer.weight_memory_mb / self.baseline_memory * 100

        if layer.sensitivity < 0.05:
            sens_label = "VERY LOW"
        elif layer.sensitivity < 0.10:
            sens_label = "LOW"
        elif layer.sensitivity < 0.18:
            sens_label = "MEDIUM"
        else:
            sens_label = "HIGH"

        lines = [
            f"━━━ LAYER PROFILE: {layer_id} ━━━",
            f"Type: {layer.layer_type}",
            f"GFLOPs: {layer.gflops:.2f}  |  Parameters: {layer.params_m:.2f}M",
            f"Baseline latency: {layer.baseline_latency_ms:.1f}ms "
            f"({latency_pct:.1f}% of total model latency)",
            f"Baseline memory:  {layer.weight_memory_mb:.1f}MB "
            f"({memory_pct:.1f}% of total model weights)",
            f"Current dtype: {current_dtype}",
            f"Quantization sensitivity: {sens_label} ({layer.sensitivity:.2f})",
            "",
            "Quantization options:",
        ]

        for dtype in ["FP16", "INT8", "INT4"]:
            cfg = DTYPE_CONFIGS[dtype]
            opt_lat = layer.baseline_latency_ms * cfg["latency_factor"]
            opt_mem = layer.weight_memory_mb * cfg["memory_factor"]
            acc_drop = layer.sensitivity * cfg["accuracy_penalty_per_sensitivity"]
            save_lat = layer.baseline_latency_ms - opt_lat
            save_mem = layer.weight_memory_mb - opt_mem

            if acc_drop < 0.10:
                risk = "SAFE"
            elif acc_drop < 0.35:
                risk = "LOW RISK"
            elif acc_drop < 0.80:
                risk = "MODERATE RISK"
            else:
                risk = "HIGH RISK"

            hint = ""
            if dtype == "INT8" and layer.sensitivity < 0.15:
                hint = " ← RECOMMENDED"
            elif dtype == "FP16" and layer.sensitivity >= 0.15:
                hint = " ← SAFER CHOICE"

            lines.append(
                f"  {dtype:5s}: lat={opt_lat:5.1f}ms (saves {save_lat:.1f}ms), "
                f"mem={opt_mem:5.1f}MB (saves {save_mem:.1f}MB), "
                f"acc_drop={acc_drop:.3f}%  [{risk}]{hint}"
            )

        return "\n".join(lines)

    def simulate(self, applied_quantization: Dict[str, str]) -> SimulationResult:
        """Compute model performance for a given quantization configuration.

        Args:
            applied_quantization: Mapping of layer_id → dtype. Layers absent from
                this dict remain at FP32.

        Returns:
            SimulationResult with full performance metrics and per-layer breakdown.
        """
        total_latency = 0.0
        total_memory = 0.0
        total_acc_drop = 0.0
        layer_details: Dict[str, dict] = {}

        for lid, layer in self.layers.items():
            dtype = applied_quantization.get(lid, "FP32")
            cfg = DTYPE_CONFIGS[dtype]

            layer_lat = layer.baseline_latency_ms * cfg["latency_factor"]
            layer_mem = layer.weight_memory_mb * cfg["memory_factor"]
            layer_drop = layer.sensitivity * cfg["accuracy_penalty_per_sensitivity"]

            total_latency += layer_lat
            total_memory += layer_mem
            total_acc_drop += layer_drop

            if dtype != "FP32":
                layer_details[lid] = {
                    "dtype": dtype,
                    "latency_ms": round(layer_lat, 2),
                    "memory_mb": round(layer_mem, 2),
                    "latency_saved_ms": round(layer.baseline_latency_ms - layer_lat, 2),
                    "memory_saved_mb": round(layer.weight_memory_mb - layer_mem, 2),
                    "accuracy_drop_pct": round(layer_drop, 3),
                }

        lat_imp_pct = (self.baseline_latency - total_latency) / self.baseline_latency * 100
        mem_saved_pct = (self.baseline_memory - total_memory) / self.baseline_memory * 100

        return SimulationResult(
            total_latency_ms=round(total_latency, 1),
            total_memory_mb=round(total_memory, 1),
            accuracy=round(self.constraints.baseline_accuracy - total_acc_drop, 2),
            accuracy_drop=round(total_acc_drop, 3),
            latency_improvement_pct=round(lat_imp_pct, 1),
            memory_saved_pct=round(mem_saved_pct, 1),
            meets_memory=total_memory <= self.constraints.memory_budget_mb,
            meets_latency=total_latency <= self.constraints.latency_target_ms,
            meets_accuracy=total_acc_drop <= self.constraints.max_accuracy_drop_pct,
            layer_details=layer_details,
        )

    def compute_reward(self, result: SimulationResult, step_count: int) -> float:
        """Compute multi-component RL reward.

        Components:
          - Latency improvement (0–0.40): proportional to % reduction from baseline.
          - Memory constraint  (0 or 0.30): binary — full credit only if within budget.
          - Accuracy retention (0–0.20): full credit within threshold, partial if just over.
          - Efficiency bonus   (0–0.10): fewer steps = small bonus.

        Args:
            result: Simulated performance metrics.
            step_count: Number of steps taken in this episode.

        Returns:
            Reward clipped to [0.05, 0.95].
        """
        latency_reward = min(0.40, max(0.0, result.latency_improvement_pct / 100.0 * 0.65))

        memory_reward = 0.30 if result.meets_memory else 0.0

        threshold = self.constraints.max_accuracy_drop_pct
        if result.meets_accuracy:
            headroom = 1.0 - (result.accuracy_drop / threshold) if threshold > 0 else 1.0
            accuracy_reward = 0.10 + 0.10 * max(0.0, headroom)
        else:
            overshoot = result.accuracy_drop - threshold
            accuracy_reward = max(0.0, 0.08 - overshoot * 0.04)

        efficiency_reward = max(0.0, 0.10 - step_count * 0.006)

        total = latency_reward + memory_reward + accuracy_reward + efficiency_reward
        return round(max(0.05, min(0.95, total)), 3)

    def get_benchmark_report(
        self,
        result: SimulationResult,
        step_count: int,
        max_steps: int,
    ) -> str:
        """Generate a human-readable benchmark report.

        Args:
            result: Simulated performance metrics.
            step_count: Steps used so far.
            max_steps: Maximum allowed steps.

        Returns:
            Formatted benchmark report string.
        """
        c = self.constraints

        def status(ok: bool, detail: str) -> str:
            return f"✓ OK" if ok else f"❌ {detail}"

        mem_ok = status(
            result.meets_memory,
            f"OVER by {result.total_memory_mb - c.memory_budget_mb:.1f}MB",
        )
        lat_ok = status(
            result.meets_latency,
            f"OVER by {result.total_latency_ms - c.latency_target_ms:.1f}ms",
        )
        acc_ok = status(
            result.meets_accuracy,
            f"OVER by {result.accuracy_drop - c.max_accuracy_drop_pct:.3f}%",
        )

        lines = [
            "━━━ BENCHMARK RESULTS ━━━",
            f"{'':22s} {'Current':>10s}  {'Target':>10s}  Status",
            f"{'Latency:':22s} {result.total_latency_ms:>9.1f}ms  {c.latency_target_ms:>9.1f}ms  {lat_ok}",
            f"{'Memory:':22s} {result.total_memory_mb:>9.1f}MB  {c.memory_budget_mb:>9.1f}MB  {mem_ok}",
            f"{'Accuracy drop:':22s} {result.accuracy_drop:>9.3f}%  {c.max_accuracy_drop_pct:>9.1f}%  {acc_ok}",
            "",
        ]

        if result.layer_details:
            lines.append(f"Optimized layers ({len(result.layer_details)}/{len(self.layers)}):")
            for lid, info in result.layer_details.items():
                lines.append(
                    f"  {lid:22s} → {info['dtype']:5s} "
                    f"(−{info['latency_saved_ms']:.1f}ms, "
                    f"−{info['memory_saved_mb']:.1f}MB, "
                    f"−{info['accuracy_drop_pct']:.3f}% acc)"
                )
            remaining = [lid for lid in self.layers if lid not in result.layer_details]
            if remaining:
                lines.append(f"Still at FP32: {', '.join(remaining)}")
        else:
            lines.append("No optimizations applied yet.")

        est_reward = self.compute_reward(result, step_count)
        lines += [
            "",
            f"Estimated reward if submitted now: {est_reward:.3f}",
            f"Steps used: {step_count} / {max_steps}",
        ]

        return "\n".join(lines)
