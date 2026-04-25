# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Hardware simulator for Snapdragon HTP quantization effects.

Maps dtype choices to latency/memory/accuracy trade-offs and computes
multi-component reward signals for RL training.
"""

from dataclasses import dataclass
from typing import Dict, List

from .model_zoo import LayerProfile


DTYPE_CONFIGS: Dict[str, Dict] = {
    "FP32": {
        "latency_factor": 1.00,
        "memory_factor": 1.00,
        "accuracy_penalty_per_sensitivity": 0.0,
    },
    "FP16": {
        "latency_factor": 0.62,
        "memory_factor": 0.50,
        "accuracy_penalty_per_sensitivity": 0.30,
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


@dataclass
class HardwareConstraints:
    latency_budget_ms: float
    memory_budget_mb: float
    min_accuracy_retention: float  # 0.0–1.0


@dataclass
class SimulationResult:
    quantized_latency_ms: float
    quantized_memory_mb: float
    estimated_accuracy_retention: float
    latency_improvement: float         # fraction saved vs baseline
    memory_fits: bool
    accuracy_ok: bool
    meets_latency: bool
    per_layer_breakdown: Dict[str, Dict]


class HardwareSimulator:
    def __init__(self, layers: List[LayerProfile], constraints: HardwareConstraints):
        self._layers = {l.layer_id: l for l in layers}
        self._constraints = constraints
        self._base_latency_ms = sum(l.base_latency_ms for l in layers)
        self._base_memory_mb = sum(l.base_memory_mb for l in layers)

    # ── core simulation ────────────────────────────────────────────────────

    def simulate(self, quantization_map: Dict[str, str]) -> SimulationResult:
        total_latency = 0.0
        total_memory = 0.0
        accuracy_penalty = 0.0
        breakdown: Dict[str, Dict] = {}

        for layer_id, layer in self._layers.items():
            dtype = quantization_map.get(layer_id, "FP32")
            cfg = DTYPE_CONFIGS[dtype]
            lat = layer.base_latency_ms * cfg["latency_factor"]
            mem = layer.base_memory_mb * cfg["memory_factor"]
            penalty = layer.sensitivity * cfg["accuracy_penalty_per_sensitivity"]

            total_latency += lat
            total_memory += mem
            accuracy_penalty += penalty
            breakdown[layer_id] = {
                "dtype": dtype,
                "latency_ms": round(lat, 3),
                "memory_mb": round(mem, 3),
                "accuracy_penalty": round(penalty, 4),
            }

        # accuracy_retention: linearly degraded by total penalty (clamped to [0, 1])
        accuracy_retention = max(0.0, min(1.0, 1.0 - accuracy_penalty / 100.0))
        latency_improvement = (self._base_latency_ms - total_latency) / self._base_latency_ms

        return SimulationResult(
            quantized_latency_ms=round(total_latency, 2),
            quantized_memory_mb=round(total_memory, 2),
            estimated_accuracy_retention=round(accuracy_retention, 4),
            latency_improvement=round(latency_improvement, 4),
            memory_fits=total_memory <= self._constraints.memory_budget_mb,
            accuracy_ok=accuracy_retention >= self._constraints.min_accuracy_retention,
            meets_latency=total_latency <= self._constraints.latency_budget_ms,
            per_layer_breakdown=breakdown,
        )

    def compute_reward(self, result: SimulationResult) -> float:
        """
        Multi-component reward:
          latency improvement  → 0.00–0.40  (continuous, proportional to % saved)
          memory constraint    → 0.00 or 0.30  (binary: fits or not)
          accuracy retention   → 0.00–0.20  (continuous within acceptable range)
          efficiency bonus     → 0.00 or 0.10  (all three constraints met)
        """
        latency_reward = min(result.latency_improvement, 1.0) * 0.40

        memory_reward = 0.30 if result.memory_fits else 0.0

        if result.accuracy_ok:
            span = 1.0 - self._constraints.min_accuracy_retention + 1e-8
            score = (result.estimated_accuracy_retention - self._constraints.min_accuracy_retention) / span
            accuracy_reward = min(score, 1.0) * 0.20
        else:
            accuracy_reward = 0.0

        efficiency_bonus = (
            0.10 if (result.meets_latency and result.memory_fits and result.accuracy_ok) else 0.0
        )

        return round(min(latency_reward + memory_reward + accuracy_reward + efficiency_bonus, 1.0), 4)

    # ── reporting helpers ──────────────────────────────────────────────────

    def get_profile_report(self, layer_id: str) -> Dict:
        """Full profile for one layer, including sensitivity (revealed on profile action)."""
        if layer_id not in self._layers:
            return {"error": f"Layer '{layer_id}' not found"}
        layer = self._layers[layer_id]
        risk = "low" if layer.sensitivity < 0.10 else "medium" if layer.sensitivity < 0.20 else "high"
        return {
            "layer_id": layer_id,
            "layer_type": layer.layer_type,
            "base_latency_ms": layer.base_latency_ms,
            "base_memory_mb": layer.base_memory_mb,
            "sensitivity": layer.sensitivity,
            "param_count": layer.param_count,
            "sensitivity_risk": risk,
        }

    def get_benchmark_report(self, quantization_map: Dict[str, str]) -> Dict:
        """Run simulation and return full benchmark report with reward."""
        result = self.simulate(quantization_map)
        reward = self.compute_reward(result)
        return {
            "quantized_latency_ms": result.quantized_latency_ms,
            "base_latency_ms": round(self._base_latency_ms, 2),
            "latency_budget_ms": self._constraints.latency_budget_ms,
            "latency_improvement_pct": round(result.latency_improvement * 100, 1),
            "meets_latency_budget": result.meets_latency,
            "quantized_memory_mb": result.quantized_memory_mb,
            "base_memory_mb": round(self._base_memory_mb, 2),
            "memory_budget_mb": self._constraints.memory_budget_mb,
            "memory_fits": result.memory_fits,
            "estimated_accuracy_retention": result.estimated_accuracy_retention,
            "min_accuracy_retention": self._constraints.min_accuracy_retention,
            "accuracy_ok": result.accuracy_ok,
            "reward": reward,
            "all_constraints_met": result.meets_latency and result.memory_fits and result.accuracy_ok,
            "per_layer_breakdown": result.per_layer_breakdown,
        }

    @property
    def base_latency_ms(self) -> float:
        return self._base_latency_ms

    @property
    def base_memory_mb(self) -> float:
        return self._base_memory_mb
