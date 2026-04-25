# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Hardware simulator for Snapdragon HTP quantization and pruning effects.

Maps dtype + sparsity choices to latency/memory/accuracy trade-offs and computes
multi-component reward signals for RL training.

Quantization and pruning effects stack multiplicatively for latency/memory
and additively for accuracy penalty — matching real HTP behaviour.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from .model_zoo import LayerProfile

PRUNE_CONFIGS: Dict[str, Dict] = {
    "NONE": {
        "latency_factor": 1.00,
        "memory_factor": 1.00,
        "accuracy_penalty_per_sensitivity": 0.0,
    },
    "LOW": {
        "latency_factor": 0.82,
        "memory_factor": 0.75,
        "accuracy_penalty_per_sensitivity": 0.8,
    },
    "MEDIUM": {
        "latency_factor": 0.65,
        "memory_factor": 0.50,
        "accuracy_penalty_per_sensitivity": 2.5,
    },
    "HIGH": {
        "latency_factor": 0.45,
        "memory_factor": 0.25,
        "accuracy_penalty_per_sensitivity": 6.0,
    },
}

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

    def simulate(
        self,
        quantization_map: Dict[str, str],
        prune_map: Optional[Dict[str, str]] = None,
    ) -> SimulationResult:
        if prune_map is None:
            prune_map = {}

        total_latency = 0.0
        total_memory = 0.0
        accuracy_penalty = 0.0
        breakdown: Dict[str, Dict] = {}

        for layer_id, layer in self._layers.items():
            dtype = quantization_map.get(layer_id, "FP32")
            sparsity = prune_map.get(layer_id, "NONE")
            q_cfg = DTYPE_CONFIGS[dtype]
            p_cfg = PRUNE_CONFIGS[sparsity]

            lat = layer.base_latency_ms * q_cfg["latency_factor"] * p_cfg["latency_factor"]
            mem = layer.base_memory_mb * q_cfg["memory_factor"] * p_cfg["memory_factor"]
            penalty = layer.sensitivity * (
                q_cfg["accuracy_penalty_per_sensitivity"] + p_cfg["accuracy_penalty_per_sensitivity"]
            )

            total_latency += lat
            total_memory += mem
            accuracy_penalty += penalty
            breakdown[layer_id] = {
                "dtype": dtype,
                "sparsity": sparsity,
                "latency_ms": round(lat, 3),
                "memory_mb": round(mem, 3),
                "accuracy_penalty": round(penalty, 4),
            }

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
        """Full profile for one layer, including sensitivity and pruning advice."""
        if layer_id not in self._layers:
            return {"error": f"Layer '{layer_id}' not found"}
        layer = self._layers[layer_id]
        sens = layer.sensitivity
        risk = "low" if sens < 0.10 else "medium" if sens < 0.20 else "high"

        if sens < 0.05:
            prune_advice = "Safe to prune HIGH (75%) — very low accuracy risk."
        elif sens < 0.12:
            prune_advice = "LOW–MEDIUM pruning viable — profile impact first."
        elif sens < 0.25:
            prune_advice = "LOW pruning only — medium sensitivity layer."
        else:
            prune_advice = "Avoid pruning — high accuracy risk."

        return {
            "layer_id": layer_id,
            "layer_type": layer.layer_type,
            "base_latency_ms": layer.base_latency_ms,
            "base_memory_mb": layer.base_memory_mb,
            "sensitivity": sens,
            "param_count": layer.param_count,
            "sensitivity_risk": risk,
            "prune_advice": prune_advice,
        }

    def get_benchmark_report(
        self,
        quantization_map: Dict[str, str],
        prune_map: Optional[Dict[str, str]] = None,
    ) -> Dict:
        """Run simulation and return full benchmark report with reward."""
        result = self.simulate(quantization_map, prune_map)
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
