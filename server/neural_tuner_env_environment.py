# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
NeuralTuner RL Environment.

An LLM agent acts as a hardware optimization engineer, profiling neural
network layers and applying quantization (FP32/FP16/INT8/INT4) to meet
Snapdragon HTP constraints (latency, memory, accuracy).

Episode flow:
  reset()           → observe model + constraints, all sensitivities hidden
  profile_layer()   → reveal sensitivity for one layer
  quantize_layer()  → apply dtype to a layer
  revert_layer()    → undo a quantization decision
  benchmark()       → run hardware simulation (≤ MAX_BENCHMARKS per episode)
  submit()          → finalize; receive multi-component reward
"""

from typing import Dict, Optional, Set
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from models import NeuralTunerAction, NeuralTunerObservation, NeuralTunerState
from server.model_zoo import get_layers, get_metadata
from server.scenarios import Scenario, sample_scenario
from server.simulator import HardwareSimulator


class NeuralTunerEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    MAX_STEPS: int = 20
    MAX_BENCHMARKS: int = 5

    def __init__(self):
        self._state = NeuralTunerState(episode_id=str(uuid4()), step_count=0)
        self._simulator: Optional[HardwareSimulator] = None
        self._scenario: Optional[Scenario] = None
        self._layer_ids: list = []
        self._quantization_map: Dict[str, str] = {}
        self._profiled_layers: Set[str] = set()
        self._benchmark_count: int = 0
        self._submitted: bool = False
        self._final_reward: Optional[float] = None

    # ── public interface ───────────────────────────────────────────────────

    def reset(
        self,
        difficulty: Optional[str] = None,
        model_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> NeuralTunerObservation:
        scenario = sample_scenario(difficulty, model_id, seed)
        layers = get_layers(scenario.model_id)
        metadata = get_metadata(scenario.model_id)

        self._scenario = scenario
        self._simulator = HardwareSimulator(layers, scenario.constraints)
        self._quantization_map = {l.layer_id: "FP32" for l in layers}
        self._layer_ids = [l.layer_id for l in layers]
        self._profiled_layers = set()
        self._benchmark_count = 0
        self._submitted = False
        self._final_reward = None

        self._state = NeuralTunerState(
            episode_id=str(uuid4()),
            step_count=0,
            model_id=scenario.model_id,
            difficulty=scenario.difficulty,
        )

        return self._build_initial_observation(metadata, scenario, layers)

    def step(self, action: NeuralTunerAction) -> NeuralTunerObservation:  # type: ignore[override]
        if self._simulator is None:
            return NeuralTunerObservation(
                output="Environment not initialized. Call reset() before step().",
                success=False,
                error="not_initialized",
                done=False,
                reward=0.0,
            )
        self._state.step_count += 1

        if self._submitted:
            return NeuralTunerObservation(
                output="Episode already submitted. Call reset() to start a new episode.",
                success=False,
                error="episode_complete",
                done=True,
                reward=self._final_reward or 0.0,
            )

        if self._state.step_count > self.MAX_STEPS:
            return NeuralTunerObservation(
                output=f"Step limit ({self.MAX_STEPS}) reached. Episode terminated with zero reward.",
                success=False,
                error="max_steps_exceeded",
                done=True,
                reward=0.0,
            )

        dispatch = {
            "profile_layer": self._handle_profile,
            "quantize_layer": self._handle_quantize,
            "revert_layer": self._handle_revert,
            "benchmark": self._handle_benchmark,
            "submit": self._handle_submit,
        }
        handler = dispatch.get(action.action_type)
        if handler is None:
            return NeuralTunerObservation(
                output=f"Unknown action_type: '{action.action_type}'. " f"Valid: {list(dispatch)}",
                success=False,
                error="invalid_action",
                done=False,
                reward=0.0,
            )
        return handler(action)

    @property
    def state(self) -> NeuralTunerState:
        self._state.benchmark_count = self._benchmark_count
        self._state.submitted = self._submitted
        self._state.final_reward = self._final_reward
        return self._state

    # ── action handlers ────────────────────────────────────────────────────

    def _handle_profile(self, action: NeuralTunerAction) -> NeuralTunerObservation:
        err = self._require_layer(action.layer_id)
        if err:
            return err
        assert action.layer_id is not None
        assert self._simulator is not None

        report = self._simulator.get_profile_report(action.layer_id)
        self._profiled_layers.add(action.layer_id)

        risk = report["sensitivity_risk"]
        advice = {
            "low": "Safe to quantize aggressively (INT4 viable).",
            "medium": "Quantize with caution — prefer INT8 over INT4.",
            "high": "High sensitivity — use FP16 or keep at FP32.",
        }[risk]

        text = (
            f"PROFILE: {action.layer_id}\n"
            f"  Type:        {report['layer_type']}\n"
            f"  Latency:     {report['base_latency_ms']} ms\n"
            f"  Memory:      {report['base_memory_mb']} MB\n"
            f"  Parameters:  {report['param_count']:,}\n"
            f"  Sensitivity: {report['sensitivity']:.3f}  [{risk} risk]\n"
            f"  Current:     {self._quantization_map[action.layer_id]}\n"
            f"  Advice:      {advice}"
        )
        return NeuralTunerObservation(
            output=text,
            success=True,
            done=False,
            reward=0.0,
            metadata=report,
        )

    def _handle_quantize(self, action: NeuralTunerAction) -> NeuralTunerObservation:
        err = self._require_layer(action.layer_id)
        if err:
            return err
        assert action.layer_id is not None
        assert self._simulator is not None
        if not action.dtype:
            return NeuralTunerObservation(
                output="quantize_layer requires dtype (FP32 | FP16 | INT8 | INT4).",
                success=False,
                error="missing_dtype",
                done=False,
                reward=0.0,
            )

        prev = self._quantization_map[action.layer_id]
        self._quantization_map[action.layer_id] = action.dtype

        warning = ""
        if action.layer_id not in self._profiled_layers:
            warning = "\n  WARNING: Layer not profiled. " "Sensitivity is unknown — accuracy impact is unpredictable."

        n_quantized = sum(1 for v in self._quantization_map.values() if v != "FP32")
        text = (
            f"QUANTIZE: {action.layer_id}  {prev} → {action.dtype}"
            f"{warning}\n"
            f"  {n_quantized}/{len(self._quantization_map)} layers quantized. "
            f"Run 'benchmark' to see updated hardware metrics."
        )
        return NeuralTunerObservation(output=text, success=True, done=False, reward=0.0)

    def _handle_revert(self, action: NeuralTunerAction) -> NeuralTunerObservation:
        err = self._require_layer(action.layer_id)
        if err:
            return err
        assert action.layer_id is not None
        assert self._simulator is not None

        prev = self._quantization_map[action.layer_id]
        self._quantization_map[action.layer_id] = "FP32"
        return NeuralTunerObservation(
            output=f"REVERT: {action.layer_id}  {prev} → FP32",
            success=True,
            done=False,
            reward=0.0,
        )

    def _handle_benchmark(self, _action: Optional[NeuralTunerAction] = None) -> NeuralTunerObservation:
        assert self._simulator is not None
        if self._benchmark_count >= self.MAX_BENCHMARKS:
            return NeuralTunerObservation(
                output=(f"Benchmark budget exhausted ({self.MAX_BENCHMARKS} used). " "You must call submit() now."),
                success=False,
                error="benchmark_limit_reached",
                done=False,
                reward=0.0,
            )

        self._benchmark_count += 1
        r = self._simulator.get_benchmark_report(self._quantization_map)
        remaining = self.MAX_BENCHMARKS - self._benchmark_count

        def tick(ok: bool) -> str:
            return "PASS" if ok else "FAIL"

        text = (
            f"BENCHMARK {self._benchmark_count}/{self.MAX_BENCHMARKS}\n"
            f"  Latency:  {r['quantized_latency_ms']:7.2f} ms  "
            f"(budget {r['latency_budget_ms']} ms)  "
            f"[{tick(r['meets_latency_budget'])}]  "
            f"↓{r['latency_improvement_pct']}%\n"
            f"  Memory:   {r['quantized_memory_mb']:7.2f} MB  "
            f"(budget {r['memory_budget_mb']} MB)  "
            f"[{tick(r['memory_fits'])}]\n"
            f"  Accuracy: {r['estimated_accuracy_retention']:.4f}  "
            f"(min {r['min_accuracy_retention']})  "
            f"[{tick(r['accuracy_ok'])}]\n"
            f"  Projected reward: {r['reward']:.4f}\n"
            f"  All constraints met: {'YES' if r['all_constraints_met'] else 'NO'}\n"
            f"  Benchmarks remaining: {remaining}"
        )
        return NeuralTunerObservation(
            output=text,
            success=True,
            done=False,
            reward=0.0,
            metadata=r,
        )

    def _handle_submit(self, _action: Optional[NeuralTunerAction] = None) -> NeuralTunerObservation:
        assert self._simulator is not None
        r = self._simulator.get_benchmark_report(self._quantization_map)
        reward = r["reward"]

        self._final_reward = reward
        self._submitted = True

        text = self._build_result_summary(r, reward)
        return NeuralTunerObservation(
            output=text,
            success=True,
            done=True,
            reward=reward,
            metadata=r,
        )

    # ── observation builders ───────────────────────────────────────────────

    def _build_initial_observation(
        self,
        metadata,
        scenario: Scenario,
        layers: list,
    ) -> NeuralTunerObservation:
        header = f"{'Layer ID':<28} {'Type':<25} {'Latency':>8}  {'Memory':>8}  Sensitivity"
        rows = "\n".join(
            f"  {l.layer_id:<26} {l.layer_type:<25} {l.base_latency_ms:>7.1f}ms  " f"{l.base_memory_mb:>6.1f}MB  HIDDEN"
            for l in layers
        )
        c = scenario.constraints
        text = (
            f"{'=' * 64}\n"
            f"NEURAL TUNER  —  {metadata.name.upper()}\n"
            f"{'=' * 64}\n"
            f"Model:       {metadata.name}  ({metadata.total_params / 1e6:.1f}M params)\n"
            f"Description: {metadata.description}\n"
            f"Difficulty:  {scenario.difficulty.upper()}\n"
            f"\nBASELINE\n"
            f"  Latency: {metadata.base_latency_ms} ms\n"
            f"  Memory:  {metadata.base_memory_mb} MB\n"
            f"\nSNAPDRAGON HTP CONSTRAINTS\n"
            f"  Latency budget:  {c.latency_budget_ms} ms\n"
            f"  Memory budget:   {c.memory_budget_mb} MB\n"
            f"  Min accuracy:    {c.min_accuracy_retention}\n"
            f"\nSCENARIO\n  {scenario.description}\n"
            f"\nLAYERS ({len(layers)} total)\n"
            f"  {header}\n"
            f"  {'-' * 85}\n"
            f"{rows}\n"
            f"\nACTIONS\n"
            f"  profile_layer  (layer_id)               reveal sensitivity & full stats\n"
            f"  quantize_layer (layer_id, dtype)         FP32 | FP16 | INT8 | INT4\n"
            f"  revert_layer   (layer_id)                reset layer to FP32\n"
            f"  benchmark      ()                        simulate hardware  "
            f"[{self.MAX_BENCHMARKS} max]\n"
            f"  submit         ()                        finalize & receive reward\n"
            f"\nMax steps: {self.MAX_STEPS}  |  Benchmarks: {self.MAX_BENCHMARKS}\n"
            f"Tip: Profile layers before quantizing to avoid unseen accuracy drops."
        )
        return NeuralTunerObservation(output=text, success=True, done=False, reward=0.0)

    def _build_result_summary(self, r: dict, reward: float) -> str:
        verdict = "PASS" if r["all_constraints_met"] else "FAIL"
        counts: Dict[str, int] = {}
        for info in r["per_layer_breakdown"].values():
            counts[info["dtype"]] = counts.get(info["dtype"], 0) + 1
        dtype_line = "  ".join(f"{d}: {n}" for d, n in sorted(counts.items()))

        return (
            f"{'=' * 64}\n"
            f"SUBMISSION  —  {verdict}\n"
            f"{'=' * 64}\n"
            f"Final reward: {reward:.4f} / 1.0000\n"
            f"\nPERFORMANCE\n"
            f"  Latency:  {r['base_latency_ms']} ms  →  {r['quantized_latency_ms']} ms  "
            f"({r['latency_improvement_pct']}% ↓)  "
            f"budget {r['latency_budget_ms']} ms  "
            f"{'PASS' if r['meets_latency_budget'] else 'FAIL'}\n"
            f"  Memory:   {r['base_memory_mb']} MB  →  {r['quantized_memory_mb']} MB  "
            f"budget {r['memory_budget_mb']} MB  "
            f"{'PASS' if r['memory_fits'] else 'FAIL'}\n"
            f"  Accuracy: {r['estimated_accuracy_retention']:.4f}  "
            f"min {r['min_accuracy_retention']}  "
            f"{'PASS' if r['accuracy_ok'] else 'FAIL'}\n"
            f"\nQUANTIZATION BREAKDOWN\n  {dtype_line}\n"
            f"\nREWARD BREAKDOWN\n"
            f"  Latency improvement ({r['latency_improvement_pct']}%):  "
            f"max 0.40\n"
            f"  Memory constraint:  "
            f"{'0.30 PASS' if r['memory_fits'] else '0.00 FAIL'}\n"
            f"  Accuracy retention:  "
            f"{'max 0.20 PASS' if r['accuracy_ok'] else '0.00 FAIL'}\n"
            f"  Efficiency bonus:  "
            f"{'0.10 PASS' if r['all_constraints_met'] else '0.00'}\n"
        )

    def _require_layer(self, layer_id: Optional[str]) -> Optional[NeuralTunerObservation]:
        if not layer_id:
            return NeuralTunerObservation(
                output="This action requires layer_id.",
                success=False,
                error="missing_layer_id",
                done=False,
                reward=0.0,
            )
        if layer_id not in self._quantization_map:
            return NeuralTunerObservation(
                output=f"Layer '{layer_id}' not found.\n" f"Available layers: {self._layer_ids}",
                success=False,
                error="layer_not_found",
                done=False,
                reward=0.0,
            )
        return None
