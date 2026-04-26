"""Data models for the NeuralTuner environment.

Data models for the NeuralTuner environment.

NeuralTunerAction drives six operations an LLM agent can take:
  profile_layer  — reveal sensitivity and stats for one layer
  quantize_layer — apply a dtype (FP32/FP16/INT8/INT4) to a layer
  prune_layer    — apply structured pruning (LOW=25%/MEDIUM=50%/HIGH=75% channels)
  revert_layer   — reset a layer back to FP32 and NONE pruning
  benchmark      — simulate hardware performance (limited budget)
  submit         — finalize and receive the episode reward

NeuralTunerObservation returns text output suitable for LLM consumption.
NeuralTunerState carries lightweight episode metadata for the client.
"""

from typing import Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class NeuralTunerAction(Action):
    action_type: Literal[
        "profile_layer",
        "quantize_layer",
        "prune_layer",
        "revert_layer",
        "benchmark",
        "submit",
    ] = Field(..., description="Which operation to perform")

    layer_id: Optional[str] = Field(
        default=None,
        description="Target layer ID (required for profile/quantize/prune/revert)",
    )
    dtype: Optional[Literal["FP32", "FP16", "INT8", "INT4"]] = Field(
        default=None,
        description="Target dtype (required for quantize_layer)",
    )
    sparsity: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = Field(
        default=None,
        description="Pruning sparsity level (required for prune_layer): LOW=25%, MEDIUM=50%, HIGH=75%",
    )


class NeuralTunerObservation(Observation):
    output: str = Field(
        default="",
        description="Human-readable text output for the LLM agent",
    )
    success: bool = Field(default=True, description="Whether the action succeeded")
    error: Optional[str] = Field(default=None, description="Error code if action failed")


class NeuralTunerState(State):
    """Extended episode state returned by the /state endpoint."""

    model_id: str = Field(default="", description="Current model being optimized")
    difficulty: str = Field(default="easy", description="Scenario difficulty level")
    submitted: bool = Field(default=False, description="Whether the episode has been submitted")
    benchmark_count: int = Field(default=0, description="Number of benchmarks used so far")
    final_reward: Optional[float] = Field(default=None, description="Final reward after submission")
    episode_id: str = Field(default="", description="Episode ID")
