"""Data models for the NeuralTuner environment.

<<<<<<< HEAD
Defines the Action, Observation, and State types used by both the server
and client. The agent interacts via NeuralTunerAction, and receives
NeuralTunerObservation after each step.
"""

from typing import Optional
=======
"""
Data models for the NeuralTuner environment.

NeuralTunerAction drives five operations an LLM agent can take:
  profile_layer  — reveal sensitivity and stats for one layer
  quantize_layer — apply a dtype (FP32/FP16/INT8/INT4) to a layer
  revert_layer   — reset a layer back to FP32
  benchmark      — simulate hardware performance (limited budget)
  submit         — finalize and receive the episode reward

NeuralTunerObservation returns text output suitable for LLM consumption.
NeuralTunerState carries lightweight episode metadata for the client.
"""

from typing import Literal, Optional
>>>>>>> master

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class NeuralTunerAction(Action):
<<<<<<< HEAD
    """An action taken by the agent in the NeuralTuner environment.

    The agent must set action_type to one of the five supported values.
    Additional fields are required depending on the action type:
      - profile_layer:      layer_id required
      - apply_quantization: layer_id + dtype required
      - revert_layer:       layer_id required
      - benchmark:          no extra fields
      - submit:             no extra fields

    Args:
        action_type: The type of action to execute.
        layer_id: Target layer ID (for profile/quantize/revert actions).
        dtype: Quantization dtype — one of FP16, INT8, INT4 (for apply_quantization).
    """

    action_type: str = Field(
        ...,
        description="One of: profile_layer, apply_quantization, revert_layer, benchmark, submit",
    )
    layer_id: Optional[str] = Field(
        None,
        description="Target layer ID (required for profile_layer, apply_quantization, revert_layer)",
    )
    dtype: Optional[str] = Field(
        None,
        description="Quantization dtype: FP16, INT8, or INT4 (required for apply_quantization)",
=======
    action_type: Literal[
        "profile_layer",
        "quantize_layer",
        "revert_layer",
        "benchmark",
        "submit",
    ] = Field(..., description="Which operation to perform")

    layer_id: Optional[str] = Field(
        default=None,
        description="Target layer ID (required for profile/quantize/revert)",
    )
    dtype: Optional[Literal["FP32", "FP16", "INT8", "INT4"]] = Field(
        default=None,
        description="Target dtype (required for quantize_layer)",
>>>>>>> master
    )


class NeuralTunerObservation(Observation):
<<<<<<< HEAD
    """Observation returned by the NeuralTuner environment after each step.

    Args:
        output: Human-readable result of the action (profile report, benchmark
            results, status message, or initial observation on reset).
        success: Whether the action executed successfully.
        error: Error message if the action failed, None on success.
    """

    output: str = Field(default="", description="Human-readable action result")
    success: bool = Field(default=True, description="Whether the action succeeded")
    error: Optional[str] = Field(None, description="Error message if the action failed")


class NeuralTunerState(State):
    """Episode state for the NeuralTuner environment.

    Args:
        model_id: The model being optimized in this episode.
        difficulty: Episode difficulty level ('easy', 'medium', 'hard').
        submitted: Whether the agent has called submit to end the episode.
        benchmark_count: Number of benchmark calls used so far.
        final_reward: Reward earned when the episode ended (0.0 until submitted).
    """

    model_id: str = Field(default="", description="Model being optimized")
    difficulty: str = Field(default="easy", description="Episode difficulty")
    submitted: bool = Field(default=False, description="Whether episode has ended")
    benchmark_count: int = Field(default=0, description="Benchmark calls used")
    final_reward: float = Field(default=0.0, description="Reward on episode end")
=======
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
>>>>>>> master
