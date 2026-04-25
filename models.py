"""Data models for the NeuralTuner environment.

Defines the Action, Observation, and State types used by both the server
and client. The agent interacts via NeuralTunerAction, and receives
NeuralTunerObservation after each step.
"""

from typing import Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class NeuralTunerAction(Action):
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
    )


class NeuralTunerObservation(Observation):
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
