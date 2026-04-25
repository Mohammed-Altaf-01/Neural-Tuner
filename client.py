"""NeuralTuner environment client.

Provides a typed client for interacting with the NeuralTuner server over
WebSocket. Used in training notebooks and evaluation scripts.
"""

from typing import Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import NeuralTunerAction, NeuralTunerObservation, NeuralTunerState


class NeuralTunerEnv(EnvClient[NeuralTunerAction, NeuralTunerObservation, NeuralTunerState]):
    """Client for the NeuralTuner environment server.

    Maintains a persistent WebSocket connection, with each instance getting
    its own isolated environment session on the server.

    Example — sync usage:
        >>> with NeuralTunerEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset(difficulty="easy")
        ...     print(result.observation.output)
        ...
        ...     action = NeuralTunerAction(action_type="profile_layer", layer_id="conv_stem")
        ...     result = env.step(action)
        ...     print(result.observation.output)

    Example — async usage:
        >>> async with NeuralTunerEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset(difficulty="medium")
        ...     action = NeuralTunerAction(
        ...         action_type="apply_quantization",
        ...         layer_id="inception_e1",
        ...         dtype="INT8",
        ...     )
        ...     result = await env.step(action)
    """

    def _step_payload(self, action: NeuralTunerAction) -> Dict:
        """Convert NeuralTunerAction to JSON payload for the WebSocket step message.

        Args:
            action: The action to send to the environment server.

        Returns:
            Dictionary representation of the action for JSON serialisation.
        """
        payload: Dict = {"action_type": action.action_type}
        if action.layer_id is not None:
            payload["layer_id"] = action.layer_id
        if action.dtype is not None:
            payload["dtype"] = action.dtype
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[NeuralTunerObservation]:
        """Parse server response into a typed StepResult.

        Args:
            payload: Raw JSON response from the server.

        Returns:
            StepResult containing a NeuralTunerObservation.
        """
        obs_data = payload.get("observation", {})
        observation = NeuralTunerObservation(
            output=obs_data.get("output", ""),
            success=obs_data.get("success", True),
            error=obs_data.get("error"),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> NeuralTunerState:
        """Parse server response into a typed NeuralTunerState.

        Args:
            payload: Raw JSON response from the server state endpoint.

        Returns:
            NeuralTunerState with episode metadata.
        """
        return NeuralTunerState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            model_id=payload.get("model_id", ""),
            difficulty=payload.get("difficulty", "easy"),
            submitted=payload.get("submitted", False),
            benchmark_count=payload.get("benchmark_count", 0),
            final_reward=payload.get("final_reward", 0.0),
        )
