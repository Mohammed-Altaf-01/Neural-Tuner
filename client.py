"""NeuralTuner environment client.

NeuralTuner Environment Client."""

from typing import Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import NeuralTunerAction, NeuralTunerObservation, NeuralTunerState


class NeuralTunerEnv(EnvClient[NeuralTunerAction, NeuralTunerObservation, NeuralTunerState]):
    """
    Client for the NeuralTuner Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step LLM-agent rollouts.

    Example:
        >>> with NeuralTunerEnv(base_url='http://localhost:8000') as env:
        ...     result = env.reset()
        ...     print(result.observation.output)
        ...
        ...     action = NeuralTunerAction(action_type="profile_layer", layer_id="fc_classifier")
        ...     result = env.step(action)
        ...     print(result.observation.output)
    """

    def _step_payload(self, action: NeuralTunerAction) -> Dict:
        payload: Dict = {"action_type": action.action_type}
        if action.layer_id is not None:
            payload["layer_id"] = action.layer_id
        if action.dtype is not None:
            payload["dtype"] = action.dtype
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[NeuralTunerObservation]:
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
        return NeuralTunerState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            model_id=payload.get("model_id", ""),
            difficulty=payload.get("difficulty", "easy"),
            submitted=payload.get("submitted", False),
            benchmark_count=payload.get("benchmark_count", 0),
            final_reward=payload.get("final_reward"),
        )
