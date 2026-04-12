# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scaler Env environment client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import ScalerAction, ScalerObservation, ScalerState


class ScalerEnv(EnvClient[ScalerAction, ScalerObservation, ScalerState]):
    """
    Client for the Scaler Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with ScalerEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.subject)
        ...
        ...     result = client.step(
        ...         ScalerAction(operation="classify", category="billing", priority="high")
        ...     )
        ...     print(result.observation.current_stage)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = ScalerEnv.from_docker_image("scaler_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(
        ...         ScalerAction(operation="respond", response="Thanks for the update.")
        ...     )
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: ScalerAction) -> Dict:
        """Convert `ScalerAction` to a JSON payload."""

        if hasattr(action, "model_dump"):
            return action.model_dump(exclude_none=True, mode="json")
        return action.dict(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[ScalerObservation]:
        """
        Parse server response into StepResult[ScalerObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with ScalerObservation
        """
        obs_data = payload.get("observation", payload)
        if isinstance(obs_data, dict) and "task_info" not in obs_data and "metadata" in obs_data:
            obs_data = {**obs_data, "task_info": obs_data.get("metadata", {})}
        if hasattr(ScalerObservation, "model_validate"):
            observation = ScalerObservation.model_validate(obs_data)
        else:  # pragma: no cover
            observation = ScalerObservation.parse_obj(obs_data)
        observation.done = payload.get("done", getattr(observation, "done", False))
        observation.reward = payload.get("reward", getattr(observation, "reward", 0.0))

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> ScalerState:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        if hasattr(ScalerState, "model_validate"):
            return ScalerState.model_validate(payload)
        return ScalerState.parse_obj(payload)
