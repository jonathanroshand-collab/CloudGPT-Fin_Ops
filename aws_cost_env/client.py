from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import AWSAction, AWSObservation, AWSState


class AWSCostEnv(EnvClient[AWSAction, AWSObservation, AWSState]):
    def _step_payload(self, action: AWSAction) -> dict:
        # FastAPI server reconstructs Action based on our model dump
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        # If the environment backend merges done/reward into the root payload
        done = payload.get("done", False)
        reward = payload.get("reward")

        return StepResult(
            observation=AWSObservation(
                done=done,
                reward=reward,
                **obs_data
            ),
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: dict) -> AWSState:
        return AWSState.model_validate(payload)
