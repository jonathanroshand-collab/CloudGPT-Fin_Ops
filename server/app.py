from __future__ import annotations

from threading import Lock

import uvicorn
from fastapi import Body, HTTPException
from openenv.core.env_server import create_fastapi_app
from openenv.core.env_server.serialization import deserialize_action, serialize_observation
from openenv.core.env_server.types import ResetRequest, ResetResponse, StepRequest, StepResponse
from pydantic import ValidationError

from aws_cost_env.models import AWSAction, AWSObservation, AWSState
from server.environment import AWSCostEnvironment

app = create_fastapi_app(AWSCostEnvironment, AWSAction, AWSObservation)

_http_env_lock = Lock()
_http_env: AWSCostEnvironment | None = None


def _get_http_env() -> AWSCostEnvironment:
    global _http_env
    with _http_env_lock:
        if _http_env is None:
            _http_env = AWSCostEnvironment()
        return _http_env


def _reset_http_env() -> AWSCostEnvironment:
    global _http_env
    with _http_env_lock:
        if _http_env is not None:
            _http_env.close()
        _http_env = AWSCostEnvironment()
        return _http_env


# The stock OpenEnv HTTP routes are stateless. Override them so plain HTTP
# callers can use reset -> step -> state against one shared environment.
app.router.routes = [
    route
    for route in app.router.routes
    if getattr(route, "path", None) not in {"/reset", "/step", "/state"}
]


@app.post("/reset", response_model=ResetResponse, tags=["Environment Control"])
async def reset(request: ResetRequest = Body(default_factory=ResetRequest)) -> ResetResponse:
    env = _reset_http_env()
    kwargs = request.model_dump(exclude_unset=True)
    observation = env.reset(**kwargs)
    return ResetResponse(**serialize_observation(observation))


@app.post("/step", response_model=StepResponse, tags=["Environment Control"])
async def step(request: StepRequest) -> StepResponse:
    env = _get_http_env()
    try:
        action = deserialize_action(request.action, AWSAction)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    kwargs = request.model_dump(exclude_unset=True, exclude={"action"})
    try:
        observation = env.step(action, **kwargs)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StepResponse(**serialize_observation(observation))


@app.get("/state", response_model=AWSState, tags=["State Management"])
async def state() -> AWSState:
    env = _get_http_env()
    try:
        return env.state
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
