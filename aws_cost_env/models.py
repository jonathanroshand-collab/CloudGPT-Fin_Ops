from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server import Action, Observation, State


class Resource(BaseModel):
    id: str
    type: Literal["vm", "db", "storage"]
    cost: float
    base_cost: float = 0.0
    cpu_usage: float = Field(ge=0, le=100)
    min_required_cpu: float = Field(ge=0, le=100)
    critical: bool
    stopped: bool = False                          # FIX: prevents re-stop exploit
    dependencies: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        if self.base_cost == 0.0:
            object.__setattr__(self, "base_cost", self.cost)


class ActionRecord(BaseModel):
    step: int
    action_type: str
    resource_id: Optional[str] = None
    new_size: Optional[str] = None
    reward: float
    system_failure: bool


class AWSState(State):
    task_id: int
    resources: list[Resource]
    initial_total_cost: float
    current_total_cost: float
    max_steps: int
    system_failure: bool
    failure_reason: Optional[str] = None
    action_history: list[ActionRecord] = Field(default_factory=list)
    total_savings_ratio: float = 0.0
    score: Literal["Excellent", "Good", "Weak", "Failure", "Running"] = "Running"


class AWSObservation(Observation):
    task_id: int
    resources: list[Resource]
    initial_total_cost: float
    current_total_cost: float
    max_steps: int
    system_failure: bool
    failure_reason: Optional[str] = None
    action_history: list[ActionRecord] = Field(default_factory=list)
    total_savings_ratio: float = 0.0
    score: Literal["Excellent", "Good", "Weak", "Failure", "Running"] = "Running"


class AWSAction(Action):
    action_type: Literal["delete", "resize", "stop", "noop"]
    resource_id: Optional[str] = None
    new_size: Optional[Literal["small", "medium", "large"]] = None
