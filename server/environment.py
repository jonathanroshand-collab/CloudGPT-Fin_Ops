from __future__ import annotations

import copy
import random
import uuid
from typing import Optional

from openenv.core.env_server import Environment
from aws_cost_env.models import AWSAction, ActionRecord, AWSObservation, AWSState, Resource

MAX_STEPS = 5
NO_IMPROVEMENT_LIMIT = 2

SIZE_TO_CPU: dict[str, float] = {
    "small": 10.0,
    "medium": 50.0,
    "large": 100.0,
}

SIZE_TO_COST_MULTIPLIER: dict[str, float] = {
    "small": 0.25,
    "medium": 0.55,
    "large": 1.0,
}

SUCCESS_SAVINGS_THRESHOLD = 0.20


def get_task(task_id: int) -> list[Resource]:
    if task_id == 1:
        return _task1()
    if task_id == 2:
        return _task2()
    if task_id == 3:
        return _task3()
    if task_id >= 4:
        return _generate_random_task(task_id)
    raise ValueError(f"Unknown task_id: {task_id}. Must be >= 1.")


def _task1() -> list[Resource]:
    return [
        Resource(id="vm-prod", type="vm", cost=200.0, cpu_usage=72.0, min_required_cpu=40.0, critical=True, dependencies=[]),
        Resource(id="vm-idle", type="vm", cost=120.0, cpu_usage=0.3, min_required_cpu=0.0, critical=False, dependencies=[]),
        Resource(id="storage-logs", type="storage", cost=50.0, cpu_usage=1.0, min_required_cpu=0.0, critical=False, dependencies=[]),
    ]


def _task2() -> list[Resource]:
    return [
        Resource(id="vm-critical", type="vm", cost=400.0, cpu_usage=85.0, min_required_cpu=60.0, critical=True, dependencies=["db-prod"]),
        Resource(id="db-prod", type="db", cost=300.0, cpu_usage=55.0, min_required_cpu=50.0, critical=True, dependencies=[]),
        Resource(id="vm-over", type="vm", cost=380.0, cpu_usage=8.0, min_required_cpu=5.0, critical=False, dependencies=[]),
        Resource(id="db-dev", type="db", cost=200.0, cpu_usage=6.0, min_required_cpu=5.0, critical=False, dependencies=[]),
        Resource(id="storage-backup", type="storage", cost=80.0, cpu_usage=2.0, min_required_cpu=0.0, critical=False, dependencies=[]),
    ]


def _task3() -> list[Resource]:
    return [
        Resource(id="vm-prod", type="vm", cost=500.0, cpu_usage=78.0, min_required_cpu=60.0, critical=True, dependencies=["db-prod", "storage-prod"]),
        Resource(id="db-prod", type="db", cost=420.0, cpu_usage=65.0, min_required_cpu=50.0, critical=True, dependencies=[]),
        Resource(id="storage-prod", type="storage", cost=100.0, cpu_usage=5.0, min_required_cpu=0.0, critical=True, dependencies=[]),  # TRAP: low CPU but critical
        Resource(id="vm-analytics", type="vm", cost=300.0, cpu_usage=12.0, min_required_cpu=5.0, critical=False, dependencies=["storage-analytics"]),
        Resource(id="storage-analytics", type="storage", cost=60.0, cpu_usage=2.0, min_required_cpu=0.0, critical=False, dependencies=[]),
        Resource(id="vm-bloat", type="vm", cost=350.0, cpu_usage=7.0, min_required_cpu=5.0, critical=False, dependencies=[]),
        Resource(id="db-staging", type="db", cost=180.0, cpu_usage=4.0, min_required_cpu=3.0, critical=False, dependencies=[]),
        Resource(id="storage-orphan", type="storage", cost=40.0, cpu_usage=0.0, min_required_cpu=0.0, critical=False, dependencies=[]),
    ]


def _generate_random_task(seed: int) -> list[Resource]:
    random.seed(seed)
    num_resources = random.randint(5, 12)
    resources = []

    for i in range(num_resources):
        rtype = random.choices(["vm", "db", "storage"], weights=[0.6, 0.2, 0.2])[0]
        critical = random.random() < 0.3
        char = random.choices(["normal", "idle", "overprovisioned"], weights=[0.5, 0.25, 0.25])[0]

        if rtype == "storage":
            cost = float(random.randint(20, 150))
            cpu = random.uniform(0, 1.0) if char == "idle" else random.uniform(2.0, 15.0)
            min_cpu = 0.0
        elif rtype == "db":
            cost = float(random.randint(150, 600))
            if char == "idle":
                cpu, min_cpu = random.uniform(1.0, 5.0), random.uniform(1.0, 3.0)
            elif char == "overprovisioned":
                cpu, min_cpu = random.uniform(10.0, 30.0), random.uniform(5.0, 10.0)
            else:
                cpu, min_cpu = random.uniform(50.0, 85.0), random.uniform(40.0, 60.0)
        else:
            cost = float(random.randint(100, 500))
            if char == "idle":
                cpu, min_cpu = random.uniform(0.0, 4.0), 0.0
            elif char == "overprovisioned":
                cpu, min_cpu = random.uniform(5.0, 20.0), random.uniform(2.0, 5.0)
            else:
                cpu, min_cpu = random.uniform(60.0, 95.0), random.uniform(40.0, 50.0)

        resources.append(Resource(
            id=f"{rtype}-{i}",
            type=rtype,
            cost=round(cost, 2),
            cpu_usage=round(cpu, 1),
            min_required_cpu=round(min_cpu, 1),
            critical=critical,
            dependencies=[]
        ))

    dbs_and_storages = [r.id for r in resources if r.type in ("db", "storage")]
    if dbs_and_storages:
        for r in resources:
            if r.type == "vm" and random.random() < 0.5:
                deps = random.sample(dbs_and_storages, k=min(random.randint(1, 2), len(dbs_and_storages)))
                r.dependencies = deps
                if r.critical:
                    for d_id in deps:
                        next(res for res in resources if res.id == d_id).critical = True

    safe_opt = any(
        not r.critical and not any(r.id in x.dependencies for x in resources)
        and (r.cpu_usage < 5.0 or r.min_required_cpu <= 10.0)
        for r in resources
    )
    if not safe_opt:
        resources.append(Resource(
            id=f"vm-{num_resources}_idle",
            type="vm",
            cost=150.0,
            cpu_usage=1.0,
            min_required_cpu=0.0,
            critical=False,
            dependencies=[]
        ))

    return resources


class AWSCostEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._state: Optional[AWSState] = None
        self._prev_costs: list[float] = []

    def reset(self, seed=None, episode_id=None, task_id: int = 1, **kwargs) -> AWSObservation:
        resources = get_task(task_id)
        total = self._sum_cost(resources)
        self._prev_costs = []
        self._state = AWSState(
            task_id=task_id,
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            resources=resources,
            initial_total_cost=total,
            current_total_cost=total,
            max_steps=MAX_STEPS,
            system_failure=False,
            failure_reason=None,
            action_history=[],
            total_savings_ratio=0.0,
            score="Running"
        )
        return self._to_observation(done=False, reward=0.0)

    def step(self, action: AWSAction, timeout_s=None, **kwargs) -> AWSObservation:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self.is_done():
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._validate_action(action)

        prev_cost = self._state.current_total_cost
        system_failure = False
        failure_reason: Optional[str] = None
        action_bonus = 0.0
        action_penalty = 0.0

        if action.action_type == "delete":
            system_failure, failure_reason, action_bonus, action_penalty = self._apply_delete(action.resource_id)
        elif action.action_type == "stop":
            system_failure, failure_reason, action_bonus, action_penalty = self._apply_stop(action.resource_id)
        elif action.action_type == "resize":
            system_failure, failure_reason, action_bonus, action_penalty = self._apply_resize(action.resource_id, action.new_size)
        elif action.action_type == "noop":
            if self._has_safe_optimizations():
                action_penalty += 0.1
            else:
                action_bonus += 0.05

        self._state.step_count += 1
        self._state.current_total_cost = self._sum_cost(self._state.resources)
        self._state.system_failure = system_failure
        self._state.failure_reason = failure_reason

        reward = self._compute_reward(prev_cost, system_failure, action_bonus, action_penalty)

        self._state.total_savings_ratio = round(
            (self._state.initial_total_cost - self._state.current_total_cost)
            / self._state.initial_total_cost,
            4,
        )

        done = self._check_done(prev_cost)
        self._state.score = self._compute_score(done)
        self._prev_costs.append(self._state.current_total_cost)

        self._state.action_history.append(ActionRecord(
            step=self._state.step_count,
            action_type=action.action_type,
            resource_id=action.resource_id,
            new_size=action.new_size,
            reward=round(reward, 4),
            system_failure=system_failure,
        ))

        return self._to_observation(done=done, reward=round(reward, 4))

    def _to_observation(self, done: bool, reward: float) -> AWSObservation:
        return AWSObservation(
            done=done,
            reward=reward,
            task_id=self._state.task_id,
            resources=copy.deepcopy(self._state.resources),
            initial_total_cost=self._state.initial_total_cost,
            current_total_cost=self._state.current_total_cost,
            max_steps=self._state.max_steps,
            system_failure=self._state.system_failure,
            failure_reason=self._state.failure_reason,
            action_history=copy.deepcopy(self._state.action_history),
            total_savings_ratio=self._state.total_savings_ratio,
            score=self._state.score
        )

    @property
    def state(self) -> AWSState:
        if self._state is None:
            raise RuntimeError("Call reset() before state.")
        return copy.deepcopy(self._state)

    def is_success(self) -> bool:
        if self._state is None or not self.is_done():
            return False
        return self._state.score in ("Excellent", "Good")

    def is_done(self) -> bool:
        return self._state.score != "Running" if self._state else False

    def _compute_score(self, done: bool) -> str:
        if self._state is None or not done:
            return "Running"
        if self._state.system_failure:
            return "Failure"
        if self._state.total_savings_ratio >= 0.30:
            return "Excellent"
        if self._state.total_savings_ratio >= 0.20:
            return "Good"
        return "Weak"

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    def _apply_delete(self, resource_id: Optional[str]) -> tuple[bool, Optional[str], float, float]:
        resource = self._find_resource(resource_id)
        bonus, penalty = 0.0, 0.0

        if resource is None:
            return True, f"Resource '{resource_id}' not found.", 0.0, 0.1
        if resource.critical:
            return True, f"Cannot delete critical resource '{resource_id}'.", 0.0, 0.5
        dependents = self._find_dependents(resource_id)
        if dependents:
            return True, f"Cannot delete '{resource_id}': depended on by {dependents}.", 0.0, 0.5

        if resource.cpu_usage < 5.0:
            bonus += 0.1

        self._state.resources = [r for r in self._state.resources if r.id != resource_id]
        return False, None, bonus, penalty

    def _apply_stop(self, resource_id: Optional[str]) -> tuple[bool, Optional[str], float, float]:
        resource = self._find_resource(resource_id)
        bonus, penalty = 0.0, 0.0

        if resource is None:
            return True, f"Resource '{resource_id}' not found.", 0.0, 0.1
        if resource.type == "storage":
            return True, f"Cannot stop storage resource '{resource_id}'.", 0.0, 0.1
        if resource.critical:
            return True, f"Cannot stop critical resource '{resource_id}'.", 0.0, 0.5
        dependents = self._find_dependents(resource_id)
        if dependents:
            return True, f"Cannot stop '{resource_id}': depended on by {dependents}.", 0.0, 0.5

        # FIX: block re-stopping an already stopped resource
        if resource.stopped:
            return True, f"Resource '{resource_id}' is already stopped.", 0.0, 0.1

        if resource.cpu_usage < 5.0:
            bonus += 0.15

        new_cost = round(resource.base_cost * 0.15, 2)
        if new_cost >= resource.cost:
            penalty += 0.1
        else:
            resource.cost = new_cost
            resource.cpu_usage = 0.0
            resource.stopped = True   # FIX: mark as stopped

        return False, None, bonus, penalty

    def _apply_resize(self, resource_id: Optional[str], new_size: Optional[str]) -> tuple[bool, Optional[str], float, float]:
        resource = self._find_resource(resource_id)
        bonus, penalty = 0.0, 0.0

        if resource is None:
            return True, f"Resource '{resource_id}' not found.", 0.0, 0.1
        if new_size not in SIZE_TO_CPU:
            return True, f"Unknown size '{new_size}'.", 0.0, 0.1

        new_cpu = SIZE_TO_CPU[new_size]
        if new_cpu < resource.min_required_cpu:
            return True, (
                f"Resize '{resource_id}' to '{new_size}' "
                f"(cpu={new_cpu}) violates min_required_cpu={resource.min_required_cpu}."
            ), 0.0, 0.5

        new_cost = round(resource.base_cost * SIZE_TO_COST_MULTIPLIER[new_size], 2)

        if new_cost >= resource.cost:
            penalty += 0.1
        else:
            if not resource.critical and resource.cpu_usage <= new_cpu:
                bonus += 0.1

        resource.cost = new_cost
        resource.cpu_usage = min(resource.cpu_usage, new_cpu)
        return False, None, bonus, penalty

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, prev_cost: float, system_failure: bool, bonus: float, penalty: float) -> float:
        step_savings = (prev_cost - self._state.current_total_cost) / self._state.initial_total_cost

        if system_failure:
            penalty += 1.0
        for r in self._state.resources:
            if r.cpu_usage < r.min_required_cpu:
                penalty += 0.5
                break

        return step_savings + bonus - penalty

    # ------------------------------------------------------------------
    # Done conditions
    # ------------------------------------------------------------------

    def _check_done(self, prev_cost: float) -> bool:
        if self._state.system_failure:
            return True
        if self._state.step_count >= MAX_STEPS:
            return True
        if len(self._prev_costs) >= NO_IMPROVEMENT_LIMIT:
            last_n = self._prev_costs[-NO_IMPROVEMENT_LIMIT:]
            if all(c >= prev_cost for c in last_n):
                return True
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_resource(self, resource_id: Optional[str]) -> Optional[Resource]:
        if not resource_id:
            return None
        for r in self._state.resources:
            if r.id == resource_id:
                return r
        return None

    def _has_safe_optimizations(self) -> bool:
        if not self._state:
            return False
        for r in self._state.resources:
            if r.critical:
                continue
            if self._find_dependents(r.id):
                continue
            if r.cpu_usage < 5.0 and not r.stopped:
                return True
            for size, mult in SIZE_TO_COST_MULTIPLIER.items():
                new_cost = round(r.base_cost * mult, 2)
                if new_cost < r.cost and SIZE_TO_CPU[size] >= r.min_required_cpu:
                    return True
        return False

    def _find_dependents(self, resource_id: str) -> list[str]:
        return [r.id for r in self._state.resources if resource_id in r.dependencies]

    @staticmethod
    def _sum_cost(resources: list[Resource]) -> float:
        return round(sum(r.cost for r in resources), 2)

    def _validate_action(self, action: AWSAction) -> None:
        if action.action_type in ("delete", "stop") and not action.resource_id:
            raise ValueError(f"{action.action_type} requires resource_id.")
        if action.action_type == "resize":
            if not action.resource_id:
                raise ValueError("resize requires resource_id.")
            if not action.new_size:
                raise ValueError("resize requires new_size.")
