"""
Inference Script for AWS Cost Env.

Mandatory stdout format:
  [START] task=<n> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>

Environment variables required:
  HF_TOKEN       Your Hugging Face token (used as API key)
  API_BASE_URL   LLM endpoint (default: https://router.huggingface.co/v1)
  MODEL_NAME     Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
  ENV_BASE_URL   OpenEnv server URL (default: http://localhost:7860)
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

from openai import OpenAI
from aws_cost_env import AWSCostEnv
from aws_cost_env.models import AWSAction as Action


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

MAX_STEPS = 5
TASKS = [1, 2, 3]

SYSTEM_PROMPT = """You are an AWS FinOps engineer. You are given the current state of a cloud infrastructure.
Your job is to reduce monthly cost WITHOUT breaking the system.

Rules you MUST follow:
- NEVER delete or stop a resource where critical=True
- NEVER delete or stop a resource that appears in another resource's dependencies list
- NEVER resize a resource if the new_size cpu capacity is less than its min_required_cpu
  (small=10 cpu, medium=50 cpu, large=100 cpu)
- Prefer deleting resources with cpu_usage < 5% (idle resources) — they are safe to remove
- Prefer resizing to "small" only if min_required_cpu <= 10
- Use "noop" ONLY if there is genuinely no safe action left

You must respond with ONLY a valid JSON object. No explanation. No markdown. Just JSON.

Valid formats:
{"action_type": "delete", "resource_id": "vm-idle"}
{"action_type": "stop", "resource_id": "vm-idle"}
{"action_type": "resize", "resource_id": "vm-over", "new_size": "small"}
{"action_type": "noop"}
"""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def map_terminal_score(state_score: Optional[str], system_failure: bool) -> float:
    """Keep task scores strictly inside (0, 1) for every terminal outcome."""
    if system_failure:
        return 0.01
    if state_score == "Excellent":
        return 0.99
    if state_score == "Good":
        return 0.75
    if state_score == "Weak":
        return 0.25
    return 0.01


def build_prompt(state: dict, step_num: int) -> str:
    resources = state.get("resources", [])
    lines = [
        f"=== AWS Infrastructure State (Step {step_num}/{state.get('max_steps', MAX_STEPS)}) ===",
        f"Monthly cost: ${state.get('current_total_cost', 0):.2f} "
        f"(started at ${state.get('initial_total_cost', 0):.2f}, "
        f"saved {state.get('total_savings_ratio', 0.0) * 100:.1f}% so far)",
        "",
        "Resources:"
    ]

    for r in resources:
        deps = r.get("dependencies", [])
        stopped = r.get("stopped", False)
        lines.append(
            f"  [{r.get('id', '?')}]"
            f" type={r.get('type', '?')}"
            f" cost=${r.get('cost', 0):.2f}/mo"
            f" cpu={r.get('cpu_usage', 0)}%"
            f" min_cpu={r.get('min_required_cpu', 0)}%"
            f" critical={r.get('critical', False)}"
            f" stopped={stopped}"
            f" deps={deps if deps else 'none'}"
        )

    history = state.get("action_history", [])
    if history:
        lines.append("")
        lines.append("Actions taken so far:")
        for rec in history:
            rid = rec.get("resource_id") or ""
            ns = rec.get("new_size") or ""
            detail = f" {rid} {ns}".strip()
            lines.append(
                f"  step {rec.get('step')}: {rec.get('action_type')}{' ' + detail if detail else ''}"
                f" → reward={rec.get('reward', 0):.4f} failure={rec.get('system_failure', False)}"
            )

    lines.append("")
    lines.append("What is your next action? Respond with JSON only.")
    return "\n".join(lines)


def get_agent_action(client: OpenAI, state: dict, step_num: int) -> dict:
    prompt = build_prompt(state, step_num)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if model adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        # Normalize: some models return "action" instead of "action_type"
        if "action" in parsed and "action_type" not in parsed:
            parsed["action_type"] = parsed.pop("action")
        # Strip unknown fields that would fail validation
        return {k: v for k, v in parsed.items() if k in ("action_type", "resource_id", "new_size")}
    except Exception:
        return {"action_type": "noop"}

def run_episode(client: OpenAI, task_id: int) -> None:
    task_name = f"task{task_id}"
    log_start(task=task_name, env="aws-cost-env", model=MODEL_NAME)

    rewards: List[float] = []
    step_num = 0
    done = False
    success = False
    final_score = 0.0

    with AWSCostEnv(base_url=ENV_BASE_URL).sync() as env:
        state = env.reset(task_id=task_id).observation.model_dump()

        while not done and step_num < MAX_STEPS:
            step_num += 1
            action_dict = get_agent_action(client, state, step_num)
            action_str = json.dumps(action_dict)

            try:
                result = env.step(Action.model_validate(action_dict))
                reward = result.reward
                done = result.done
                state = result.observation.model_dump()
                error = state.get("failure_reason")

                rewards.append(reward)
                log_step(step_num, action_str, reward, done, error)

                if done:
                    state_score = state.get("score")
                    system_failure = bool(state.get("system_failure", False))
                    final_score = map_terminal_score(state_score, system_failure)
                    if not system_failure and state_score in ("Excellent", "Good"):
                        success = True

            except Exception as exc:
                log_step(step_num, action_str, 0.0, True, str(exc))
                rewards.append(0.0)
                final_score = 0.01
                done = True
                break

    log_end(success=success, steps=step_num, score=final_score, rewards=rewards)


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_id in TASKS:
        run_episode(client, task_id)


if __name__ == "__main__":
    main()
