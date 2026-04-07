import json
import os
import random
from typing import Any, Callable, Dict, List

from openai import OpenAI

from aws_cost_env import AWSAction, AWSCostEnv
from inference import API_BASE_URL, API_KEY, MODEL_NAME, get_agent_action

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")
TASKS = [1, 2, 3, 4, 5]  # 1-3 are manual, 4-5 are procedurally generated


def random_agent(state: dict) -> dict:
    """Agent that chooses purely random actions. Baseline for failure/waste."""
    resources = state.get("resources", [])
    if not resources:
        return {"action_type": "noop"}
    
    r = random.choice(resources)
    action_type = random.choice(["delete", "stop", "resize", "noop"])
    
    if action_type == "resize":
        size = random.choice(["small", "medium", "large"])
        return {"action_type": "resize", "resource_id": r["id"], "new_size": size}
    elif action_type in ["delete", "stop"]:
        return {"action_type": action_type, "resource_id": r["id"]}
        
    return {"action_type": "noop"}


def heuristic_agent(state: dict) -> dict:
    """Smart rule-based classical algorithmic baseline."""
    resources = state.get("resources", [])
    dependencies = {d for r in resources for d in r.get("dependencies", [])}
            
    # 1. Try to stop perfectly idle, non-critical resources first (safe tradeoff)
    for r in resources:
        if not r["critical"] and r["id"] not in dependencies:
            if r["cpu_usage"] < 2.0:
                return {"action_type": "stop", "resource_id": r["id"]}
                
    # 2. Try to downsize obviously overprovisioned resources
    for r in resources:
        if not r["critical"]:
            if r["cpu_usage"] <= 10.0 and r["min_required_cpu"] <= 10.0:
                return {"action_type": "resize", "resource_id": r["id"], "new_size": "small"}
            elif r["cpu_usage"] <= 50.0 and r["min_required_cpu"] <= 50.0:
                return {"action_type": "resize", "resource_id": r["id"], "new_size": "medium"}
                
    # 3. Give up intentionally rather than guessing and causing a system failure
    return {"action_type": "noop"}


class LLMAgent:
    """Wrapper that taps into the AI capabilities using our inference.py setup."""
    def __init__(self):
        self.client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        
    def __call__(self, state: dict) -> dict:
        step_num = len(state.get("action_history", [])) + 1
        action_dict = get_agent_action(self.client, state, step_num)
        return action_dict


def run_benchmark(agent_name: str, agent_fn: Callable[[dict], dict], tasks: List[int]) -> Dict[str, Any]:
    print(f"\n--- Running Benchmark for [{agent_name}] Agent ---")
    results = {
        "episodes": len(tasks),
        "successes": 0,
        "failures": 0,
        "total_savings_ratio": 0.0,
        "total_reward": 0.0,
        "total_steps": 0
    }
    
    for task_id in tasks:
        with AWSCostEnv(base_url=ENV_BASE_URL).sync() as env:
            try:
                state = env.reset(task_id=task_id).observation.model_dump()
            except Exception as e:
                print(f"Failed to reset task {task_id}: {e}")
                continue
                
            done = False
            total_r = 0.0
            steps = 0
            
            while not done:
                action_dict = agent_fn(state)
                # Ensure safety bounds on random agent sending absolute garbage formats
                try:
                    action_payload = AWSAction.model_validate(action_dict)
                except Exception:
                    action_payload = AWSAction(action_type="noop")
                    
                try:
                    result = env.step(action_payload)
                    state = result.observation.model_dump()
                    total_r += result.reward
                    done = result.done
                    steps += 1
                except Exception as exc:
                    # Random bounds hit framework crash constraint
                    state["system_failure"] = True
                    break
                    
            # Capture Score End States
            score = state.get("score", "Failure")
            if score in ("Excellent", "Good"):
                results["successes"] += 1
            if state.get("system_failure", False):
                results["failures"] += 1
                
            results["total_savings_ratio"] += state.get("total_savings_ratio", 0.0)
            results["total_reward"] += total_r
            results["total_steps"] += steps

    return results


def main():
    agents = {
        "Random Baseline": random_agent,
        "Heuristic": heuristic_agent,
        f"LLM:{MODEL_NAME.split('/')[-1]}": LLMAgent()
    }
    
    print(f"Starting Benchmark Evaluation on Tasks: {TASKS}")
    print("Tasks 1-3: Curated Baseline | Tasks 4-5: Procedurally Generated Demos")
    
    report = {}
    for name, fn in agents.items():
        stats = run_benchmark(name, fn, TASKS)
        report[name] = stats
        
    print("\n" + "="*85)
    print(f"{'AGENT MODEL':<25} | {'SUCCESS RATE':<15} | {'CRASHES':<10} | {'AVG SAVINGS':<15} | {'AVG REWARD'}")
    print("="*85)
    
    for name, stats in report.items():
        eps = stats["episodes"]
        if eps == 0: eps = 1
        s_rate = (stats["successes"] / eps) * 100
        crashes = stats["failures"]
        a_sav = (stats["total_savings_ratio"] / eps) * 100
        a_rew = stats["total_reward"] / eps
        
        print(f"{name:<25} | {s_rate:>13.1f}% | {crashes:>10} | {a_sav:>14.1f}% | {a_rew:>10.2f}")
    print("="*85)

if __name__ == "__main__":
    main()
