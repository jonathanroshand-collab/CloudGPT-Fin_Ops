import json
import time
from benchmark import LLMAgent
from aws_cost_env import AWSCostEnv, Action

def print_table(resources, total_cost):
    print("+" + "-"*81 + "+")
    print(f"| {'RESOURCE ID':<15} | {'TYPE':<7} | {'CRITICAL':<8} | {'CPU%':<6} | {'DEPENDENCIES':<15} | {'COST/mo':<13} |")
    print("+" + "-"*81 + "+")
    for r in resources:
        crit = "Yes" if r['critical'] else "No"
        deps = ",".join(r['dependencies']) if r['dependencies'] else "None"
        print(f"| {r['id']:<15} | {r['type']:<7} | {crit:<8} | {r['cpu_usage']:<6.1f} | {deps:<15} | ${r['cost']:<12.2f} |")
    print("+" + "-"*81 + "+")
    print(f"                                                          TOTAL: ${total_cost:.2f} / month")
    print()

def run_demo(task_id: int):
    agent = LLMAgent()
    
    print("\n" + "="*85)
    print(f"☁️  AWS FINOPS VISUAL DEMO --- (Task Seed: {task_id}) ☁️")
    print("="*85)
    
    with AWSCostEnv().sync() as env:
        state = env.reset(task_id=task_id).model_dump()
        
        print("\n[BEFORE] INITIAL INFRASTRUCTURE STATE:")
        print_table(state["resources"], state["initial_total_cost"])
        
        done = False
        while not done:
            print("🤖 Agent is analyzing infrastructure telemetry...")
            time.sleep(1)  # tiny pause for dramatic visual effect 
            
            action_dict = agent(state)
            action_type = action_dict.get("action_type")
            res_id = action_dict.get("resource_id", "N/A")
            
            if action_type == "resize":
                move = f"Resize {res_id} to '{action_dict.get('new_size')}' tier"
            elif action_type in ("delete", "stop"):
                move = f"{action_type.capitalize()} instances backing {res_id}"
            else:
                move = "Idle Observability Cycle (No-op)"
                
            print(f"\n⚡ ACTION INJECTED: [ {move} ]")
            
            # Use fallback handling logic directly to prevent runtime crash formatting errors
            try:
                action_payload = Action.model_validate(action_dict)
            except Exception:
                action_payload = Action(action_type="noop")
                
            result = env.step(action_payload)
            state = result.observation.model_dump()
            done = result.done
            
            print(f"   -> Action Multi-Factor Reward: {result.reward:+.2f}")
            print(f"   -> Cumulative Savings Ratio:   {state['total_savings_ratio']*100:.1f}%")
            if state['failure_reason']:
                print(f"   -> ⚠️  WARNING: {state['failure_reason']}")
                
            print("\n[CURRENT] INFRASTRUCTURE STATE:")
            print_table(state["resources"], state["current_total_cost"])
            time.sleep(0.5)
            
        print("="*85)
        print("🎯 FINOPS INCIDENT RESOLVED 🎯")
        print("="*85)
        
        score_icon = "❌" if state['system_failure'] else ("🌟" if state['score'] in ('Good', 'Excellent') else "📉")
        
        print(f"Score / Grade:       {score_icon} {state['score']}")
        print(f"System Failed:       {'Yes (Downtime Caused)' if state['system_failure'] else 'No (Zero Downtime)'}")
        print(f"Money Saved:         ${state['initial_total_cost'] - state['current_total_cost']:.2f} / month")
        print(f"Final Total Cost:    ${state['current_total_cost']:.2f} (Savings: {state['total_savings_ratio']*100:.1f}%)")
        print("\n")

if __name__ == "__main__":
    # Test one manual curated backend and one heavy procedural scenario
    for task in [2, 10]:
        try:
            run_demo(task_id=task)
        except Exception as e:
            print(f"Connection failed running demo. Is the server running? error: {e}")
