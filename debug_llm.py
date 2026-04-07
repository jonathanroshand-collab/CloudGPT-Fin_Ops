import os, json
from openai import OpenAI
from aws_cost_env import AWSCostEnv

client = OpenAI(
    base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
    api_key=os.getenv("HF_TOKEN", "")
)

with AWSCostEnv(base_url="http://localhost:7860").sync() as env:
    obs = env.reset(task_id=1).observation.model_dump()
    resources = obs.get("resources", [])

    prompt = "Current infrastructure:\n"
    for r in resources:
        prompt += f"  id={r['id']} cost=${r['cost']}/mo cpu={r['cpu_usage']}% critical={r['critical']} deps={r['dependencies']}\n"
    prompt += "\nDelete or resize the most wasteful non-critical resource. JSON only."

    print("PROMPT SENT:")
    print(prompt)
    print()

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"),
        messages=[
            {"role": "system", "content": "You are an AWS FinOps engineer. Respond with JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=100,
    )
    raw = response.choices[0].message.content
    print("LLM RAW RESPONSE:")
    print(repr(raw))