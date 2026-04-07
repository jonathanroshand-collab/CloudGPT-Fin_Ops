---
title: CloudGPT-Fin Ops
emoji: "🏆"
colorFrom: yellow
colorTo: yellow
sdk: docker
pinned: false
---

# Safe FinOps: AWS Cost Env Benchmark

> *"We built an OpenEnv benchmark for safe cloud cost optimization, where agents must balance savings against infrastructure reliability and dependency constraints."*

---

## 1. The Core Problem: Safe FinOps Under Operational Constraints
Cloud waste is a massive, multi-billion-dollar dilemma. Every day, enterprise teams grapple with sprawling infrastructure, trying to squeeze monthly spend. But here is the brutal reality: **cost reduction is easy; *safe* cost reduction without breaking production is incredibly hard.** 

Teams desperately need tools to identify idle resources and optimize spending *without* accidentally deleting critical data pipelines or destroying downstream database dependents.

## 2. Why This Environment Matters
Most existing Reinforcement Learning (RL) environments and benchmarks focus heavily on simplistic games or highly controlled physics simulators. Real enterprise AI, however, requires environments with operational constraints, nuanced cost tradeoffs, and complex infrastructure dependency graphs. 

This repository shifts the focus from "playing games" to solving serious enterprise challenges. It is a strong, scalable environment built specifically for training cost-aware AI agents on real-world cloud infrastructure optimization.

## 3. Environment Design
This environment is cleanly designed aligning with robust OpenEnv frameworks:

* **Observation Space:** The agent observes an array of cloud resources containing types (`vm`, `db`, `storage`), `cpu_usage`, `min_required_cpu`, `critical` tags, dependency graphs, and granular `$ cost/mo`.
* **Action Space:** 
  - `delete`: Destroys a resource to eliminate 100% of its cost.
  - `resize`: Scales provisioned instance tiers (`small`, `medium`, `large`). 
  - `stop`: Halts a compute resource. Zeroes out CPU consumption and reduces cost to a residual 15% (EBS storage footprint). Safer than delete.
  - `noop`: Takes no action.
* **Reward Function:** A highly dynamic, **multi-factor FinOps reward system**. The agent earns positive continuous rewards based on fractional initial cost saved per step. It receives *Action Bonuses* for prioritizing non-critical idle waste, and incurs brutal *Negative Penalties* for triggering system failures via critical data loss, downstream dependency outages, or wasteful API sequences.
* **Done Conditions:** The episode terminates successfully after 5 optimized steps without system failure. Terminations also forcibly kick in if the system breaks or if an agent exhausts steps with zero cost trajectory improvements.

## 4. Benchmark Tasks
The environment ships out-of-the-box with standard benchmarking limits spanning 3 manual curated baselines, plus an infinite procedural generator:
- **Task 1 (Easy):** Simple, isolated VMs. An obvious single safe deletion target.
- **Task 2 (Medium):** Overprovisioned non-critical instances requiring surgical resizing while dancing around critical infrastructure.
- **Task 3 (Hard):** Multi-tier clusters with deep dependency graphs, misleading "idle" critical resources, and budget traps.
- **Task 4+ (Dynamic Generator):** Calling `task_id >= 4` transforms the core backend into an infinite cloud topology synthesizer, building randomized multi-node networks and realistic procedural waste.

## 5. Baselines and Benchmarks
This repository validates itself natively against a suite of AI implementations out of the box:
* **Random Agent:** A chaotic baseline testing the environment's strict failure penalties and constraints.
* **Heuristic Agent:** A rule-based classical algorithmic agent designed to blindly prune "idle" states.
* **LLM Agent (Qwen):** A state-of-the-art Hugging Face inference pipeline connecting directly with the AWS Cost Env to parse infrastructure graphs contextually.

*(Execute `python benchmark.py` to live-test comparative performance algorithms).*

## 6. Example API Episode
*In just a few steps, you can watch an agent slice infra costs dynamically.*

**[BEFORE] Initial Infrastructure State:**
`Total Cost: $550.00/mo` | `vm-analytics` costing $300 is running at `12.0% CPU`. 

⬇️ **Action Processed:**
```json
{
  "action_type": "resize",
  "resource_id": "vm-analytics",
  "new_size": "medium"
}
```
⬇️ **Environment Multi-Factor Evaluation:**
* `Reward:` +0.28 (Cost delta secured + Overprovisioned Bonus - No Penalties)
* `Valid Constraint:` `min_required_cpu` bounds passed successfully.

**[AFTER] Final Infrastructure State:**
`Total Cost: $415.00/mo` | `vm-analytics` resized successfully down to $165.00.
`Score:` ⭐⭐ Good!
`Money Saved:` $135.00 / month

*(Execute `python visual_demo.py` for a full, gorgeous terminal-based UI simulation matching this flow).*

---

## 7. Why This Matters for Agent Training
This isn't merely an API wrapper—it is a rigorous, battle-tested RL scaffolding platform. It can be utilized right now to train, validate, or evaluate state-of-the-art AI agents. From hackathon proof-of-concepts all the way to enterprise scaling workflows, the **AWS Cost Env** provides a standardized arena to benchmark exactly how safe and cost-aware large language models truly are when unleashed on production cloud topology.
