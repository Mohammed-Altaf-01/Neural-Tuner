"""
Inference script for NeuralTuner — runs all 15 scenarios using the HF router.

Usage:
    HF_TOKEN=hf_... python inference.py                           # all 15 scenarios
    HF_TOKEN=hf_... python inference.py --difficulty easy         # 5 easy only
    HF_TOKEN=hf_... python inference.py --scenario inception_v3_medium
    HF_TOKEN=hf_... python inference.py --model Qwen/Qwen2.5-72B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

from openai import OpenAI

try:
    from models import NeuralTunerAction
    from server.neural_tuner_env_environment import NeuralTunerEnvironment
    from server.scenarios import EASY_SCENARIOS, HARD_SCENARIOS, MEDIUM_SCENARIOS, Scenario
except ImportError as _e:
    print(f"Import error: {_e}. Run from the project root with dependencies installed.", file=sys.stderr)
    sys.exit(1)

# ── Configuration ──────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
TEMPERATURE = 0.0
MAX_TOKENS = 1024
MAX_STEPS = 20

ALL_SCENARIOS: List[Scenario] = EASY_SCENARIOS + MEDIUM_SCENARIOS + HARD_SCENARIOS

SYSTEM_PROMPT = """You are an expert ML optimization agent for Qualcomm Snapdragon hardware.
Your goal is to reduce the latency and memory of a neural network while preserving accuracy.

You interact with the NeuralTuner environment by emitting exactly one tool call per turn in this format:
<tool_call>{"name": "<action>", "arguments": {<args>}}</tool_call>

Available actions:
- profile_layer(layer_id)        — reveal sensitivity and optimization hints for a layer
- quantize_layer(layer_id, dtype) — apply dtype quantization (FP32 | FP16 | INT8 | INT4)
- prune_layer(layer_id, sparsity) — structured pruning (LOW | MEDIUM | HIGH)
- revert_layer(layer_id)         — reset a layer to FP32 / no pruning
- benchmark()                    — simulate current plan and see latency/memory/accuracy
- submit()                       — finalise and score the episode

Strategy:
1. Profile the most expensive or sensitive-looking layers first.
2. Apply aggressive quantization (INT8 / INT4) to low-sensitivity layers.
3. Use FP16 for medium-sensitivity layers; leave high-sensitivity layers at FP32.
4. Call benchmark() after quantizing a batch to track progress.
5. Call submit() once all constraints are met."""


# ── Response parsing ───────────────────────────────────────────────────────────

def _parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Extract tool call from model output.

    Accepts three formats:
    1. <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    2. Bare JSON {"name": "...", "arguments": {...}}
    3. Bare JSON {"action_type": "...", "layer_id": ..., ...}
    """
    m = re.search(r"<tool_call>([\s\S]*?)</tool_call>", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Bare JSON object
    m2 = re.search(r"\{[\s\S]*\}", text)
    if m2:
        try:
            obj = json.loads(m2.group(0))
            if "name" in obj and "arguments" in obj:
                return obj
            if "action_type" in obj:
                return {"name": obj["action_type"], "arguments": {k: v for k, v in obj.items() if k != "action_type"}}
        except json.JSONDecodeError:
            pass
    return None


# ── Logging helpers ────────────────────────────────────────────────────────────

def _log_step(step: int, name: str, args: Dict, reward: float, done: bool) -> None:
    args_str = "  ".join(f"{k}={v}" for k, v in args.items() if v is not None)
    print(f"  [{step:02d}] {name}({args_str})  reward={reward:.4f}  done={done}", flush=True)


# ── Single episode ─────────────────────────────────────────────────────────────

def run_episode(client: OpenAI, scenario: Scenario, model: str = MODEL_NAME, max_steps: int = MAX_STEPS) -> float:
    """Run one full episode for *scenario* and return the final reward."""
    env = NeuralTunerEnvironment()
    reset_obs = env.reset(model_id=scenario.model_id, difficulty=scenario.difficulty)
    obs_text = reset_obs.output

    print(f"\n{'─'*60}", flush=True)
    print(f"Scenario : {scenario.name}  ({scenario.difficulty})", flush=True)
    print(f"Model    : {scenario.model_id}", flush=True)
    print(f"Constraints: latency≤{scenario.constraints.latency_budget_ms}ms  "
          f"memory≤{scenario.constraints.memory_budget_mb}MB  "
          f"accuracy≥{scenario.constraints.min_accuracy_retention}", flush=True)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": obs_text},
    ]

    final_reward = 0.0

    for step in range(1, max_steps + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [{step:02d}] API error: {exc}", flush=True)
            break

        tool_call = _parse_tool_call(response_text)
        if tool_call is None:
            print(f"  [{step:02d}] Could not parse tool call from: {response_text[:120]!r}", flush=True)
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": "Invalid response. Emit exactly one <tool_call>...</tool_call> block."
            })
            continue

        name = tool_call.get("name", "")
        args: Dict[str, Any] = tool_call.get("arguments", {})
        action = NeuralTunerAction(
            action_type=name,
            layer_id=args.get("layer_id"),
            dtype=args.get("dtype"),
            sparsity=args.get("sparsity"),
        )

        try:
            result = env.step(action)
        except Exception as exc:
            print(f"  [{step:02d}] env.step error: {exc}", flush=True)
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": f"Environment error: {exc}"})
            continue

        final_reward = float(result.reward)
        _log_step(step, name, args, final_reward, bool(result.done))

        messages.append({"role": "assistant", "content": response_text})
        messages.append({"role": "user", "content": result.output})

        if result.done:
            break

    return final_reward


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run NeuralTuner inference across all scenarios.")
    parser.add_argument("--model", default=MODEL_NAME, help="HF model ID (default: Qwen/Qwen2.5-72B-Instruct)")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None,
                        help="Restrict to one difficulty tier.")
    parser.add_argument("--scenario", default=None,
                        help="Run a single scenario by name (e.g. inception_v3_medium).")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()

    token = HF_TOKEN
    if not token:
        print("Error: set HF_TOKEN environment variable.", file=sys.stderr)
        sys.exit(1)

    model = args.model
    client = OpenAI(base_url=API_BASE_URL, api_key=token)

    # Select scenarios to run
    if args.scenario:
        scenarios = [s for s in ALL_SCENARIOS if s.name == args.scenario]
        if not scenarios:
            print(f"Unknown scenario '{args.scenario}'. Available:", file=sys.stderr)
            for s in ALL_SCENARIOS:
                print(f"  {s.name}", file=sys.stderr)
            sys.exit(1)
    elif args.difficulty:
        scenarios = [s for s in ALL_SCENARIOS if s.difficulty == args.difficulty]
    else:
        scenarios = ALL_SCENARIOS

    print(f"NeuralTuner Inference — model={model}", flush=True)
    print(f"Running {len(scenarios)} scenario(s)\n", flush=True)

    scores: Dict[str, float] = {}
    for scenario in scenarios:
        scores[scenario.name] = run_episode(client, scenario, model=model, max_steps=args.max_steps)

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for diff in ["easy", "medium", "hard"]:
        tier = {n: r for n, r in scores.items() if n.endswith(f"_{diff}")}
        if not tier:
            continue
        print(f"\n  {diff.upper()}")
        for name, reward in tier.items():
            print(f"    {name:<35} {reward:.4f}")
        print(f"    {'avg':35} {sum(tier.values())/len(tier):.4f}")
    if scores:
        avg = sum(scores.values()) / len(scores)
        print(f"\n  Overall average ({len(scores)} scenarios): {avg:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
