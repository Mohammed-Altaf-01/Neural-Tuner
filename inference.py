"""
Inference helper for NeuralTuner using a small Hugging Face reasoning model.

This file is intentionally inference-only:
- no RL training loop
- no checkpoint updates
- no optimizer logic

Use this to generate one action per turn from model output.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, cast

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_MODE = "chat_completion"
DEFAULT_MAX_STEPS = 20
ALLOWED_ACTIONS = {"profile_layer", "quantize_layer", "prune_layer", "revert_layer", "benchmark", "submit"}
ALLOWED_DTYPES = {"FP32", "FP16", "INT8", "INT4"}

try:
    from models import NeuralTunerAction
    from server.neural_tuner_env_environment import NeuralTunerEnvironment
except ImportError:  # pragma: no cover - defensive import fallback
    NeuralTunerAction = None  # type: ignore[assignment]
    NeuralTunerEnvironment = None  # type: ignore[assignment]


@dataclass
class ActionDecision:
    action_type: str
    layer_id: Optional[str] = None
    dtype: Optional[str] = None
    raw_text: str = ""
    thinking: str = ""


def build_prompt(observation_text: str) -> str:
    return (
        "You are an RL environment agent for NeuralTuner.\n"
        "Return exactly one JSON object with keys: action_type, layer_id, dtype.\n"
        "Allowed action_type: profile_layer, quantize_layer, revert_layer, benchmark, submit.\n"
        "If action_type does not require layer_id/dtype, set them to null.\n"
        "Choose safe, strategic quantization decisions.\n\n"
        f"Observation:\n{observation_text}\n"
    )


def _extract_json_block(text: str) -> Dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("Model output did not contain a JSON object.")
    return json.loads(match.group(0))


def _extract_thinking(text: str) -> str:
    match = re.search(r"<think>([\s\S]*?)</think>", text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def infer_with_chat_completion(prompt: str, model: str, timeout_s: int = 120) -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for Hugging Face router chat completion mode.")

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on local setup
        raise RuntimeError("openai package is required. Install with: pip install openai") from exc

    client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=token, timeout=timeout_s)
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=220,
    )
    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError("Empty response from chat completion API.")
    return content


def infer_local_transformers(prompt: str, model: str) -> str:
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - depends on local setup
        raise RuntimeError("transformers is not installed. Install with: pip install transformers torch") from exc

    generator = pipeline("text-generation", model=model)
    out = generator(
        prompt,
        max_new_tokens=220,
        do_sample=True,
        temperature=0.2,
        return_full_text=False,
    )
    return str(out[0]["generated_text"]).strip()


def decide_action(observation_text: str, model: str, mode: str) -> ActionDecision:
    prompt = build_prompt(observation_text)
    raw_text = (
        infer_with_chat_completion(prompt, model)
        if mode == "chat_completion"
        else infer_local_transformers(prompt, model)
    )
    thinking = _extract_thinking(raw_text)
    action_obj = _extract_json_block(raw_text)
    return ActionDecision(
        action_type=str(action_obj.get("action_type")),
        layer_id=action_obj.get("layer_id"),
        dtype=action_obj.get("dtype"),
        raw_text=raw_text,
        thinking=thinking,
    )


def run_default_pipeline(model: str, mode: str, difficulty: str, scenario_model: Optional[str], max_steps: int) -> int:
    if NeuralTunerEnvironment is None or NeuralTunerAction is None:
        raise RuntimeError("Environment imports unavailable. Run from project root with dependencies installed.")

    env = NeuralTunerEnvironment()
    reset_obs = env.reset(difficulty=difficulty, model_id=scenario_model)
    observation_text = reset_obs.output

    print("=== NeuralTuner default inference pipeline ===")
    print(f"model={model}")
    print(f"mode={mode}")
    print(f"scenario_model={scenario_model or 'random'} difficulty={difficulty}")
    print()

    for step_idx in range(1, max_steps + 1):
        decision = decide_action(observation_text, model, mode)
        if decision.action_type not in ALLOWED_ACTIONS:
            raise RuntimeError(f"Model returned unsupported action_type: {decision.action_type}")
        if decision.dtype is not None and decision.dtype not in ALLOWED_DTYPES:
            raise RuntimeError(f"Model returned unsupported dtype: {decision.dtype}")

        action_type = cast(
            Literal["profile_layer", "quantize_layer", "revert_layer", "benchmark", "submit"],
            decision.action_type,
        )
        dtype = cast(Optional[Literal["FP32", "FP16", "INT8", "INT4"]], decision.dtype)
        action = NeuralTunerAction(
            action_type=action_type,
            layer_id=decision.layer_id,
            dtype=dtype,
        )
        result = env.step(action)
        observation_text = result.output

        print(f"[step {step_idx}] action={decision.action_type} layer={decision.layer_id} dtype={decision.dtype}")
        if decision.thinking:
            print("  thinking:", decision.thinking[:220].replace("\n", " "))
        print(f"  reward={result.reward} done={result.done} success={result.success}")

        if result.done:
            print("\n=== Episode complete ===")
            print(f"final_reward={result.reward}")
            return 0

    print("\nPipeline reached max steps without termination.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run NeuralTuner inference. Without args, runs default end-to-end episode pipeline."
    )
    parser.add_argument("--observation", default=None, help="Observation text for one-step action inference.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id to run.")
    parser.add_argument(
        "--mode",
        choices=["chat_completion", "local"],
        default=DEFAULT_MODE,
        help="chat_completion uses HF router + OpenAI client, local uses transformers pipeline.",
    )
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--scenario-model", default="inception_v3", help="Scenario model id for default pipeline.")
    parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Max inference turns in default pipeline."
    )
    args = parser.parse_args()

    try:
        if args.observation:
            decision = decide_action(args.observation, args.model, args.mode)
            payload = {
                "action_type": decision.action_type,
                "layer_id": decision.layer_id,
                "dtype": decision.dtype,
            }
            print(json.dumps(payload, indent=2))
            if decision.thinking:
                print("\n--- extracted_thinking ---\n")
                print(decision.thinking)
            return 0

        return run_default_pipeline(
            model=args.model,
            mode=args.mode,
            difficulty=args.difficulty,
            scenario_model=args.scenario_model,
            max_steps=args.max_steps,
        )
    except Exception as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
