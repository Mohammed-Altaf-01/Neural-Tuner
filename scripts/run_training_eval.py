from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import trl.extras.profiling as trl_profiling
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from trl.chat_template_utils import qwen3_chat_template, qwen3_schema

import wandb

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rollout_eval import run_baseline_episode, run_heuristic_episode, run_random_episode
from scripts.neural_tuner import NeuralTunerOpenEnv
from server.neural_tuner_env_environment import NeuralTunerEnvironment
from training_utils import load_jsonl, mean, split_scenarios, write_json


def build_prompt_row(system_prompt: str, scenario: dict) -> list[dict]:
    scenario_text = (
        f"Scenario ID: {scenario['id']}\n"
        f"Model: {scenario['model_id']}\n"
        f"Difficulty: {scenario['difficulty']}\n"
        f"Hint: {scenario['scenario_hint']}\n"
        f"Target behavior: {scenario['target_behavior']}\n"
        f"Notes: {scenario['notes']}\n"
    )
    return [{"role": "user", "content": system_prompt + "\n\n" + scenario_text}]


def reward_func(environments, **kwargs) -> list[float]:
    rewards = []
    for env in environments:
        if not env.done:
            env.submit()
        rewards.append(float(env.reward))
    return rewards


def evaluate_split(eval_rows: list[dict], n_random: int = 10) -> dict:
    env = NeuralTunerEnvironment()
    random_rewards = []
    baseline_rewards = []
    heuristic_rewards = []

    for idx, row in enumerate(eval_rows):
        model_id = row["model_id"]
        difficulty = row["difficulty"]
        for seed in range(n_random):
            random_rewards.append(run_random_episode(env, model_id, difficulty, seed=idx * 100 + seed).final_reward)
        baseline_rewards.append(run_baseline_episode(env, model_id, difficulty).final_reward)
        heuristic_rewards.append(run_heuristic_episode(env, model_id, difficulty).final_reward)

    random_mean = mean(random_rewards)
    oracle_ceiling = mean(heuristic_rewards) if heuristic_rewards else 0.0
    return {
        "eval_random_mean": random_mean,
        "eval_baseline_mean": mean(baseline_rewards),
        "eval_heuristic_mean": oracle_ceiling,
        "eval_oracle_ceiling": oracle_ceiling,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible NeuralTuner training + held-out eval.")
    parser.add_argument("--model-path", default=".hf_cache/Qwen--Qwen2.5-1.5B-Instruct")
    parser.add_argument("--scenarios", default="data/mock_data/neural_tuner_train_scenarios.jsonl")
    parser.add_argument("--output-dir", default="outputs/neural_tuner_grpo_tiny_lora")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    wandb_key = os.getenv("WANDB_API_KEY")
    trl_profiling.wandb = wandb
    if wandb_key:
        wandb.login(key=wandb_key)
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "round_2"),
            name=f"neural_tuner_grpo_script_{args.seed}",
            resume="never",
            reinit=True,
        )

    model_name = str(Path(args.model_path).resolve())
    scenarios = load_jsonl(Path(args.scenarios))
    train_rows, eval_rows = split_scenarios(scenarios, train_fraction=args.train_fraction, seed=args.seed)

    system_prompt = (
        "You are a hardware optimization agent for NeuralTuner.\n"
        "Use tools to profile layers, apply quantization/pruning, benchmark, and submit."
    )
    train_dataset = Dataset.from_dict(
        {
            "prompt": [build_prompt_row(system_prompt, s) for s in train_rows],
            "model_id": [s["model_id"] for s in train_rows],
            "difficulty": [s["difficulty"] for s in train_rows],
            "scenario_id": [s["id"] for s in train_rows],
        }
    )

    processing_class = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    processing_class.chat_template = qwen3_chat_template
    processing_class.response_schema = qwen3_schema

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    NeuralTunerOpenEnv.scenario_schedule = train_rows
    NeuralTunerOpenEnv.schedule_idx = 0

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        run_name=f"neural_tuner_grpo_script_{args.seed}",
        max_steps=args.max_steps,
        learning_rate=8e-7,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=2,
        max_completion_length=256,
        num_generations=8,
        generation_batch_size=8,
        temperature=0.9,
        top_p=0.95,
        generation_kwargs={"do_sample": True, "renormalize_logits": True, "remove_invalid_values": True},
        mask_truncated_completions=False,
        max_tool_calling_iterations=20,
        log_completions=False,
        gradient_checkpointing=True,
        torch_empty_cache_steps=1,
        logging_steps=1,
        report_to="wandb" if wandb_key else "none",
        # Force CPU for scripted sweeps to avoid MPS/offload meta-device backward errors.
        use_cpu=True,
        bf16=False,
        fp16=False,
        use_vllm=False,
    )

    train_metrics = {"train_skipped": bool(args.skip_train)}
    if not args.skip_train:
        trainer = GRPOTrainer(
            model=model_name,
            reward_funcs=reward_func,
            train_dataset=train_dataset,
            args=grpo_config,
            processing_class=processing_class,
            peft_config=peft_config,
            environment_factory=NeuralTunerOpenEnv,
        )
        trainer_stats = trainer.train()
        trainer.save_model(args.output_dir)
        train_metrics.update(trainer_stats.metrics)
        reward_logs = [x["reward"] for x in trainer.state.log_history if "reward" in x]
        if reward_logs:
            train_metrics["training_reward_mean"] = mean(reward_logs)
            train_metrics["training_reward_last5_mean"] = mean(reward_logs[-5:])

    eval_metrics = evaluate_split(eval_rows, n_random=5)
    merged = {
        **train_metrics,
        **eval_metrics,
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
    }
    # Keep lift metrics explicit to avoid mixing train/eval interpretations.
    merged["lift_eval_baseline_vs_random"] = merged["eval_baseline_mean"] - merged["eval_random_mean"]
    merged["lift_eval_heuristic_vs_random"] = merged["eval_heuristic_mean"] - merged["eval_random_mean"]
    if "training_reward_mean" in merged:
        merged["lift_train_reward_vs_eval_random"] = merged["training_reward_mean"] - merged["eval_random_mean"]
    write_json(Path("artifacts/training/train_eval_script_metrics.json"), merged)

    if wandb_key:
        wandb.log(merged)
        wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
