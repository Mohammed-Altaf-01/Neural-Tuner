"""
Deterministic rollout evaluation for NeuralTuner.

Runs baseline and heuristic policies directly against the environment class
and writes episode metrics to JSON/CSV for plotting in notebooks/README.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from models import NeuralTunerAction
from server.neural_tuner_env_environment import NeuralTunerEnvironment


@dataclass
class EpisodeMetrics:
    policy: str
    episode_index: int
    model_id: str
    difficulty: str
    final_reward: float
    done: bool
    step_count: int
    benchmark_count: int
    latency_ms: float
    memory_mb: float
    accuracy_retention: float
    constraints_met: bool


def _layer_ids_from_reset(reset_output: str) -> List[str]:
    layer_ids: List[str] = []
    for line in reset_output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("ACTIONS") or stripped.startswith("Tip:"):
            break
        if stripped.startswith("Layer ID") or stripped.startswith("-"):
            continue
        if "HIDDEN" not in stripped:
            continue
        tokens = stripped.split()
        if tokens:
            layer_ids.append(tokens[0])
    return layer_ids


def run_random_episode(
    env: NeuralTunerEnvironment,
    model_id: str,
    difficulty: str,
    seed: Optional[int] = None,
) -> EpisodeMetrics:
    """Truly random policy: pick a random dtype for every layer without profiling.

    This is the pre-training baseline that shows what zero knowledge looks like.
    Expected reward: 0.20–0.45 depending on model/difficulty.
    """
    rng = random.Random(seed)
    dtypes = ["FP32", "FP16", "INT8", "INT4"]

    reset_obs = env.reset(model_id=model_id, difficulty=difficulty, seed=seed)
    layer_ids = _layer_ids_from_reset(reset_obs.output)
    for layer_id in layer_ids:
        env.step(NeuralTunerAction(action_type="quantize_layer", layer_id=layer_id, dtype=rng.choice(dtypes)))
    env.step(NeuralTunerAction(action_type="benchmark"))
    final = env.step(NeuralTunerAction(action_type="submit"))
    report = final.metadata or {}
    st = env.state
    return EpisodeMetrics(
        policy="random",
        episode_index=0,
        model_id=model_id,
        difficulty=difficulty,
        final_reward=final.reward,
        done=final.done,
        step_count=st.step_count,
        benchmark_count=st.benchmark_count,
        latency_ms=float(report.get("quantized_latency_ms", 0.0)),
        memory_mb=float(report.get("quantized_memory_mb", 0.0)),
        accuracy_retention=float(report.get("estimated_accuracy_retention", 0.0)),
        constraints_met=bool(report.get("all_constraints_met", False)),
    )


def run_baseline_episode(env: NeuralTunerEnvironment, model_id: str, difficulty: str) -> EpisodeMetrics:
    reset_obs = env.reset(model_id=model_id, difficulty=difficulty)
    layer_ids = _layer_ids_from_reset(reset_obs.output)
    for idx, layer_id in enumerate(layer_ids[:8]):
        dtype = "INT4" if idx % 2 == 0 else "INT8"
        env.step(NeuralTunerAction(action_type="quantize_layer", layer_id=layer_id, dtype=dtype))
    env.step(NeuralTunerAction(action_type="benchmark"))
    final = env.step(NeuralTunerAction(action_type="submit"))
    report = final.metadata or {}
    st = env.state
    return EpisodeMetrics(
        policy="baseline",
        episode_index=0,
        model_id=model_id,
        difficulty=difficulty,
        final_reward=final.reward,
        done=final.done,
        step_count=st.step_count,
        benchmark_count=st.benchmark_count,
        latency_ms=float(report.get("quantized_latency_ms", 0.0)),
        memory_mb=float(report.get("quantized_memory_mb", 0.0)),
        accuracy_retention=float(report.get("estimated_accuracy_retention", 0.0)),
        constraints_met=bool(report.get("all_constraints_met", False)),
    )


def run_heuristic_episode(env: NeuralTunerEnvironment, model_id: str, difficulty: str) -> EpisodeMetrics:
    reset_obs = env.reset(model_id=model_id, difficulty=difficulty)
    layer_ids = _layer_ids_from_reset(reset_obs.output)
    profiled: Dict[str, float] = {}

    # Keep enough budget for benchmark + submit under MAX_STEPS=20.
    for layer_id in layer_ids[:6]:
        profile_obs = env.step(NeuralTunerAction(action_type="profile_layer", layer_id=layer_id))
        sensitivity = float((profile_obs.metadata or {}).get("sensitivity", 1.0))
        profiled[layer_id] = sensitivity

    for layer_id, sensitivity in profiled.items():
        if sensitivity < 0.10:
            dtype = "INT4"
        elif sensitivity < 0.20:
            dtype = "INT8"
        elif sensitivity < 0.30:
            dtype = "FP16"
        else:
            dtype = "FP32"
        env.step(NeuralTunerAction(action_type="quantize_layer", layer_id=layer_id, dtype=dtype))

    env.step(NeuralTunerAction(action_type="benchmark"))
    final = env.step(NeuralTunerAction(action_type="submit"))
    report = final.metadata or {}
    st = env.state
    return EpisodeMetrics(
        policy="heuristic",
        episode_index=0,
        model_id=model_id,
        difficulty=difficulty,
        final_reward=final.reward,
        done=final.done,
        step_count=st.step_count,
        benchmark_count=st.benchmark_count,
        latency_ms=float(report.get("quantized_latency_ms", 0.0)),
        memory_mb=float(report.get("quantized_memory_mb", 0.0)),
        accuracy_retention=float(report.get("estimated_accuracy_retention", 0.0)),
        constraints_met=bool(report.get("all_constraints_met", False)),
    )


def _write_metrics(metrics: List[EpisodeMetrics], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "episode_metrics.json"
    csv_path = out_dir / "episode_metrics.csv"

    as_rows = [asdict(m) for m in metrics]
    json_path.write_text(json.dumps(as_rows, indent=2))

    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(as_rows[0].keys()))
        writer.writeheader()
        writer.writerows(as_rows)


def run_policy_sweep(
    model_id: str,
    difficulty: str,
    n_random_seeds: int = 10,
) -> List[EpisodeMetrics]:
    """Run all three policies and aggregate random over multiple seeds.

    Returns a list of EpisodeMetrics suitable for writing to disk and plotting.
    Useful for the before/after comparison in the training notebook.
    """
    results: List[EpisodeMetrics] = []
    env = NeuralTunerEnvironment()

    # Multiple random seeds to get a stable pre-training estimate
    for seed in range(n_random_seeds):
        m = run_random_episode(env, model_id, difficulty, seed=seed)
        m.episode_index = seed
        results.append(m)

    # Single deterministic policies for reference ceiling
    results.append(run_baseline_episode(env, model_id, difficulty))
    results.append(run_heuristic_episode(env, model_id, difficulty))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NeuralTuner policy evaluation.")
    parser.add_argument("--model-id", default="inception_v3", help="Model id from model_zoo.")
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--output-dir", default="artifacts/eval", help="Where metrics files are written.")
    parser.add_argument("--n-random", type=int, default=10, help="Number of random-seed episodes to run.")
    args = parser.parse_args()

    metrics = run_policy_sweep(args.model_id, args.difficulty, n_random_seeds=args.n_random)
    _write_metrics(metrics, Path(args.output_dir))
    print(f"Wrote {len(metrics)} episode metrics to {args.output_dir}")
    for m in metrics:
        print(f"  policy={m.policy:12s}  reward={m.final_reward:.4f}  constraints_met={m.constraints_met}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
