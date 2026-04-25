from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in: {path}")
    return rows


def split_scenarios(
    scenarios: list[dict],
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    if not 0.1 <= train_fraction <= 0.95:
        raise ValueError("train_fraction must be in [0.1, 0.95]")
    if len(scenarios) < 4:
        raise ValueError("Need at least 4 scenarios to split train/eval")

    shuffled = list(scenarios)
    random.Random(seed).shuffle(shuffled)
    cut = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * train_fraction))))
    train_rows = shuffled[:cut]
    eval_rows = shuffled[cut:]

    train_ids = {row["id"] for row in train_rows}
    eval_ids = {row["id"] for row in eval_rows}
    overlap = train_ids & eval_ids
    if overlap:
        raise ValueError(f"Scenario leakage between train/eval split: {sorted(overlap)}")
    return train_rows, eval_rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def ensure_metric_keys(metrics: dict, required_keys: list[str]) -> None:
    missing = [key for key in required_keys if key not in metrics]
    if missing:
        raise ValueError(f"Missing metric keys: {missing}")


def collect_rewards_with_autosubmit(environments) -> list[float]:
    rewards: list[float] = []
    for env in environments:
        if not getattr(env, "done", False):
            env.submit()
        rewards.append(float(getattr(env, "reward", 0.0)))
    return rewards
