# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Scenario definitions for NeuralTuner — 15 scenarios across 5 models × 3 difficulties.

Constraints are calibrated against actual layer sums so targets are achievable
with intelligent quantization but require careful sensitivity management.
"""

import random
from dataclasses import dataclass
from typing import List, Optional

from .model_zoo import get_metadata, list_models
from .simulator import HardwareConstraints


@dataclass
class Scenario:
    name: str
    model_id: str
    difficulty: str  # "easy" | "medium" | "hard"
    constraints: HardwareConstraints
    description: str


def _make_constraints(model_id: str, lat_frac: float, mem_frac: float, min_acc: float) -> HardwareConstraints:
    """Build constraints as fractions of the model's actual baseline."""
    meta = get_metadata(model_id)
    return HardwareConstraints(
        latency_budget_ms=round(meta.base_latency_ms * lat_frac, 1),
        memory_budget_mb=round(meta.base_memory_mb * mem_frac, 1),
        min_accuracy_retention=min_acc,
    )


# ── Easy (need ~40% latency reduction, loose memory, accuracy ≥ 0.85) ──────
# Achievable by uniformly applying INT8 to most non-sensitive layers.
EASY_SCENARIOS: List[Scenario] = [
    Scenario(
        "inception_v3_easy",
        "inception_v3",
        "easy",
        _make_constraints("inception_v3", lat_frac=0.60, mem_frac=0.60, min_acc=0.85),
        "Reduce Inception V3 latency for real-time mobile inference. "
        "Lenient memory budget — focus on latency first.",
    ),
    Scenario(
        "resnet50_easy",
        "resnet50",
        "easy",
        _make_constraints("resnet50", lat_frac=0.60, mem_frac=0.60, min_acc=0.85),
        "Deploy ResNet-50 on an entry-level Snapdragon device. " "Wide memory margin leaves room for exploration.",
    ),
    Scenario(
        "mobilenet_v3_easy",
        "mobilenet_v3",
        "easy",
        _make_constraints("mobilenet_v3", lat_frac=0.65, mem_frac=0.65, min_acc=0.85),
        "Further compress MobileNet V3 for an IoT sensor platform. "
        "Model is already efficient — moderate reductions needed.",
    ),
    Scenario(
        "gm_perception_easy",
        "gm_perception_net",
        "easy",
        _make_constraints("gm_perception_net", lat_frac=0.60, mem_frac=0.65, min_acc=0.85),
        "Optimize GM Perception Net for in-vehicle ADAS compute budget. "
        "Accuracy threshold is relaxed; focus on throughput.",
    ),
    Scenario(
        "bmw_drive_easy",
        "bmw_drive_net",
        "easy",
        _make_constraints("bmw_drive_net", lat_frac=0.62, mem_frac=0.62, min_acc=0.85),
        "Speed up BMW DriveNet for real-time lane and depth fusion. "
        "Moderate constraints allow straightforward INT8 conversion.",
    ),
]

# ── Medium (need ~55% latency reduction, tighter memory, accuracy ≥ 0.90) ──
# Requires strategic use of INT4 on low-sensitivity layers while protecting heads.
MEDIUM_SCENARIOS: List[Scenario] = [
    Scenario(
        "inception_v3_medium",
        "inception_v3",
        "medium",
        _make_constraints("inception_v3", lat_frac=0.45, mem_frac=0.45, min_acc=0.90),
        "Tight latency target for a flagship Snapdragon mobile deployment. "
        "Must use INT4 selectively while keeping classifier accuracy intact.",
    ),
    Scenario(
        "resnet50_medium",
        "resnet50",
        "medium",
        _make_constraints("resnet50", lat_frac=0.45, mem_frac=0.45, min_acc=0.90),
        "ResNet-50 in an edge inference pipeline with strict memory budget. "
        "Backbone layers tolerate INT4; the FC head is sensitive.",
    ),
    Scenario(
        "mobilenet_v3_medium",
        "mobilenet_v3",
        "medium",
        _make_constraints("mobilenet_v3", lat_frac=0.50, mem_frac=0.50, min_acc=0.90),
        "Extreme compression of MobileNet V3 for a wearable sensor. " "Already compact — every MB and ms counts.",
    ),
    Scenario(
        "gm_perception_medium",
        "gm_perception_net",
        "medium",
        _make_constraints("gm_perception_net", lat_frac=0.45, mem_frac=0.48, min_acc=0.88),
        "GM Perception Net on a shared SoC with competing workloads. "
        "Detector heads have high sensitivity — profile before quantizing.",
    ),
    Scenario(
        "bmw_drive_medium",
        "bmw_drive_net",
        "medium",
        _make_constraints("bmw_drive_net", lat_frac=0.45, mem_frac=0.46, min_acc=0.90),
        "BMW DriveNet for a production HW ECU with tight thermal envelope. "
        "Segmentation and lane heads are fragile under INT4.",
    ),
]

# ── Hard (need ~62% latency reduction, tight memory, accuracy ≥ 0.92) ─────
# Forces mixed-precision: INT4 on early/robust layers, FP16 on sensitive heads.
HARD_SCENARIOS: List[Scenario] = [
    Scenario(
        "inception_v3_hard",
        "inception_v3",
        "hard",
        _make_constraints("inception_v3", lat_frac=0.38, mem_frac=0.35, min_acc=0.92),
        "Aggressive Inception V3 optimization for real-time edge video analytics. "
        "Requires INT4 on backbone with careful protection of the classifier head.",
    ),
    Scenario(
        "resnet50_hard",
        "resnet50",
        "hard",
        _make_constraints("resnet50", lat_frac=0.38, mem_frac=0.35, min_acc=0.92),
        "ResNet-50 squeezed into a 2W power envelope. "
        "Memory budget is near the INT8 floor; INT4 needed on early layers.",
    ),
    Scenario(
        "mobilenet_v3_hard",
        "mobilenet_v3",
        "hard",
        _make_constraints("mobilenet_v3", lat_frac=0.40, mem_frac=0.38, min_acc=0.92),
        "Ultra-low-power MobileNet V3 for always-on vision at 1W TDP. "
        "Minimal headroom — every layer choice matters.",
    ),
    Scenario(
        "gm_perception_hard",
        "gm_perception_net",
        "hard",
        _make_constraints("gm_perception_net", lat_frac=0.38, mem_frac=0.36, min_acc=0.90),
        "GM Perception Net on next-gen ADAS SoC with stringent safety accuracy floor. "
        "Sensitive detection heads must stay at FP16; backbone can go INT4.",
    ),
    Scenario(
        "bmw_drive_hard",
        "bmw_drive_net",
        "hard",
        _make_constraints("bmw_drive_net", lat_frac=0.38, mem_frac=0.36, min_acc=0.92),
        "BMW DriveNet on an L4 autonomous ECU with simultaneous segmentation and depth. "
        "All three output heads are highly sensitive — careful mixed-precision required.",
    ),
]

_ALL_SCENARIOS: List[Scenario] = EASY_SCENARIOS + MEDIUM_SCENARIOS + HARD_SCENARIOS

_BY_DIFFICULTY = {
    "easy": EASY_SCENARIOS,
    "medium": MEDIUM_SCENARIOS,
    "hard": HARD_SCENARIOS,
}


def sample_scenario(
    difficulty: Optional[str] = None,
    model_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> Scenario:
    """
    Sample a scenario.  Parameters are all optional:
      - difficulty: "easy" | "medium" | "hard" | None (random)
      - model_id: restrict to a specific model | None (random)
      - seed: random seed for reproducibility
    """
    rng = random.Random(seed)

    pool = _ALL_SCENARIOS if difficulty is None else _BY_DIFFICULTY.get(difficulty, _ALL_SCENARIOS)

    if model_id is not None:
        pool = [s for s in pool if s.model_id == model_id]
        if not pool:
            raise ValueError(
                f"No scenario found for model='{model_id}' difficulty='{difficulty}'. "
                f"Available models: {list_models()}"
            )

    return rng.choice(pool)
