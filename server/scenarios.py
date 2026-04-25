<<<<<<< HEAD
"""Episode scenario definitions for the NeuralTuner environment.

Each scenario pairs a model with hardware constraints at a specific difficulty
level. Easy scenarios have generous targets (FP16 on a few layers is enough).
Hard scenarios require careful INT8 across most layers while avoiding sensitive ones.
=======
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Scenario definitions for NeuralTuner — 15 scenarios across 5 models × 3 difficulties.

Constraints are calibrated against actual layer sums so targets are achievable
with intelligent quantization but require careful sensitivity management.
>>>>>>> master
"""

import random
from dataclasses import dataclass
from typing import List, Optional

<<<<<<< HEAD
from server.simulator import HardwareConstraints
=======
from .model_zoo import get_metadata, list_models
from .simulator import HardwareConstraints
>>>>>>> master


@dataclass
class Scenario:
<<<<<<< HEAD
    """A single optimization episode definition.

    Args:
        model_id: Key into MODEL_REGISTRY.
        difficulty: One of 'easy', 'medium', 'hard'.
        constraints: Hardware deployment constraints.
        description: Human-readable scenario description shown to the agent.
    """

    model_id: str
    difficulty: str
=======
    name: str
    model_id: str
    difficulty: str  # "easy" | "medium" | "hard"
>>>>>>> master
    constraints: HardwareConstraints
    description: str


<<<<<<< HEAD
# ── InceptionV3 scenarios ──────────────────────────────────────────────────

_INCEPTION_EASY = Scenario(
    model_id="inception_v3",
    difficulty="easy",
    constraints=HardwareConstraints(
        memory_budget_mb=100.0,
        latency_target_ms=80.0,
        max_accuracy_drop_pct=2.0,
        baseline_accuracy=77.3,
        hardware_name="Snapdragon 8 Gen 3 (Mobile)",
    ),
    description="Optimize InceptionV3 for a mid-range Snapdragon mobile device.",
)

_INCEPTION_MEDIUM = Scenario(
    model_id="inception_v3",
    difficulty="medium",
    constraints=HardwareConstraints(
        memory_budget_mb=70.0,
        latency_target_ms=62.0,
        max_accuracy_drop_pct=1.5,
        baseline_accuracy=77.3,
        hardware_name="Snapdragon 8 Gen 3 (Mobile)",
    ),
    description="Optimize InceptionV3 for a flagship Snapdragon mobile device with tight memory.",
)

_INCEPTION_HARD = Scenario(
    model_id="inception_v3",
    difficulty="hard",
    constraints=HardwareConstraints(
        memory_budget_mb=55.0,
        latency_target_ms=50.0,
        max_accuracy_drop_pct=0.8,
        baseline_accuracy=77.3,
        hardware_name="Snapdragon X Elite (Laptop)",
    ),
    description="Optimize InceptionV3 for on-device AI on a thin-and-light laptop with strict accuracy.",
)

# ── ResNet50 scenarios ─────────────────────────────────────────────────────

_RESNET_EASY = Scenario(
    model_id="resnet50",
    difficulty="easy",
    constraints=HardwareConstraints(
        memory_budget_mb=55.0,
        latency_target_ms=45.0,
        max_accuracy_drop_pct=2.0,
        baseline_accuracy=76.1,
        hardware_name="Snapdragon 8 Gen 3 (Mobile)",
    ),
    description="Optimize ResNet50 for real-time ADAS perception on a mobile SoC.",
)

_RESNET_MEDIUM = Scenario(
    model_id="resnet50",
    difficulty="medium",
    constraints=HardwareConstraints(
        memory_budget_mb=38.0,
        latency_target_ms=33.0,
        max_accuracy_drop_pct=1.2,
        baseline_accuracy=76.1,
        hardware_name="Snapdragon Ride (Automotive)",
    ),
    description="Optimize ResNet50 for an automotive ADAS SoC with tighter constraints.",
)

_RESNET_HARD = Scenario(
    model_id="resnet50",
    difficulty="hard",
    constraints=HardwareConstraints(
        memory_budget_mb=28.0,
        latency_target_ms=25.0,
        max_accuracy_drop_pct=0.7,
        baseline_accuracy=76.1,
        hardware_name="Snapdragon Ride Elite (Automotive L2+)",
    ),
    description="Optimize ResNet50 for L2+ autonomous driving — safety-critical accuracy required.",
)

# ── MobileNetV3 scenarios ──────────────────────────────────────────────────

_MOBILE_EASY = Scenario(
    model_id="mobilenet_v3",
    difficulty="easy",
    constraints=HardwareConstraints(
        memory_budget_mb=12.0,
        latency_target_ms=13.0,
        max_accuracy_drop_pct=2.0,
        baseline_accuracy=74.0,
        hardware_name="Snapdragon 7s Gen 3 (Mid-range)",
    ),
    description="Optimize MobileNetV3 for a mid-range phone with limited memory.",
)

_MOBILE_MEDIUM = Scenario(
    model_id="mobilenet_v3",
    difficulty="medium",
    constraints=HardwareConstraints(
        memory_budget_mb=8.0,
        latency_target_ms=10.0,
        max_accuracy_drop_pct=1.2,
        baseline_accuracy=74.0,
        hardware_name="Snapdragon 7s Gen 3 (Mid-range)",
    ),
    description="Optimize MobileNetV3 for a budget device — very tight memory and accuracy.",
)

_MOBILE_HARD = Scenario(
    model_id="mobilenet_v3",
    difficulty="hard",
    constraints=HardwareConstraints(
        memory_budget_mb=6.0,
        latency_target_ms=8.0,
        max_accuracy_drop_pct=0.8,
        baseline_accuracy=74.0,
        hardware_name="Snapdragon W5 Gen 3 (Wearable)",
    ),
    description="Optimize MobileNetV3 for a smartwatch — extremely constrained, accuracy must hold.",
)

# ── GMPerceptionNet scenarios ──────────────────────────────────────────────

_GM_EASY = Scenario(
    model_id="gm_perception_net",
    difficulty="easy",
    constraints=HardwareConstraints(
        memory_budget_mb=120.0,
        latency_target_ms=110.0,
        max_accuracy_drop_pct=2.0,
        baseline_accuracy=83.5,
        hardware_name="Snapdragon Ride (Automotive)",
    ),
    description="Optimize GMPerceptionNet for deployment in a production vehicle ADAS system.",
)

_GM_MEDIUM = Scenario(
    model_id="gm_perception_net",
    difficulty="medium",
    constraints=HardwareConstraints(
        memory_budget_mb=80.0,
        latency_target_ms=75.0,
        max_accuracy_drop_pct=1.0,
        baseline_accuracy=83.5,
        hardware_name="Snapdragon Ride Elite (Automotive L2+)",
    ),
    description="Optimize GMPerceptionNet for L2+ driving — safety accuracy constraints apply.",
)

_GM_HARD = Scenario(
    model_id="gm_perception_net",
    difficulty="hard",
    constraints=HardwareConstraints(
        memory_budget_mb=60.0,
        latency_target_ms=55.0,
        max_accuracy_drop_pct=0.5,
        baseline_accuracy=83.5,
        hardware_name="Snapdragon Ride Elite (L3 Autonomous)",
    ),
    description="Optimize GMPerceptionNet for L3 autonomous — near-lossless compression required.",
)

# ── BMWDriveNet scenarios ──────────────────────────────────────────────────

_BMW_EASY = Scenario(
    model_id="bmw_drive_net",
    difficulty="easy",
    constraints=HardwareConstraints(
        memory_budget_mb=75.0,
        latency_target_ms=75.0,
        max_accuracy_drop_pct=2.0,
        baseline_accuracy=81.2,
        hardware_name="Snapdragon Ride (Automotive)",
    ),
    description="Optimize BMWDriveNet for a production BMW ADAS system.",
)

_BMW_MEDIUM = Scenario(
    model_id="bmw_drive_net",
    difficulty="medium",
    constraints=HardwareConstraints(
        memory_budget_mb=52.0,
        latency_target_ms=55.0,
        max_accuracy_drop_pct=1.0,
        baseline_accuracy=81.2,
        hardware_name="Snapdragon Ride Elite (Automotive L2+)",
    ),
    description="Optimize BMWDriveNet for L2+ — lane detection head must stay accurate.",
)

_BMW_HARD = Scenario(
    model_id="bmw_drive_net",
    difficulty="hard",
    constraints=HardwareConstraints(
        memory_budget_mb=38.0,
        latency_target_ms=40.0,
        max_accuracy_drop_pct=0.6,
        baseline_accuracy=81.2,
        hardware_name="Snapdragon Ride Elite (L3 Autonomous)",
    ),
    description="Optimize BMWDriveNet for L3 autonomous driving — strict safety requirements.",
)


# ── Curriculum pools ───────────────────────────────────────────────────────

EASY_SCENARIOS: List[Scenario] = [
    _INCEPTION_EASY,
    _RESNET_EASY,
    _MOBILE_EASY,
    _GM_EASY,
    _BMW_EASY,
]

MEDIUM_SCENARIOS: List[Scenario] = [
    _INCEPTION_MEDIUM,
    _RESNET_MEDIUM,
    _MOBILE_MEDIUM,
    _GM_MEDIUM,
    _BMW_MEDIUM,
]

HARD_SCENARIOS: List[Scenario] = [
    _INCEPTION_HARD,
    _RESNET_HARD,
    _MOBILE_HARD,
    _GM_HARD,
    _BMW_HARD,
]

ALL_SCENARIOS: List[Scenario] = EASY_SCENARIOS + MEDIUM_SCENARIOS + HARD_SCENARIOS

_POOL_MAP = {
    "easy": EASY_SCENARIOS,
    "medium": MEDIUM_SCENARIOS,
    "hard": HARD_SCENARIOS,
    "all": ALL_SCENARIOS,
=======
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
>>>>>>> master
}


def sample_scenario(
<<<<<<< HEAD
    difficulty: str = "easy",
    model_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> Scenario:
    """Sample a random scenario from the given difficulty pool.

    Args:
        difficulty: One of 'easy', 'medium', 'hard', 'all'.
        model_id: If provided, restricts sampling to this model only.
        seed: Optional random seed for reproducibility.

    Returns:
        A Scenario object ready for use in an episode.

    Raises:
        ValueError: If difficulty is unknown or no matching scenarios exist.
    """
    if difficulty not in _POOL_MAP:
        raise ValueError(f"Unknown difficulty '{difficulty}'. Use: {list(_POOL_MAP.keys())}")

    pool = _POOL_MAP[difficulty]
=======
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
>>>>>>> master

    if model_id is not None:
        pool = [s for s in pool if s.model_id == model_id]
        if not pool:
<<<<<<< HEAD
            raise ValueError(f"No scenarios found for model_id='{model_id}' at difficulty='{difficulty}'")

    rng = random.Random(seed)
=======
            raise ValueError(
                f"No scenario found for model='{model_id}' difficulty='{difficulty}'. "
                f"Available models: {list_models()}"
            )

>>>>>>> master
    return rng.choice(pool)
