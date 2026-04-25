"""Episode scenario definitions for the NeuralTuner environment.

Each scenario pairs a model with hardware constraints at a specific difficulty
level. Easy scenarios have generous targets (FP16 on a few layers is enough).
Hard scenarios require careful INT8 across most layers while avoiding sensitive ones.
"""

import random
from dataclasses import dataclass
from typing import List, Optional

from server.simulator import HardwareConstraints


@dataclass
class Scenario:
    """A single optimization episode definition.

    Args:
        model_id: Key into MODEL_REGISTRY.
        difficulty: One of 'easy', 'medium', 'hard'.
        constraints: Hardware deployment constraints.
        description: Human-readable scenario description shown to the agent.
    """

    model_id: str
    difficulty: str
    constraints: HardwareConstraints
    description: str


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
}


def sample_scenario(
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

    if model_id is not None:
        pool = [s for s in pool if s.model_id == model_id]
        if not pool:
            raise ValueError(f"No scenarios found for model_id='{model_id}' at difficulty='{difficulty}'")

    rng = random.Random(seed)
    return rng.choice(pool)
