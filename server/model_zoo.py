<<<<<<< HEAD
"""Neural network model profiles for the NeuralTuner environment.

Each model is a list of LayerProfile objects with realistic FLOPs, parameter
counts, latency, and memory numbers representative of inference on Snapdragon
edge hardware. Sensitivity scores reflect real quantization behavior:
early conv layers are generally safe (< 0.10), while classifier and detection
heads are fragile (> 0.20).
"""

from typing import Dict, List

from server.simulator import LayerProfile


def _inception_v3() -> List[LayerProfile]:
    """InceptionV3 — image classification backbone (47M params).

    Classic vision model used heavily in mobile AI.
    Early inception blocks are very safe for INT8; the final classifier is fragile.
    Baseline: 175ms, 186MB, accuracy 77.3%.
    """
    return [
        LayerProfile("conv_stem",    "Conv2d",        0.37, 0.50,  5.0,  2.0,  0.02),
        LayerProfile("inception_a1", "InceptionBlock",1.49, 3.00, 14.0, 12.0,  0.04),
        LayerProfile("inception_a2", "InceptionBlock",1.49, 3.00, 13.0, 12.0,  0.04),
        LayerProfile("inception_a3", "InceptionBlock",1.49, 3.00, 13.0, 12.0,  0.05),
        LayerProfile("inception_b",  "InceptionBlock",1.86, 5.00, 18.0, 20.0,  0.07),
        LayerProfile("inception_c1", "InceptionBlock",2.08, 4.00, 20.0, 16.0,  0.08),
        LayerProfile("inception_c2", "InceptionBlock",2.08, 4.00, 19.0, 16.0,  0.08),
        LayerProfile("inception_d",  "InceptionBlock",2.31, 6.00, 22.0, 24.0,  0.10),
        LayerProfile("inception_e1", "InceptionBlock",2.57, 8.00, 24.0, 32.0,  0.11),
        LayerProfile("inception_e2", "InceptionBlock",2.57, 8.00, 23.0, 32.0,  0.11),
        LayerProfile("classifier",   "Linear",        0.08, 2.05,  4.0,  8.2,  0.25),
    ]


def _resnet50() -> List[LayerProfile]:
    """ResNet50 — ADAS perception backbone (25M params).

    Standard residual network widely deployed in automotive computer vision.
    Residual blocks are generally INT8-safe; layer4 and fc become sensitive.
    Baseline: 88ms, 100MB, accuracy 76.1%.
    """
    return [
        LayerProfile("conv1",      "Conv2d",   0.23, 0.03,  3.0,  0.1,  0.01),
        LayerProfile("layer1_b0",  "ResBlock", 0.57, 0.74,  6.5,  3.0,  0.03),
        LayerProfile("layer1_b1",  "ResBlock", 0.57, 0.74,  6.0,  3.0,  0.03),
        LayerProfile("layer2_b0",  "ResBlock", 1.25, 1.67, 12.0,  6.7,  0.05),
        LayerProfile("layer2_b1",  "ResBlock", 1.25, 1.67, 11.0,  6.7,  0.05),
        LayerProfile("layer3_b0",  "ResBlock", 1.55, 3.74, 10.0, 15.0,  0.07),
        LayerProfile("layer3_b1",  "ResBlock", 1.55, 3.74,  9.5, 15.0,  0.08),
        LayerProfile("layer4_b0",  "ResBlock", 2.10, 4.98,  4.5, 19.9,  0.12),
        LayerProfile("layer4_b1",  "ResBlock", 2.10, 4.98,  4.0, 19.9,  0.15),
        LayerProfile("fc",         "Linear",   0.04, 1.00,  2.0,  4.0,  0.20),
    ]


def _mobilenet_v3() -> List[LayerProfile]:
    """MobileNetV3-Large — already-optimized mobile backbone (5.4M params).

    Designed for mobile inference, so INT8 quickly degrades quality.
    The challenge here is maintaining accuracy under tight constraints.
    Baseline: 24ms, 21MB, accuracy 74.0%.
    """
    return [
        LayerProfile("features_0",   "Conv2d",      0.04, 0.03,  1.5,  0.1,  0.03),
        LayerProfile("features_1_3", "MobileBlock", 0.11, 0.36,  2.5,  1.4,  0.06),
        LayerProfile("features_4_6", "MobileBlock", 0.26, 0.74,  4.0,  3.0,  0.09),
        LayerProfile("features_7_9", "MobileBlock", 0.35, 1.28,  5.5,  5.1,  0.13),
        LayerProfile("features_10",  "MobileBlock", 0.42, 1.95,  6.0,  7.8,  0.15),
        LayerProfile("features_11",  "MobileBlock", 0.18, 0.91,  3.0,  3.6,  0.10),
        LayerProfile("classifier",   "Linear",      0.01, 0.07,  1.5,  0.3,  0.22),
    ]


def _gm_perception_net() -> List[LayerProfile]:
    """GMPerceptionNet — synthetic automotive multi-task model (58M params).

    Represents the class of multi-task ADAS models (detection + segmentation)
    deployed in production vehicles. Safety-critical — accuracy thresholds
    are strict. The segmentation decoder is highly sensitive.
    Baseline: 210ms, 232MB, accuracy 83.5%.
    """
    return [
        LayerProfile("backbone_stem",   "Conv2d",   0.52, 0.75,  7.0,  3.0,  0.02),
        LayerProfile("backbone_stage1", "ResBlock", 2.24, 4.50, 20.0, 18.0,  0.04),
        LayerProfile("backbone_stage2", "ResBlock", 3.84, 8.50, 32.0, 34.0,  0.05),
        LayerProfile("backbone_stage3", "ResBlock", 5.12,12.00, 40.0, 48.0,  0.06),
        LayerProfile("fpn_lateral",     "Conv2d",   1.24, 2.10, 12.0,  8.4,  0.08),
        LayerProfile("fpn_output",      "Conv2d",   0.86, 1.50,  9.0,  6.0,  0.09),
        LayerProfile("detection_head",  "Conv2d",   0.62, 2.50,  7.0, 10.0,  0.18),
        LayerProfile("seg_decoder_a",   "Conv2d",   1.42, 5.50, 15.0, 22.0,  0.22),
        LayerProfile("seg_decoder_b",   "Conv2d",   0.86, 3.50, 10.0, 14.0,  0.28),
        LayerProfile("seg_output",      "Conv2d",   0.32, 0.50,  4.0,  2.0,  0.32),
        LayerProfile("det_output",      "Linear",   0.08, 0.75,  2.0,  3.0,  0.30),
        LayerProfile("aux_branch",      "Conv2d",   0.62, 2.50,  7.0, 10.0,  0.14),
    ]


def _bmw_drive_net() -> List[LayerProfile]:
    """BMWDriveNet — synthetic automotive lane and object perception (35M params).

    Represents a dedicated driving perception backbone. Multi-scale feature
    extraction with a sensitive lane-detection head that must stay at FP16.
    Baseline: 145ms, 140MB, accuracy 81.2%.
    """
    return [
        LayerProfile("encoder_stem",  "Conv2d",     0.45, 0.60,  6.0,  2.4,  0.02),
        LayerProfile("encoder_b1",    "ResBlock",   1.80, 3.20, 16.0, 12.8,  0.04),
        LayerProfile("encoder_b2",    "ResBlock",   2.60, 5.80, 22.0, 23.2,  0.05),
        LayerProfile("encoder_b3",    "ResBlock",   3.40, 8.20, 28.0, 32.8,  0.06),
        LayerProfile("neck_fpn",      "Conv2d",     1.10, 2.00, 11.0,  8.0,  0.08),
        LayerProfile("neck_pan",      "Conv2d",     0.90, 1.80,  9.0,  7.2,  0.09),
        LayerProfile("obj_head",      "Conv2d",     0.60, 2.40,  7.0,  9.6,  0.16),
        LayerProfile("lane_head",     "Conv2d",     0.55, 2.20,  6.0,  8.8,  0.26),
        LayerProfile("depth_head",    "Conv2d",     0.40, 1.60,  5.0,  6.4,  0.20),
        LayerProfile("fusion_output", "Linear",     0.20, 0.80,  3.0,  3.2,  0.28),
        LayerProfile("cls_output",    "Linear",     0.10, 0.60,  2.0,  2.4,  0.22),
    ]


MODEL_REGISTRY: Dict[str, callable] = {
    "inception_v3": _inception_v3,
    "resnet50": _resnet50,
    "mobilenet_v3": _mobilenet_v3,
    "gm_perception_net": _gm_perception_net,
    "bmw_drive_net": _bmw_drive_net,
}

MODEL_METADATA: Dict[str, Dict] = {
    "inception_v3": {
        "display_name": "InceptionV3",
        "task": "Image Classification (ImageNet)",
        "params_m": 47.0,
        "baseline_latency_ms": 175.0,
        "baseline_memory_mb": 186.2,
        "baseline_accuracy": 77.3,
        "domain": "Mobile Vision",
    },
    "resnet50": {
        "display_name": "ResNet50",
        "task": "Object Detection / ADAS Backbone",
        "params_m": 25.0,
        "baseline_latency_ms": 88.0,
        "baseline_memory_mb": 93.3,
        "baseline_accuracy": 76.1,
        "domain": "Automotive ADAS",
    },
    "mobilenet_v3": {
        "display_name": "MobileNetV3-Large",
        "task": "Image Classification (Mobile)",
        "params_m": 5.4,
        "baseline_latency_ms": 24.0,
        "baseline_memory_mb": 21.3,
        "baseline_accuracy": 74.0,
        "domain": "Mobile Vision",
    },
    "gm_perception_net": {
        "display_name": "GMPerceptionNet",
        "task": "Multi-Task Perception (Detection + Segmentation)",
        "params_m": 58.0,
        "baseline_latency_ms": 210.0,
        "baseline_memory_mb": 232.0,
        "baseline_accuracy": 83.5,
        "domain": "Automotive ADAS (General Motors style)",
    },
    "bmw_drive_net": {
        "display_name": "BMWDriveNet",
        "task": "Lane Detection + Object Perception",
        "params_m": 35.0,
        "baseline_latency_ms": 145.0,
        "baseline_memory_mb": 140.0,
        "baseline_accuracy": 81.2,
        "domain": "Automotive Perception (BMW style)",
    },
=======
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Model zoo for NeuralTuner — layer profiles for 5 neural networks spanning
vision, automotive perception, and autonomous driving.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class LayerProfile:
    layer_id: str
    layer_type: str
    base_latency_ms: float
    base_memory_mb: float
    sensitivity: float  # 0.0–1.0; hidden from agent until profiled
    param_count: int


@dataclass
class ModelMetadata:
    model_id: str
    name: str
    total_params: int
    base_latency_ms: float
    base_memory_mb: float
    description: str


# ── Inception V3 (47M, 175 ms, 186 MB) ─────────────────────────────────────
_INCEPTION_V3_LAYERS: List[LayerProfile] = [
    LayerProfile("conv_stem", "conv2d", 10.7, 11.2, 0.04, 864),
    LayerProfile("conv_bn_1", "bn_relu", 2.7, 2.8, 0.02, 128),
    LayerProfile("mixed_3a", "inception_block", 24.0, 25.8, 0.08, 2_097_152),
    LayerProfile("mixed_4a", "inception_block", 28.9, 31.8, 0.12, 3_145_728),
    LayerProfile("mixed_5a", "inception_block", 25.7, 27.5, 0.09, 2_621_440),
    LayerProfile("mixed_6a", "inception_block", 33.0, 35.0, 0.15, 4_194_304),
    LayerProfile("mixed_7a", "inception_block", 37.1, 38.0, 0.18, 5_242_880),
    LayerProfile("avg_pool", "global_avg_pool", 5.6, 5.6, 0.03, 0),
    LayerProfile("dropout", "dropout", 1.6, 1.6, 0.01, 0),
    LayerProfile("fc_classifier", "linear", 5.8, 6.8, 0.28, 2_048_000),
]

# ── ResNet-50 (25M, 88 ms, 93 MB) ──────────────────────────────────────────
_RESNET50_LAYERS: List[LayerProfile] = [
    LayerProfile("conv1", "conv2d", 6.4, 6.4, 0.05, 9_408),
    LayerProfile("bn1", "bn_relu", 1.8, 1.8, 0.02, 128),
    LayerProfile("layer1", "residual_block", 12.3, 12.3, 0.07, 215_808),
    LayerProfile("layer2", "residual_block", 18.7, 19.2, 0.10, 1_218_048),
    LayerProfile("layer3", "residual_block", 24.6, 25.1, 0.14, 7_077_888),
    LayerProfile("layer4", "residual_block", 16.4, 16.6, 0.22, 14_964_736),
    LayerProfile("avg_pool", "global_avg_pool", 2.8, 2.8, 0.02, 0),
    LayerProfile("fc", "linear", 5.2, 8.9, 0.30, 2_048_000),
]

# ── MobileNet V3 Large (5.4M, 24 ms, 21 MB) ────────────────────────────────
_MOBILENET_V3_LAYERS: List[LayerProfile] = [
    LayerProfile("features_0", "conv_bn_hardswish", 2.1, 1.8, 0.06, 864),
    LayerProfile("features_1_3", "depthwise_separable", 3.4, 2.9, 0.08, 144_768),
    LayerProfile("features_4_6", "inverted_residual_se", 4.8, 4.1, 0.12, 576_384),
    LayerProfile("features_7_10", "inverted_residual_se", 5.9, 5.0, 0.16, 1_792_000),
    LayerProfile("features_11_12", "inverted_residual_se", 3.2, 2.7, 0.10, 921_600),
    LayerProfile("conv_head", "conv_bn_hardswish", 2.8, 2.4, 0.07, 720_000),
    LayerProfile("classifier", "linear", 1.8, 2.1, 0.25, 1_280_000),
]

# ── GM Perception Net (58M, 210 ms, 232 MB) — automotive object detection ──
_GM_PERCEPTION_LAYERS: List[LayerProfile] = [
    LayerProfile("backbone_stem", "conv2d", 10.4, 12.1, 0.04, 3_456),
    LayerProfile("backbone_stage1", "residual_block", 22.8, 26.6, 0.06, 589_824),
    LayerProfile("backbone_stage2", "residual_block", 31.5, 36.8, 0.10, 4_718_592),
    LayerProfile("backbone_stage3", "residual_block", 38.7, 45.2, 0.14, 18_874_368),
    LayerProfile("fpn_lateral", "conv2d", 18.3, 21.4, 0.08, 1_048_576),
    LayerProfile("fpn_output", "conv2d", 21.6, 25.2, 0.11, 2_097_152),
    LayerProfile("rpn_head", "conv2d", 24.9, 29.1, 0.20, 4_718_592),
    LayerProfile("roi_head", "linear", 18.4, 21.5, 0.32, 10_485_760),
    LayerProfile("bbox_predictor", "linear", 7.2, 8.4, 0.35, 2_097_152),
    LayerProfile("cls_predictor", "linear", 16.2, 5.6, 0.28, 917_504),
    LayerProfile("nms_post", "custom", 0.2, 0.1, 0.01, 0),
]

# ── BMW DriveNet (35M, 145 ms, 140 MB) — autonomous driving segmentation ───
_BMW_DRIVE_LAYERS: List[LayerProfile] = [
    LayerProfile("encoder_conv1", "conv2d", 9.8, 9.5, 0.05, 1_728),
    LayerProfile("encoder_res1", "residual_block", 16.4, 15.8, 0.08, 1_179_648),
    LayerProfile("encoder_res2", "residual_block", 22.1, 21.4, 0.12, 9_437_184),
    LayerProfile("encoder_res3", "residual_block", 26.8, 25.9, 0.16, 25_165_824),
    LayerProfile("aspp_module", "atrous_conv", 18.5, 17.9, 0.11, 4_194_304),
    LayerProfile("decoder_up1", "transposed_conv", 14.2, 13.7, 0.09, 2_097_152),
    LayerProfile("decoder_up2", "transposed_conv", 12.6, 12.2, 0.07, 1_048_576),
    LayerProfile("seg_head", "conv2d", 8.4, 8.1, 0.22, 2_359_296),
    LayerProfile("depth_head", "conv2d", 7.3, 7.0, 0.26, 1_572_864),
    LayerProfile("lane_head", "conv2d", 8.9, 8.6, 0.30, 2_097_152),
]

_MODEL_LAYERS: Dict[str, List[LayerProfile]] = {
    "inception_v3": _INCEPTION_V3_LAYERS,
    "resnet50": _RESNET50_LAYERS,
    "mobilenet_v3": _MOBILENET_V3_LAYERS,
    "gm_perception_net": _GM_PERCEPTION_LAYERS,
    "bmw_drive_net": _BMW_DRIVE_LAYERS,
}

_MODEL_METADATA: Dict[str, ModelMetadata] = {
    "inception_v3": ModelMetadata(
        "inception_v3",
        "Inception V3",
        47_000_000,
        round(sum(l.base_latency_ms for l in _INCEPTION_V3_LAYERS), 1),
        round(sum(l.base_memory_mb for l in _INCEPTION_V3_LAYERS), 1),
        "Google Inception V3 image classifier (47M params).",
    ),
    "resnet50": ModelMetadata(
        "resnet50",
        "ResNet-50",
        25_000_000,
        round(sum(l.base_latency_ms for l in _RESNET50_LAYERS), 1),
        round(sum(l.base_memory_mb for l in _RESNET50_LAYERS), 1),
        "Microsoft ResNet-50 residual network for image classification (25M params).",
    ),
    "mobilenet_v3": ModelMetadata(
        "mobilenet_v3",
        "MobileNet V3 Large",
        5_400_000,
        round(sum(l.base_latency_ms for l in _MOBILENET_V3_LAYERS), 1),
        round(sum(l.base_memory_mb for l in _MOBILENET_V3_LAYERS), 1),
        "Google MobileNet V3 designed for edge and mobile deployment (5.4M params).",
    ),
    "gm_perception_net": ModelMetadata(
        "gm_perception_net",
        "GM Perception Net",
        58_000_000,
        round(sum(l.base_latency_ms for l in _GM_PERCEPTION_LAYERS), 1),
        round(sum(l.base_memory_mb for l in _GM_PERCEPTION_LAYERS), 1),
        "General Motors automotive perception model for real-time object detection (58M params).",
    ),
    "bmw_drive_net": ModelMetadata(
        "bmw_drive_net",
        "BMW DriveNet",
        35_000_000,
        round(sum(l.base_latency_ms for l in _BMW_DRIVE_LAYERS), 1),
        round(sum(l.base_memory_mb for l in _BMW_DRIVE_LAYERS), 1),
        "BMW autonomous driving model for segmentation and depth estimation (35M params).",
    ),
>>>>>>> master
}


def get_layers(model_id: str) -> List[LayerProfile]:
<<<<<<< HEAD
    """Return the list of LayerProfile objects for a given model.

    Args:
        model_id: One of the keys in MODEL_REGISTRY.

    Returns:
        List of LayerProfile objects.

    Raises:
        ValueError: If model_id is not in MODEL_REGISTRY.
    """
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_id '{model_id}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_id]()


def get_metadata(model_id: str) -> Dict:
    """Return display metadata for a given model.

    Args:
        model_id: One of the keys in MODEL_METADATA.

    Returns:
        Dict with display_name, task, params_m, baseline stats, and domain.
    """
    return MODEL_METADATA[model_id]
=======
    if model_id not in _MODEL_LAYERS:
        raise ValueError(f"Unknown model: '{model_id}'. Available: {list(_MODEL_LAYERS)}")
    return list(_MODEL_LAYERS[model_id])


def get_metadata(model_id: str) -> ModelMetadata:
    if model_id not in _MODEL_METADATA:
        raise ValueError(f"Unknown model: '{model_id}'. Available: {list(_MODEL_METADATA)}")
    return _MODEL_METADATA[model_id]


def list_models() -> List[str]:
    return list(_MODEL_LAYERS.keys())
>>>>>>> master
