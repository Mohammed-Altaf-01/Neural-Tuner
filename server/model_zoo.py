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
}


def get_layers(model_id: str) -> List[LayerProfile]:
    if model_id not in _MODEL_LAYERS:
        raise ValueError(f"Unknown model: '{model_id}'. Available: {list(_MODEL_LAYERS)}")
    return list(_MODEL_LAYERS[model_id])


def get_metadata(model_id: str) -> ModelMetadata:
    if model_id not in _MODEL_METADATA:
        raise ValueError(f"Unknown model: '{model_id}'. Available: {list(_MODEL_METADATA)}")
    return _MODEL_METADATA[model_id]


def list_models() -> List[str]:
    return list(_MODEL_LAYERS.keys())
