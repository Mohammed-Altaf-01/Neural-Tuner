# NeuralTuner: Hardware-Aware RL Environment for LLMs

NeuralTuner trains an LLM to behave like a hardware optimization engineer: profile layers, choose per-layer quantization (`FP16`/`INT8`/`INT4`), validate constraints, and submit a deployable configuration.

**[🤗 HuggingFace Space (live demo)](https://huggingface.co/spaces/Mohammed-Altaf/Neural-Tuner)**
| **[📓 Training Notebook](neural_tuner_trl_mac.ipynb)**

## Why This Matters

Deploying neural networks on edge devices requires balancing latency, memory, and accuracy. NeuralTuner converts this expert-heavy workflow into an OpenEnv task with measurable RL rewards.

Every AI model in your phone or car goes through this process manually. NeuralTuner trains an LLM to do it automatically — profile layer sensitivity, apply the right quantization dtype per layer, validate hardware constraints, and submit a deployable configuration.

## Environment Loop

1. `reset()` gives model metadata, constraints, and layer table (sensitivities hidden).
2. `profile_layer(layer_id)` reveals sensitivity for strategic investigation.
3. `quantize_layer(layer_id, dtype)` applies a quantization decision (`FP32`/`FP16`/`INT8`/`INT4`).
4. `prune_layer(layer_id, sparsity)` applies structured pruning (`LOW`/`MEDIUM`/`HIGH`).
5. `benchmark()` returns latency/memory/accuracy estimates and projected reward.
6. `submit()` finalizes and returns episode reward.

Core files:
- `server/neural_tuner_env_environment.py`
- `server/simulator.py`
- `server/scenarios.py`
- `server/model_zoo.py`
- `models.py`

## Model Zoo

Five models spanning mobile and automotive domains:

| Model | Params | Baseline | Domain |
|-------|--------|----------|--------|
| InceptionV3 | 47M | 175 ms, 186 MB | Mobile vision |
| ResNet-50 | 25M | 88 ms, 93 MB | ADAS backbone |
| MobileNet V3 | 5.4M | 24 ms, 21 MB | Mobile edge |
| GM Perception Net | 58M | 210 ms, 232 MB | Automotive detection |
| BMW DriveNet | 35M | 145 ms, 140 MB | Autonomous segmentation |

## Reward Design

Reward combines four parts:
- **Latency improvement** (0–0.40, continuous) — proportional to % reduction from baseline
- **Memory fit** (0 or 0.30, binary) — hard constraint: model must fit on device
- **Accuracy retention** (0–0.20, continuous) — rewards staying above minimum threshold
- **Efficiency bonus** (0 or 0.10) — all three constraints met simultaneously

This multi-component design discourages reward hacking (for example, blindly INT4-quantizing everything collapses accuracy → reward ≈ 0.1).

## Training Results

![Pre-training reward distribution](artifacts/plots/pre_training_reward_distribution.png)
*Pre-training: random policy reward distribution vs oracle ceiling — inception\_v3 medium, n=20 seeds*

![Post-training evaluation](artifacts/plots/post_training_eval.png)
*Post-training: reward progression over GRPO training steps vs random baseline and oracle ceiling*

## Episode Trace: Random vs Heuristic Agent

The behavioral difference between an untrained (random) agent and a heuristic (profile-first) agent on `inception_v3` medium:

### Random Agent (no profiling)
**Step 1:** `quantize_layer(conv_stem, FP32)` → _QUANTIZE: conv\_stem  FP32 → FP32 WARNING: Layer not profiled._
**Step 2:** `quantize_layer(conv_bn_1, FP32)` → _WARNING: Layer not profiled._
**Step 3:** `quantize_layer(mixed_3a, INT8)` → _WARNING: Layer not profiled._
**Step 4:** `quantize_layer(mixed_4a, FP16)` → _WARNING: Layer not profiled._
**Step 5:** `quantize_layer(mixed_5a, FP16)` → _WARNING: Layer not profiled._
**Step 6:** `quantize_layer(mixed_6a, FP16)` → _WARNING: Layer not profiled._
**Step 7:** `quantize_layer(mixed_7a, FP32)` → _WARNING: Layer not profiled._
**Step 8:** `quantize_layer(avg_pool, FP32)` → _WARNING: Layer not profiled._
**Step 9:** `quantize_layer(dropout, INT4)` → _WARNING: Layer not profiled._
**Step 10:** `quantize_layer(fc_classifier, FP32)` → _WARNING: Layer not profiled._
**Step 11:** `benchmark()`
**Step 12:** `submit()`

**Final reward: 0.3037** | constraints\_met=False

### Heuristic Agent (profile-first)
**Step 1:** `profile_layer(conv_stem)` → _sensitivity=0.040 [low risk — INT4 safe]_
**Step 2:** `profile_layer(conv_bn_1)` → _sensitivity=0.020 [low risk]_
**Step 3:** `profile_layer(mixed_3a)` → _sensitivity=0.080 [low risk]_
**Step 4:** `profile_layer(mixed_4a)` → _sensitivity=0.120 [medium risk — INT8 preferred]_
**Step 5:** `profile_layer(mixed_5a)` → _sensitivity=0.090 [low risk]_
**Step 6:** `profile_layer(mixed_6a)` → _sensitivity=0.150 [medium risk]_
**Step 7:** `quantize_layer(conv_stem, INT4)`
**Step 8:** `quantize_layer(conv_bn_1, INT4)`
**Step 9:** `quantize_layer(mixed_3a, INT4)`
**Step 10:** `quantize_layer(mixed_4a, INT8)` ← protects medium-sensitivity layer
**Step 11:** `quantize_layer(mixed_5a, INT4)`
**Step 12:** `quantize_layer(mixed_6a, INT8)` ← protects medium-sensitivity layer
**Step 13:** `benchmark()`
**Step 14:** `submit()`

**Final reward: 0.6428** | constraints\_met=False

The trained RL agent learns to behave like the heuristic: profile first, then make sensitivity-aware dtype decisions. The goal of GRPO training is to bring the model's reward from ~0.47 (random) toward ~0.79 (oracle ceiling).

## Quick Start

### Install

```bash
uv sync
```

Optional notebook/training dependencies:

```bash
uv sync --extra training
```

### Run Server

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### Generate Baseline vs Heuristic Eval Artifacts

```bash
python rollout_eval.py --model-id inception_v3 --difficulty medium --output-dir artifacts/eval
```

Generate a side-by-side episode trace (random vs heuristic):

```bash
python rollout_eval.py --trace --model-id inception_v3 --difficulty medium
```

Outputs:
- `artifacts/eval/episode_metrics.json`
- `artifacts/eval/episode_metrics.csv`
- `artifacts/eval/episode_trace.md`

### Run Inference-Only Action Prediction

```bash
HF_TOKEN=... python inference.py --mode hf_api --observation "..." --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

`inference.py` is intentionally inference-only. Training and reward improvement demos belong in the notebook.

## Training Notebook (TRL)

Use `neural_tuner_trl_mac.ipynb` for:
- environment smoke test
- baseline metric collection (random policy n=20 + oracle ceiling)
- TRL GRPO training with curriculum learning (easy → medium → hard)
- reward logging and plot export (pre-training distribution + post-training trajectory)

## Tests

```bash
pytest -q
```

Current tests validate:
- benchmark budget enforcement
- invalid layer handling
- submission terminal behavior
- reward sanity for safe vs over-aggressive quantization

## OpenEnv Deployment

Manifest: `openenv.yaml`
App entrypoint: `server.app:app`

Push to Hugging Face Space:

```bash
openenv push
```

## Submission Checklist

- [x] Embed baseline vs trained reward plots in README.
- [x] Add notebook outputs/screenshots and short captions.
- [ ] Link video demo and blog post.
- [x] Include one side-by-side episode trace (random vs heuristic).
