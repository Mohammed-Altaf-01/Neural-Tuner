# NeuralTuner: Hardware-Aware RL Environment for LLMs

NeuralTuner trains an LLM to behave like a hardware optimization engineer: profile layers, choose per-layer quantization (`FP16`/`INT8`/`INT4`), validate constraints, and submit a deployable configuration.

## Why This Matters

Deploying neural networks on edge devices requires balancing latency, memory, and accuracy. NeuralTuner converts this expert-heavy workflow into an OpenEnv task with measurable RL rewards.

## Environment Loop

1. `reset()` gives model metadata, constraints, and layer table (sensitivities hidden).
2. `profile_layer(layer_id)` reveals sensitivity for strategic investigation.
3. `quantize_layer(layer_id, dtype)` applies a quantization decision.
4. `benchmark()` returns latency/memory/accuracy estimates and projected reward.
5. `submit()` finalizes and returns episode reward.

Core files:
- `server/neural_tuner_env_environment.py`
- `server/simulator.py`
- `server/scenarios.py`
- `server/model_zoo.py`
- `models.py`

## Reward Design

Reward combines four parts:
- latency improvement (continuous, capped)
- memory fit (binary)
- accuracy retention (continuous when above threshold)
- all-constraints efficiency bonus

This design discourages reward hacking (for example, blindly INT4-quantizing everything).

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

Outputs:
- `artifacts/eval/episode_metrics.json`
- `artifacts/eval/episode_metrics.csv`

### Run Inference-Only Action Prediction

```bash
HF_TOKEN=... python inference.py --mode hf_api --observation "..." --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

`inference.py` is intentionally inference-only. Training and reward improvement demos belong in the notebook.

## Training Notebook (TRL)

Use `neural_tuner_trl_mac.ipynb` for:
- environment smoke test
- baseline metric collection
- TRL GRPO setup skeleton with DeepSeek-R1 distill model
- reward logging and plot export

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

- [ ] Embed baseline vs trained reward plots in README.
- [ ] Add notebook outputs/screenshots and short captions.
- [ ] Link video demo, blog post, and HF Space.
- [ ] Include one side-by-side episode trace (random vs trained).
