# NeuralTuner Environment

An OpenEnv-compatible RL environment that trains LLMs to optimize neural networks for Qualcomm Snapdragon edge hardware via per-layer quantization and structured pruning.

> **Full write-up:** [BLOG.md](BLOG.md)  |  **Live demo:** [HuggingFace Space](https://huggingface.co/spaces/Mohammed-Altaf/Neural-Tuner)

---

## Overview

NeuralTuner wraps a hardware simulator as a multi-step RL environment. An LLM agent receives a model's layer table and a set of Snapdragon HTP constraints (latency budget, memory budget, minimum accuracy), then issues tool calls to profile layers, apply quantization dtypes (`FP32`/`FP16`/`INT8`/`INT4`), apply structured pruning (`LOW`/`MEDIUM`/`HIGH`), benchmark the current plan, and submit a final configuration for scoring.

The environment supports 19 scenarios across 5 models and 3 difficulty tiers.

---

## Requirements

- Python ≥ 3.10
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

---

## Installation

```bash
# Clone and install core dependencies
uv sync

# Include training dependencies (TRL, transformers, datasets, matplotlib)
uv sync --extra training
```

---

## Running the Server

```bash
# Development (auto-reload)
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4
```

The server exposes:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode |
| `/step` | POST | Execute an action |
| `/state` | GET | Current episode state |
| `/schema` | GET | Action and observation schemas |
| `/ws` | WebSocket | Persistent session |

---

## Running Inference

Run the trained agent across all 19 scenarios using the HuggingFace router:

```bash
HF_TOKEN=hf_... python inference.py
```

Filter by difficulty or scenario:

```bash
# Easy tier only
HF_TOKEN=hf_... python inference.py --difficulty easy

# Single scenario
HF_TOKEN=hf_... python inference.py --scenario mobilenet_v3_medium

# Different model
HF_TOKEN=hf_... python inference.py --model Qwen/Qwen2.5-72B-Instruct
```

---

## Training

Open and run `neural_tuner_trl_mac.ipynb` for the full pipeline:

1. Environment smoke test
2. Random policy baseline (n=20 seeds) and oracle ceiling
3. SFT warm-up on heuristic trajectories
4. GRPO training with curriculum scheduling (easy → medium → hard)
5. Post-training evaluation and plot export

```bash
# Optional: regenerate baseline vs heuristic episode traces
python rollout_eval.py --trace --model-id inception_v3 --difficulty medium

# Outputs
artifacts/eval/episode_trace.md
artifacts/eval/episode_metrics.json
artifacts/eval/episode_metrics.csv
```

---

## Project Structure

```
.
├── server/
│   ├── app.py                          # FastAPI server
│   ├── neural_tuner_env_environment.py # RL environment (reset/step logic)
│   ├── simulator.py                    # Hardware simulator (latency/memory/accuracy)
│   ├── scenarios.py                    # 19 scenarios × 3 difficulty tiers
│   └── model_zoo.py                    # Layer profiles for 5 neural networks
├── scripts/
│   ├── neural_tuner.py                 # TRL-compatible OpenEnv wrapper
│   └── run_training_eval.py            # Post-training evaluation sweep
├── tests/                              # pytest suite
├── artifacts/
│   ├── plots/                          # Training reward plots
│   ├── eval/                           # Episode metrics and traces
│   └── training/                       # Baseline and eval metrics JSON
├── neural_tuner_trl_mac.ipynb          # Training notebook
├── inference.py                        # Multi-scenario inference runner
├── rollout_eval.py                     # Baseline vs heuristic evaluator
├── client.py                           # OpenEnv WebSocket client
├── models.py                           # Pydantic action/observation/state models
├── openenv.yaml                        # OpenEnv deployment manifest
├── Dockerfile                          # Container build
└── pyproject.toml                      # Package config and dependencies
```

---

## Tests

```bash
pytest -q
```

Covers:
- Reward ordering for safe vs over-aggressive quantization
- Benchmark budget enforcement (5 per episode)
- Step limit enforcement (20 per episode)
- Invalid layer ID and missing argument handling
- Episode terminal state after `submit()`
- Training metrics schema validation

---

## Environment Actions

| Action | Arguments | Description |
|--------|-----------|-------------|
| `profile_layer` | `layer_id` | Reveal sensitivity score and optimization advice |
| `quantize_layer` | `layer_id`, `dtype` | Apply `FP32` / `FP16` / `INT8` / `INT4` |
| `prune_layer` | `layer_id`, `sparsity` | Remove `LOW=25%` / `MEDIUM=50%` / `HIGH=75%` channels |
| `revert_layer` | `layer_id` | Reset to FP32, no pruning |
| `benchmark` | — | Simulate hardware; returns latency, memory, accuracy, reward (max 5/episode) |
| `submit` | — | Finalise episode and receive reward |

---

## Reward

```
reward = latency_reward (0–0.40)
       + memory_reward  (0 or 0.30)
       + accuracy_reward (0–0.20)
       + efficiency_bonus (0 or 0.10)
```

Maximum reward: **1.0**. All constraints must be met simultaneously for the efficiency bonus.

---

## Deployment

```bash
# Push to HuggingFace Space
git push space master
```

Manifest: `openenv.yaml` — runtime: `fastapi`, entrypoint: `server.app:app`, port: `8000`.

---

## License

BSD-style — see `LICENSE` in the root of the repository.
