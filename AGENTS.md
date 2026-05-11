# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **Python virtualenv**: `.venv/` in project root
- **PyTorch constraint**: Must use `torch=2.9.1`
- **Hardware**: NVIDIA RTX 3090 x2 (24GB each), system RAM ~47GB minimum
- The venv is NOT yet configured with all dependencies (`pip install -e .` has not been run in it yet)

## Build & Development Commands

```bash
# Install in editable mode (choose one backend)
pip install -e .[test,vllm]
pip install -e .[test,sglang]

# Linting (pre-commit must be installed first)
pre-commit install
pre-commit run                    # staged changes only
pre-commit run --all-files        # entire repo
pre-commit run --all-files --show-diff-on-failure --color=always ruff

# Run tests
pytest tests/                     # all tests
pytest tests/special_sanity/      # sanity checks (CPU, no GPU needed)
pytest tests/trainer/             # trainer tests

# Build docs
cd docs && pip install -r requirements-docs.txt && make html
```

## Architecture Overview

verl is a Ray-based RL training framework for LLMs. The core abstraction is a **hybrid-controller** programming model that decouples computation from data dependencies.

### Key Layers

1. **Trainer** (`verl/trainer/`) — Top-level orchestration via Ray. `main_ppo.py` is the main entry point (`python -m verl.trainer.main_ppo`). It creates Ray actors for actors, rollout engines, critics, and reward models, then runs the PPO/GRPO loop.

2. **Workers** (`verl/workers/`) — Distributed workers that execute training, rollout generation, and reward computation:
   - `engine/` — FSDP/Megatron model engines for training
   - `rollout/` — vLLM/SGLang-based inference for generation
   - `reward_manager/` — Parallel reward computation

3. **Protocol** (`verl/protocol.py`) — `DataProto`, the core data container wrapping tensordict for passing data between workers. All inter-worker communication uses this.

4. **Tools** (`verl/tools/`) — Native tool implementations for tool-calling RL. Each tool is a Python class registered in a YAML config (e.g., `tools/mobile-actions.yaml`). Tools are loaded dynamically by the `tool_agent` agent loop during multi-turn rollout.

5. **Config** (`verl/trainer/config/`) — Hydra/OmegaConf YAML configs. Override via CLI: `algorithm.adv_estimator=grpo`, `actor_rollout_ref.model.path=/path`, etc.

### Data Flow (GRPO/PPO Training)
```
Parquet dataset → rollout (vLLM/SGLang generation) → reward computation → PPO/GRPO update → repeat
```

### Experimental features live in `verl/experimental/` — including async policy, transfer queue, VLA, and agent loop.

## Key Scripts in This Repo

| File | Purpose |
|------|---------|
| `run_grpo.sh` | 2-GPU GRPO training with mobile tool-calling |
| `run_ppo.sh` | 2-GPU PPO training with mobile tool-calling |
| `mobile-action-reward.py` | Custom reward function: matches predicted `<tool_call>` blocks against ground truth tool calls |
| `main.py` | Standalone script to test chat template application on a single parquet row |

## Companion Project: mobile-action-data

Located at `/root/autodl-tmp/mobile-action-data/`. This generates the training data used by verl:

1. **`gen_answer_per_query.py`** — Calls DeepSeek API to generate tool-call ground truth for each user query → `train_data.json`
2. **`data_process.py`** — Splits `train_data.json` into train/validation parquet files with a strict schema (messages + reward_model)

The resulting parquet files (`train.parquet`, `validation.parquet`) are consumed by verl's GRPO/PPO training with `data.prompt_key=messages`.

## Model

- Base model: **Qwen3-0.6B** at `/data/models/Qwen3-0.6B` (also cloned at `/root/autodl-tmp/Qwen3-0.6B`)
- Tokenizer loaded via `transformers.AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)`
- Chat template: Qwen format with Hermes-style `<tool_call>` blocks for function calling
