#! /bin/bash

# ============================================================
# veRL PPO Training with 2 GPUs
# ============================================================
# 2-GPU PPO 训练配置（RTX 3090 x2）。
# PPO vs GRPO 关键区别：
#   - PPO 需要 Critic 模型（额外的参数副本）
#   - PPO 使用 GAE 优势估计（而非 GRPO 组内比较）
#   - PPO 每 prompt 只生成 1 条回复（而非 5 条）
#   - PPO 通过 KL 惩罚控制策略更新幅度
#
# 系统要求: 至少 47GB 系统 RAM
# GPU 要求: 2 x 24GB+
# ============================================================

export MODEL_PATH=/root/autodl-tmp/Qwen3-0.6B
export DATA_PATH=/root/autodl-tmp/mobile-action-data/data

# 提高 Ray 系统内存 OOM 阈值（默认 0.95），避免 worker 被误杀
export RAY_memory_usage_threshold=0.98

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  data.train_files=$DATA_PATH/train.parquet \
  data.val_files=$DATA_PATH/validation.parquet \
  data.train_batch_size=128 \
  data.max_prompt_length=4096 \
  data.max_response_length=512 \
  actor_rollout_ref.model.path=$MODEL_PATH \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
  actor_rollout_ref.rollout.max_model_len=4096 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.max_num_seqs=32 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=64 \
  actor_rollout_ref.rollout.agent.num_workers=2 \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  actor_rollout_ref.rollout.multi_turn.tool_config_path=/root/autodl-tmp/verl/tools/mobile-actions.yaml \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  critic.optim.lr=1e-5 \
  critic.model.path=$MODEL_PATH \
  critic.ppo_micro_batch_size_per_gpu=1 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=llm-rl \
  trainer.experiment_name=td-mobile-ppo \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=80 \
  trainer.test_freq=80 \
  reward.num_workers=2 \
  trainer.total_epochs=15 \
  data.filter_overlong_prompts=true \
  data.prompt_key=messages 2>&1 | tee verl_ppo.log
