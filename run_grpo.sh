#! /bin/bash

# ============================================================
# veRL GRPO Training with 2 GPUs
# ============================================================
# 2-GPU 训练配置（RTX 3090 x2）。
# 相比 1 GPU 的改进：
#   - 吞吐量翻倍: ~150 tokens/sec（vs 74）
#   - 单 step 时间减半: ~10 min（vs ~20 min）
#   - GPU 使用更均衡
#
# 系统要求: 至少 47GB 系统 RAM
# GPU 要求: 2 x 24GB+
# ============================================================

export MODEL_PATH=/data/models/Qwen3-0.6B
export DATA_PATH=/data/datasets/td-mobile-data

# 提高 Ray 系统内存 OOM 阈值（默认 0.95），避免 worker 被误杀
export RAY_memory_usage_threshold=0.98

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  data.train_files=$DATA_PATH/train.parquet \
  data.val_files=$DATA_PATH/validation.parquet \
  data.train_batch_size=128 \
  data.max_prompt_length=4096 \
  data.max_response_length=512 \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path=$MODEL_PATH \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.max_model_len=4096 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=64 \
  actor_rollout_ref.rollout.agent.num_workers=2 \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  actor_rollout_ref.rollout.multi_turn.tool_config_path=/home/niwang/code/verl/tools/mobile-actions.yaml \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  reward.num_workers=2 \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name=llm-rl \
  trainer.experiment_name=gsm8k_grpo \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=80 \
  trainer.test_freq=80 \
  trainer.total_epochs=15 \
  data.filter_overlong_prompts=true \
  data.prompt_key=messages 2>&1 | tee verl_demo.log
