# RayPPOTrainer 工作流程

基于 Ray 的分布式 PPO 训练器，驱动进程在单 CPU/GPU 节点上运行，通过 RPC 调用分布式 worker 组来编排 PPO 数据流。

## 1. 初始化 (`__init__`)

- 存储 tokenizer、config、processor
- 确定 hybrid engine（当前强制使用）、是否需要 reference policy / reward model / critic
- 调用 `_create_dataloader()` 创建 train/val dataloader
- 计算 `total_training_steps`

## 2. Worker 初始化 (`init_workers`)

创建 Ray WorkerGroup：

- **ActorRollout**：混合 engine，同时负责推理和训练
- **Critic**（可选）：value network
- **RefPolicy**（可选）：参考策略，用于 KL 惩罚
- **RewardLoopManager**：管理 reward model（colocate 或 standalone）
- **AgentLoopManager / AsyncRolloutManager**：封装 rollout 的 async 生成流程
- **CheckpointEngineManager**：管理 checkpoint 的 sleep/wake/load/save

## 3. 训练主循环 (`fit`)

```
for epoch in range(total_epochs):
    for batch_dict in train_dataloader:
```

### 3a. 生成 (Rollout)

```
gen_batch → repeat(n) → async_rollout_manager.generate_sequences()
```

- 从 dataloader 拿到一个 batch，调用 `_get_gen_batch()` 剥离 reward 相关字段
- 通过 `AgentLoopManager.generate_sequences()` 调用 vLLM/SGLang 做异步生成
- 生成后调用 `checkpoint_manager.sleep_replicas()` 释放 GPU 显存给 reward model

> **关于 `rollout.n`（即每个 prompt 生成的 response 数）：**
> - 标准 PPO (GAE)：`rollout.n = 1`，每个 prompt 只生成 1 个 response，advantage 通过 value function 做 GAE 估计
> - GRPO：`rollout.n > 1`，每个 prompt 生成多个 response，按 uid 分组做 reward 归一化得到 advantage（不需要 value function）
> - verl 的代码是通用的，不论用哪个 `adv_estimator`，都会按 `rollout.n` 重复 prompt。使用 PPO/GAE 时应设 `rollout.n = 1`

### 3b. 计算 Reward

```
extract_reward(batch) → reward_tensor, reward_extra_infos
```

- 如果有 colocate reward model，调用 `_compute_reward_colocate()`
- 调用 `extract_reward()` 从 batch 中提取奖励信号（rule-based + model-based）
- 将 `reward_tensor` 存入 `batch.batch["token_level_scores"]`

### 3c. 计算 Old Log Prob

```
_compute_old_log_prob(batch) → old_log_probs
```

- 将 batch 从 padding 格式转为 no-padding 格式
- 调用 `actor_rollout_wg.compute_log_prob()` 计算当前策略下 response 的 log probability
- 用于 PPO 的 importance sampling 比率 (`ratio = exp(new_log_probs - old_log_probs)`)

### 3d. 计算 Ref Log Prob（可选）

```
_compute_ref_log_prob(batch) → ref_log_probs
```

- 如果 `use_reference_policy=True`，计算参考策略下的 log prob
- 用于 KL 惩罚：`reward = score - β * KL(π_ref || π_θ)`

### 3e. 计算 Value（可选）

```
_compute_values(batch) → values
```

- 如果有 critic，调用 `critic_wg.infer_batch()` 计算 state values
- 用于 GAE advantage 估计

### 3f. 计算 Advantage

```
compute_advantage(batch, adv_estimator, gamma, lam)
```

支持多种 advantage estimator：

- **GAE**：Generalized Advantage Estimation，需要 value function
- **GRPO**：Group Relative Policy Optimization，按 uid 分组归一化
- **REMAX / RLOO / Reinforce++** 等

### 3g. 更新 Critic（可选）

```
_update_critic(batch) → critic_metrics
```

- 在 mini-batch 上训练 value function
- 使用 MSE loss 拟合 returns

### 3h. 更新 Actor

```
_update_actor(batch) → actor_metrics
```

- 将 batch 转为 no-padding，设置 mini-batch 参数
- 调用 `actor_rollout_wg.update_actor()` 执行 PPO clip loss 更新
- 更新后调用 `checkpoint_manager.update_weights()` 将新权重同步到 rollout worker

## 4. 验证 (`_validate`)

- 每 `test_freq` 个 step 执行一次
- 对 validation dataset 做 rollout → reward 计算 → metrics 聚合
- 支持 merged validation（多数据源合并）

## 5. 辅助机制

| 机制 | 说明 |
|------|------|
| `_balance_batch` | 按序列长度重新排序，使各 DP rank 的 token 数均衡 |
| `apply_kl_penalty` | `token_level_rewards = token_level_scores - β * KL` |
| Rollout Correction | 支持 bypass mode (2-policy) 和 decoupled mode (3-policy, 含 π_old) |
| Checkpoint | 每次 `save_freq` 保存 actor + critic + dataloader 状态，支持 HDFS |
| Profiling | 支持 nsys / torch_memory 等 profiling 工具 |

## 6. Reward Score 分发机制

Reward score 函数通过 `data_source` 字符串动态分发，以 `openai/gsm8k` 为例：

### 路由入口

**`verl/utils/reward_score/__init__.py`** — `default_compute_score()` 函数：

```python
if data_source == "openai/gsm8k":
    from . import gsm8k
    res = gsm8k.compute_score(solution_str, ground_truth)
```

### 生命周期

```
RayPPOTrainer.fit()
  → extract_reward() [verl/trainer/ppo/reward.py]
    → load_reward_manager() 注入 compute_score 到 reward manager
      → reward_manager.__call__() / run_single()
        → self.compute_score(data_source, solution_str, ground_truth, ...)
          → default_compute_score()
            → gsm8k.compute_score(solution_str, ground_truth)
```

### 涉及的 Reward Manager

所有 reward manager 都通过 `default_compute_score` 路由到具体的数据集 scoring 函数：

| Reward Manager | 文件路径 |
|------|------|
| NaiveRewardManager | `verl/workers/reward_manager/naive.py` |
| BatchRewardManager | `verl/workers/reward_manager/batch.py` |
| PrimeRewardManager | `verl/workers/reward_manager/prime.py` |
| DAPORewardManager | `verl/workers/reward_manager/dapo.py` |
| Exp. Naive | `verl/experimental/reward_loop/reward_manager/naive.py` |
| Exp. DAPO | `verl/experimental/reward_loop/reward_manager/dapo.py` |
| Exp. GDPO | `verl/experimental/reward_loop/reward_manager/gdpo.py` |
| Exp. Limited | `verl/experimental/reward_loop/reward_manager/limited.py` |
| Exp. Remote | `verl/experimental/reward_loop/reward_manager/remote.py` |

## 数据流总结

```
DataLoader → GenBatch → Rollout(每个 prompt 生成 rollout.n 个 response) → Reward
    → OldLogProb → RefLogProb → Value → Advantage → UpdateCritic → UpdateActor → Weights Sync

- PPO(GAE): rollout.n = 1, 依赖 value function
- GRPO: rollout.n > 1, 按 uid 分组归一化 reward, 不需要 value function
```
