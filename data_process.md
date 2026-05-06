# veRL 数据流：从命令行到 RLHFDataset

## 1. 入口：Hydra 配置解析

`python3 -m verl.trainer.main_ppo` 使用 **Hydra** 作为配置系统（`main_ppo.py` 第 36-46 行）：

```python
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
```

加载 `verl/trainer/config/ppo_trainer.yaml`，关键声明：

```yaml
- data@data: legacy_data
```

这条声明告诉 Hydra：加载 `data/legacy_data.yaml`，内容放到 `config.data` 下。

## 2. CLI 参数覆盖

命令行参数形如：

```
data.train_files=/path/to/train.parquet
data.max_prompt_length=512
data.train_batch_size=512
```

Hydra 将这些值合并覆盖默认的 `config.data`。

## 3. `tool_config_path` 的插值跳转

`legacy_data.yaml` 中有 OmegaConf 插值：

```yaml
tool_config_path: ${oc.select:actor_rollout_ref.rollout.multi_turn.tool_config_path, null}
```

即 `config.data.tool_config_path` 的值来自 `config.actor_rollout_ref.rollout.multi_turn.tool_config_path`。命令行指定的是后者，解析后二者同步。

## 4. 数据集创建（main_ppo.py）

```python
train_dataset = create_rl_dataset(
    config.data.train_files,   # → data_files 参数
    config.data,               # → config 参数（整个 data.* 子树）
    tokenizer,
    processor,
    is_train=True,
    max_samples=config.data.get("train_max_samples", -1),
)
```

`create_rl_dataset` 内部调用 `get_dataset_class(data_config)`，默认返回 `RLHFDataset`，然后：

```python
dataset = dataset_cls(
    data_files=data_paths,
    tokenizer=tokenizer,
    processor=processor,
    config=data_config,   # 就是 config.data DictConfig
    max_samples=max_samples,
)
```

## 5. RLHFDataset 内部

### 初始化参数读取

```python
self.cache_dir = config.get("cache_dir", "~/.cache/verl/rlhf")
self.prompt_key = config.get("prompt_key", "prompt")
self.max_prompt_length = config.get("max_prompt_length", 1024)
self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
self.truncation = config.get("truncation", "error")
self.return_raw_chat = config.get("return_raw_chat", False)
self.use_shm = config.get("use_shm", False)
self.shuffle = config.get("shuffle", False)
self.seed = config.get("seed")

self.tool_config_path = config.get("tool_config_path", None)
if self.tool_config_path:
    from verl.tools.utils.tool_registry import initialize_tools_from_config
    tool_list = initialize_tools_from_config(self.tool_config_path)
    self.tool_schemas = [tool.tool_schema.model_dump(...) for tool in tool_list]
```

### 数据处理流程

1. **`_download()`** — 将 parquet 文件下载/缓存到本地
2. **`_read_files_and_tokenize()`** — 读取 parquet → HuggingFace Dataset，按 `max_samples` 采样，按 `max_prompt_length` 过滤超长 prompt

### 过滤逻辑（`maybe_filter_out_long_prompts`）

逐条数据 tokenize，若超 `max_prompt_length` 则丢弃。过滤时若 `tool_schemas` 非空，注入 `apply_chat_template(tools=tool_schemas)`，使 tokenizer 正确处理 tool 格式的 prompt。

## 6. 参数归属总表

| 参数 | 去向 |
|------|------|
| `data.train_files` | RLHFDataset (`data_files`) |
| `data.val_files` | 验证集 RLHFDataset |
| `data.max_prompt_length` | RLHFDataset 过滤 + 插值到 `rollout.prompt_length` |
| `data.max_response_length` | 仅插值到 `rollout.response_length`，给 vLLM |
| `data.train_batch_size` | RayPPOTrainer DataLoader |
| `data.gen_batch_size` | RayPPOTrainer DataLoader（优先） |
| `data.tool_config_path` | RLHFDataset 加载工具 schema（值来自插值同步） |
| `data.prompt_key` | RLHFDataset 指定 prompt 列名 |
| `data.truncation` | RLHFDataset |
| `data.filter_overlong_prompts` | RLHFDataset |
| `data.shuffle` | RLHFDataset + DataLoader 的 Sampler |
| `data.seed` | RLHFDataset + DataLoader 的 Sampler |

## 7. 训练时迭代

`__getitem__` 做三件事：

1. 构建 `raw_prompt`（替换 `<image>`/`<video>` 占位符）
2. 提取 `tools_kwargs`（来自 `extra_info` 字段）
3. 添加 `dummy_tensor`（保持 DataProto batch 非空）

由 `StatefulDataLoader` 按 `train_batch_size` 批量输出，送入 PPO 训练循环。

## 完整数据流

```
命令行参数 (data.*)
    │
    ▼
Hydra/OmegaConf 合并 → config.data: DictConfig
    │                        │
    │                        ├─ train_files/val_files → RLHFDataset(data_files=)
    │                        ├─ max_prompt_length     → 过滤超长 prompt
    │                        ├─ tool_config_path      → 初始化 tool_schemas
    │                        ├─ prompt_key            → 指定 prompt 列名
    │                        └─ ...其他配置
    │
    ▼
RLHFDataset._read_files_and_tokenize()
    │
    ├─ 读取 parquet → HuggingFace Dataset
    ├─ 按 max_prompt_length 过滤
    ├─ (可选)加载 tool_schemas，注入 apply_chat_template
    │
    ▼
StatefulDataLoader(batch_size=train_batch_size) 迭代输出
    │
    ▼
PPO Trainer 训练循环

## 8. 模型输出与标准答案的对比链路

PPO/GRPO 训练中，模型输出与训练数据标签（ground_truth）的对比发生在 **reward computation（奖励计算）** 阶段，自上而下分 5 层：

### 8.1 训练循环入口

**`verl/trainer/ppo/ray_trainer.py`** ~line 1422-1427

```python
if self.use_rm and "rm_scores" not in batch.batch.keys():
    batch_reward = self._compute_reward_colocate(batch)
    batch = batch.union(batch_reward)
reward_tensor, reward_extra_infos_dict = extract_reward(batch)
```

### 8.2 Reward Manager 派发

**`verl/trainer/ppo/reward.py`** `load_reward_manager()` 根据配置加载 RewardManager（naive/batch/dapo/prime），每个 Manager 做两件事：

1. 调用 tokenizer 将模型输出 token 序列解码为文本 `response_str`
2. 从 `data_item.non_tensor_batch["reward_model"]["ground_truth"]` 取出数据集中的标准答案

### 8.3 具体对比执行

**单样本** — `verl/workers/reward_manager/naive.py` ~line 77-90：

```python
ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
score = self.compute_score(
    data_source=data_source,
    solution_str=response_str,
    ground_truth=ground_truth,
    extra_info=extra_info,
)
```

**批量** — `verl/workers/reward_manager/batch.py` ~line 62-76：

```python
ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
scores = self.compute_score(
    data_sources=data_sources,
    solution_strs=responses_str,
    ground_truths=ground_truths,
)
```

### 8.4 数据集特定的评分函数

**`verl/utils/reward_score/__init__.py`** `default_compute_score()` 根据 `data_source` 路由：

| 数据集 | 对比方式 | 关键文件 |
|--------|---------|---------|
| GSM8k | 提取 `####` 后数字，`==` 精确匹配 | `gs8m.py` |
| MATH | 提取 `\boxed{}`，LaTeX 归一化后等价比较 | `math_reward.py` |
| MATH DAPO | 数学表达式归一化比较 | `math_dapo.py` |
| PRIME Math | sympy 符号表达式等价判定 | `prime_math/__init__.py` |
| SearchR1 QA | 提取 `<answer>` 标签，归一化 Exact Match | `search_r1_like_qa_em.py` |
| Code（Codeforces/APPS） | 沙箱执行代码，比对运行结果 | `sandbox_fusion.py` |
| Tool Calling（RLLA） | 比较工具调用 JSON 结构（函数名+参数） | `rlla.py` |
| Geometry3K | `\boxed{}` 提取后用 mathruler 评分 | `geo3k.py` |

### 8.5 Advantage 计算

得分放在 response 最后一个 token 位置：`rm_scores` → `token_level_rewards`，进入 **`verl/trainer/ppo/core_algos.py`** 计算 advantage：

- **GRPO**: 组内归一化 `(score - group_mean) / group_std`
- **PPO**: GAE（Generalized Advantage Estimation）
- **RLOO**: Leave-One-Out 基线

```
Training Data (prompt + ground_truth label)
    │
    ▼
模型生成 responses
    │
    ▼
RewardManager.__call__() 解码 response token → 文本，取出 ground_truth
    │  调用: compute_score(data_source, solution_str, ground_truth)
    ▼
default_compute_score() 按 data_source 路由
    │
    ▼
数据集专用评分函数:
  - 提取模型输出中的答案 (#### / \boxed{} / <answer>)
  - 与 ground_truth 比较 (精确匹配 / LaTeX等价 / sympy / 沙箱执行)
    │
    ▼
Score (0/1 或 -1/1) → rm_scores → token_level_rewards
    │
    ▼
Advantage 计算 (GRPO: 组归一化, PPO: GAE) → policy gradient 更新
```

veRL 不像监督学习那样直接计算 loss（CrossEntropy），而是通过 **reward function** 对模型生成做评分，评分结果驱动 advantage 计算，再通过 policy gradient 更新模型。
```
