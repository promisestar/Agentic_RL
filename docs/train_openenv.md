# `train_openenv.py` 使用与设计说明

本文档说明 `scripts/train_openenv.py` 的定位、依赖安装、命令行用法、训练流水线、**用户话术生成与脚本衔接**、奖励设计、输出产物，以及常见问题排查。若你更关心「训练 prompt 是如何从环境采样出来的」数据结构细节，请另见同目录下的 [train_openenv_data_generation.md](./train_openenv_data_generation.md)。

---

## 1. 脚本定位

### 1.1 它做什么

`train_openenv.py` 是 EcomRLVE-GYM 的 **OpenEnv 风格强化学习训练入口**。它用 Unsloth 加载大语言模型（默认 Qwen3-1.7B），以 LoRA 适配器做参数高效微调，并通过 TRL 的 `GRPOTrainer`（Group Relative Policy Optimization）进行策略优化。训练信号来自 EcomRLVE-GYM 环境：模型需要输出符合协议的 JSON 动作（含工具调用与最终答案），环境据此给出可验证奖励。

整体目标是：让模型学会扮演电商购物助手——搜索商品、操作购物车、处理订单与退货、查询政策，并在合适时机给出结构化最终答案。

### 1.2 与 `scripts/train.py` 的区别

| 对比项 | `scripts/train.py` | `scripts/train_openenv.py` |
| --- | --- | --- |
| 主要目的 | 验证环境与 rollout 管线 | 真正训练 LLM 策略 |
| 策略来源 | `DummyModelFn`（规则/随机假智能体） | Unsloth 加载的真实 HuggingFace 模型 |
| 是否更新参数 | 否 | 是（GRPO + LoRA） |
| 核心依赖 | 项目主依赖即可 | `uv sync --extra train`（`unsloth` + `transformers` 5.2–5.5） |
| GPU | 通常不需要 | 需要；建议用 `CUDA_VISIBLE_DEVICES` 绑单卡 |
| 典型输出 | 统计 JSON（奖励、成功率等） | LoRA checkpoint + 训练日志 |

简言之：`train.py` 适合先确认 Gym / Collection / 奖励是否正常；确认无误后再用 `train_openenv.py` 做模型训练。

### 1.3 设计来源

脚本结构参考 OpenEnv 生态中「Unsloth + TRL GRPOTrainer」的常见写法（例如 2048 等 notebook 范式），并适配到多轮电商对话与可验证环境奖励：先构建仅含 prompt 的数据集，训练步内对每个 prompt 采样一组 completion，再由多路奖励函数打分并做组内相对优势更新。

训练时，prompt 中的**用户首句**以及（若走到多轮）后续用户回复，并不是写死在脚本里的，而是由 EcomRLVE-GYM 环境在 `reset()` / `step()` 时通过用户模拟器生成。下文第 6 节专门说明这条链路，以及它与 `train_openenv.py` 中 `EcomRLVEOpenEnv` / `env_reward` 的对应关系。

---

## 2. 环境与依赖

### 2.1 运行前提

- **Python**：项目要求 `>=3.13`（见 `pyproject.toml` / `.python-version`）。
- **GPU**：需要可用的 NVIDIA GPU。脚本默认 `fast_inference=False`（不依赖 vLLM）；训练结束后会在 `cuda` 上做一次推理冒烟测试。
- **CUDA 与 PyTorch**：通过 `uv` 的 `torch-backend = "auto"` 尽量匹配本机 CUDA 对应的 torch 轮子。若本机 `nvcc` 与 PyTorch 编译所用 CUDA 不一致，应优先安装预编译 wheel，避免对旧版包做源码编译。
- **可选用户模拟 LLM**：环境侧用户话术默认尝试 Ollama（`localhost:11434`），也可通过环境变量改用 vLLM 等 OpenAI 兼容服务；未启动时回退模板，训练不中断（见第 6 节）。

### 2.2 安装训练依赖

训练相关包放在可选依赖组 `train` 中（体积较大，且偏 CUDA）：

```bash
# 在项目根目录
uv sync --extra train
```

该 extra 当前包含：

- `unsloth`：模型加载、LoRA、训练加速封装
- `transformers>=5.2.0,<=5.5.0`：支持 Qwen3.5（`model_type=qwen3_5`）且落在当前 Unsloth 允许区间内

**关于 vLLM / `--fast_inference`：**  
当前 Unsloth 要求 `transformers<=5.5.0`，而较新的 vLLM（≥0.24）要求 `transformers>=5.5.3`，二者没有交集；旧版 vLLM 又排斥 Transformers 5.x。因此 `train` extra **不再硬依赖 vLLM**。脚本默认 `fast_inference=False`；只有在你自行装好与版本匹配的 vLLM，并明确传入 `--fast_inference` 时才开启。

```bash
UV_HTTP_TIMEOUT=600 uv sync --extra train
```

安装完成后推荐用项目环境运行：

```bash
uv run python scripts/train_openenv.py --help
```

### 2.3 与主依赖的关系

`torch`、`transformers`、`trl`、`datasets` 等已在项目主依赖中。`train` extra 主要补齐 Unsloth，并把 `transformers` 收紧到支持 Qwen3.5 的区间；其传递依赖通常还包括 `bitsandbytes`、`triton`、`peft` 等。解析结果以 `uv.lock` 为准。

---

## 3. 快速开始

### 3.1 最小命令

```bash
uv run python scripts/train_openenv.py
```

默认行为概要：

- 模型：`Qwen/Qwen3-1.7B`
- 环境集合：`C1`
- 训练步数：`300`
- LoRA rank：`16`
- GRPO 组大小 `G`：`4`
- 4-bit 量化加载：开启
- 输出目录：`outputs/ecomrlve_grpo`

### 3.2 常用完整示例

```bash
uv run python scripts/train_openenv.py \
  --model Qwen/Qwen3-1.7B \
  --collection C1 \
  --max_steps 300 \
  --lora_rank 16 \
  --num_generations 4 \
  --n_prompts 1000 \
  --load_in_4bit \
  --output_dir outputs/ecomrlve_grpo
```

显存紧张时可降低序列长度、组大小或 vLLM 显存占比（后者在脚本里写死为 `gpu_memory_utilization=0.6`，若需改动需改源码）。

### 3.3 16-bit 加载

若不想用 4-bit：

```bash
uv run python scripts/train_openenv.py --load_in_16bit
```

`--load_in_16bit` 会覆盖默认的 4-bit 行为。

---

## 4. 命令行参数一览

### 4.1 模型相关

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | `Qwen/Qwen3-1.7B` | HuggingFace 模型名或本地路径 |
| `--load_in_4bit` | `True`（flag，默认开启） | 4-bit 量化加载 |
| `--load_in_16bit` | `False` | 16-bit 加载，优先级高于 4-bit |
| `--max_seq_length` | `2048` | 最大序列长度（prompt + completion 总预算上界） |
| `--lora_rank` | `16` | LoRA 秩；`lora_alpha` 在代码中设为 `rank * 2` |
| `--fast_inference` | `False` | 开启 Unsloth/vLLM 快速生成；需自行安装兼容的 vLLM |

### 4.2 环境与数据相关

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--collection` | `C1` | 环境集合，取值见 `COLLECTIONS`（如 C1/C2/C4/C8） |
| `--seed` | `42` | 主随机种子（环境与 LoRA `random_state`） |
| `--n_prompts` | `1000` | 预先采样的训练 prompt 条数 |

### 4.3 训练超参

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--max_steps` | `300` | 最大训练步数 |
| `--num_generations` | `4` | GRPO 每组采样条数 \(G\) |
| `--batch_size` | `1` | `per_device_train_batch_size` |
| `--gradient_accumulation_steps` | `1` | 梯度累积步数 |
| `--learning_rate` | `2e-5` | 学习率 |
| `--temperature` | `0.7` | 采样温度 |
| `--max_prompt_length` | `None` | prompt 最大 token 数；默认按首条样本自动估计并加约 20% 余量 |
| `--max_completion_length` | `512` | completion 最大 token 数；实际还会被序列总预算截断 |

### 4.4 输出与日志

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--output_dir` | `outputs/ecomrlve_grpo` | checkpoint 与最终权重目录 |
| `--save_steps` | `50` | 每隔多少步保存一次中间 checkpoint |
| `--report_to` | `none` | 实验追踪：`none` / `wandb` / `tensorboard` / `trackio` |
| `--log_to_file` | `False` | 追加参数，开启后把训练日志同时落到 `<output_dir>/train.log` |
| `--log_path` | `None` | 显式指定日志文件路径；设置后隐含开启落盘，会覆盖默认的 `<output_dir>/train.log` |

其他由 `GRPOConfig` 写死或半写死的配置包括：`weight_decay=0.01`、`warmup_ratio=0.1`、cosine 学习率调度、`optim=adamw_8bit`、`logging_steps=1`，以及按硬件自动选择 `bf16` 或 `fp16`。

---

## 5. 端到端训练流水线

`main()` 按固定顺序执行下列阶段。

```text
┌──────────────────────────────────────────────────────────────────────┐
│ 1. 初始化 EcomRLVEOpenEnv（包装 EcomRLVEEnv，关闭磁盘 trace）          │
│ 2. Unsloth FastLanguageModel 加载基座 + 挂载 LoRA                      │
│ 3. build_dataset()：采样 {prompt, env_id, episode_seed}               │
│ 4. 自动/手动确定 max_prompt_length，并裁剪 max_completion_length       │
│ 5. 配置 GRPOConfig，构造 GRPOTrainer（三个 reward_funcs）             │
│ 6. trainer.train()                                                     │
│ 7. 保存最终 LoRA 到 output_dir/final                                   │
│ 8. for_inference + 一次 TextStreamer 冒烟生成                          │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.1 环境封装：`EcomRLVEOpenEnv`

该类是对 `EcomRLVEEnv` 的薄封装，核心职责有二：

1. **采样 prompt**：轮询 collection 中的 `env_id`，用确定性 `episode_seed` 调用 `env.reset()`，把 `SYSTEM_PROMPT` 与观测中的对话拼成 chat messages。
2. **奖励阶段复现题目**：`evaluate_completion(completion, env_id, episode_seed)` 用**同一对** `(env_id, episode_seed)` 再次 `reset()`，保证模型生成时看到的隐藏目标、约束与用户首句，与打分时环境实例一致；然后 `step(completion)` 得到环境标量奖励。

若模型输出未将 episode 置为 `done`，封装会构造一个带 `done=true` 的 fallback JSON 再 step 一次，以便拿到可终止的评估结果。

训练期间关闭了 `dump_dir` 与 `trace_episodes`，避免大量磁盘 IO；`validate_rewards` 保持开启。

### 5.2 System Prompt、交卷 JSON 与 `done` 语义

`SYSTEM_PROMPT`（定义于 `scripts/train_openenv.py`）告诉模型：它是电商助手，可调用 `catalog.*` / `cart.*` / `order.*` / `return.*` / `policy.*` 等工具，并且**每一轮都必须输出合法 JSON 动作**。格式解析统一走 `ecom_rlve.server.state.parse_action`；结构化类型见 `ActionSchema` / `AnswerSchema`（`src/ecom_rlve/server/state.py`）。

#### 5.2.1 Agent 交卷 / 动作 JSON 模板

**完整动作（含终答交卷）模板：**

```json
{
  "assistant_message": "面向用户的自然语言回复",
  "tool_calls": [
    {"name": "catalog.search", "args": {"query": "...", "top_k": 5}}
  ],
  "answer": {
    "env": "PD",
    "recommended_product_ids": ["p1", "p2"],
    "selected_order_id": null,
    "selected_line_id": null,
    "policy_answer": null,
    "done": true
  }
}
```

字段说明：

| 字段 | 是否必需 | 含义 |
| --- | --- | --- |
| `assistant_message` | 是 | 展示给用户的文本；解析要求非空字符串 |
| `tool_calls` | 否（可 `[]`） | 本轮工具调用列表；每项含 `name` 与 `args` |
| `answer` | 否（可为 `null` / 省略） | 结构化终答；**交卷时**应给出，并设 `done: true` |
| `answer.env` | 有 `answer` 时通常需要 | 任务所属环境，如 `PD` / `SUB` / `CART` / `RETURN` / … |
| `answer.recommended_product_ids` | 视任务 | PD/SUB/BUNDLE 等推荐商品 ID 列表 |
| `answer.selected_order_id` / `selected_line_id` | 视任务 | STATUS/RETURN 等场景 |
| `answer.policy_answer` | 视任务 | POLICY 场景的答案 |
| `answer.done` | 有 `answer` 时关键 | **是否主动声明任务完成**；缺省按 `false` |

**未交卷、仅调用工具（中间步）示例：**

```json
{
  "assistant_message": "我先帮您在目录里搜索一下。",
  "tool_calls": [
    {"name": "catalog.search", "args": {"query": "透气跑鞋", "top_k": 5}}
  ],
  "answer": null
}
```

**已搜到结果、准备交卷示例：**

```json
{
  "assistant_message": "根据您的需求，推荐这款商品。",
  "tool_calls": [],
  "answer": {
    "env": "PD",
    "recommended_product_ids": ["prod_42"],
    "done": true
  }
}
```

脚本里的 system prompt 明确要求：找到答案后应在 `answer` 中设置 `"done": true`。

#### 5.2.2 何时算「已标记 done」（agent 主动交卷）

环境判定（`EcomRLVEEnv.step`）为：

```text
agent_done = (action.answer is not None) and (action.answer.done is True)
```

因此必须**同时**满足：

1. JSON 中存在可解析的 `answer` 对象；
2. `answer.done == true`（布尔真）。

此时环境认为 agent 主动交卷，终止原因通常为 `agent_done`。一般**不会再**调用 `UserSimulator.generate_response()` 追加用户话。

#### 5.2.3 何时算「未标记 done」

下列任一情况，`agent_done` 均为 `False`：

| 情况 | 说明 |
| --- | --- |
| 没有 `answer` 字段，或 `"answer": null` | `answer is None` |
| 有 `answer`，但未写 `done` | `AnswerSchema` 默认 `done=False` |
| 显式 `"done": false` | 明确表示尚未完成 |
| `"done"` 为其它假值 | 按布尔假处理 |

典型「未标记 done」场景：还在搜商品、还在调购物车工具、还在追问用户——本轮只有 `assistant_message` + `tool_calls`，或不带终答块。

#### 5.2.4 未标记 done 时如何处理

分两层：**完整环境 `step`**，以及 **本训练脚本的 `evaluate_completion`**。

**（A）`EcomRLVEEnv.step`（环境本体）**

若本轮 `agent_done == False`：

1. 若存在 `UserSimulator`，会调用 `generate_response(...)`，把新的用户消息写入对话（此处可能触发 Ollama；失败则模板 fallback，见第 6 节）。
2. episode 仍可能因其它条件终止：

```text
terminal = agent_done or (turn >= T_max) or user_quit
```

即：即使 agent 不交卷，达到最大回合或用户 ragequit，也会结束并结算奖励。

**（B）`EcomRLVEOpenEnv.evaluate_completion`（`train_openenv.py` 奖励路径）**

GRPO 的 `env_reward` 走的是训练封装，逻辑是：

1. 用同一 `(env_id, episode_seed)` `reset` 后，对模型 completion 执行一次 `env.step(completion)`；
2. 若返回的 episode 仍 `done == False`（模型未交卷，且本步也未因其它条件结束），脚本会**再构造一条强制终答 JSON** 并二次 `step`：
   - 若原输出已有 `answer`：在其基础上把 `done` 改成 `true`；
   - 若没有可用 `answer`：使用 `{"env": <当前env>, "done": true, "recommended_product_ids": []}`；
   - `assistant_message` 使用固定短句（如 `"Here is my final answer."`），`tool_calls` 为空。

这样做的目的是：在「单步 GRPO 评估」设定下，保证一定能拿到**终局**环境奖励，而不是卡在非终止状态。代价是：模型若本该继续交互却忘了标 `done`，训练侧仍会强制交卷再打分，与真实多轮交互不完全等价。

**对照小结：**

| 模型输出 | 环境 `agent_done` | 训练 `evaluate_completion` |
| --- | --- | --- |
| `answer.done: true` | 主动交卷并终止 | 通常一次 `step` 即结束 |
| 无 `answer` / `done: false` | 未交卷；可能继续用户模拟或等到 `T_max`/quit | 若第一次仍未 `done`，会强制补 `done: true` 再 `step` |

### 5.3 数据集形态

`build_dataset` 生成 HuggingFace `Dataset`，每行包含：

| 字段 | 含义 |
| --- | --- |
| `prompt` | chat messages 列表（含 system + user 等） |
| `env_id` | 出题所用环境 ID |
| `episode_seed` | 复现该题的确定性种子 |

**没有**预先标注的标准 completion。训练步内由策略采样生成回复；TRL 会把 `env_id`、`episode_seed` 等额外列传入奖励函数的 `**kwargs`。细节见 [train_openenv_data_generation.md](./train_openenv_data_generation.md)。

### 5.4 序列长度预算

若未指定 `--max_prompt_length`，脚本对第一条样本 `apply_chat_template` 后估长，再乘约 1.2 并加 1。随后：

```text
max_completion_length = min(用户设定的 max_completion_length,
                            max_seq_length - max_prompt_length)
```

若算得的 completion 长度 ≤ 0，脚本会报错退出，需要增大 `--max_seq_length` 或减小 prompt 上限。

### 5.5 模型与 LoRA

- `FastLanguageModel.from_pretrained(..., fast_inference=args.fast_inference, gpu_memory_utilization=0.6)`
- 对 Qwen3.5 等可能返回 `Qwen3VLProcessor` 的情况，脚本通过 `resolve_text_tokenizer()` 解包出内部文本 tokenizer，再用于 `encode` / `apply_chat_template` / `GRPOTrainer`
- LoRA 目标模块：`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- `use_gradient_checkpointing="unsloth"`

训练结束后调用 `FastLanguageModel.for_inference(model)`，再做一次短生成，便于肉眼检查输出是否仍为乱码或明显未学会 JSON。

---

## 6. 用户话术生成：环境实现与 `train_openenv.py` 的衔接

训练日志里若出现：

```text
Ollama call failed: ... localhost:11434 ... Connection refused
```

或（在配置了 OpenAI/vLLM 后端时）：

```text
OpenAI-compatible LLM call failed: ...
```

说明环境在尝试用外部 LLM 生成用户话术，但服务未启动或不可达。这是**警告而非训练崩溃**：库设计为 graceful fallback，会改用模板生成用户语句，GRPO 与 `env_reward` 仍可继续。本节说明话术从哪里来、如何接到训练脚本，以及如何改用 **vLLM**。

### 6.1 三条路径概览

用户话术生成优先走可配置的 HTTP LLM 后端，失败则模板：

| 路径 | 何时使用 | 说明 |
| --- | --- | --- |
| **Ollama**（默认） | `ECOM_RLVE_LLM_BACKEND=ollama` 或未设置 | `POST {base}/api/chat` |
| **OpenAI 兼容 / vLLM** | `ECOM_RLVE_LLM_BACKEND=openai` | `POST {base}/chat/completions`（通常 `http://host:8000/v1`） |
| **模板 fallback** | LLM 调用返回 `None` | [`templates.py`](../src/ecom_rlve/simulator/templates.py) 填槽；训练不中断 |

**不必须**启动任何 LLM 服务；不开服务也能训，只是用户句更模板化。

### 6.2 总体调用链

```text
scripts/train_openenv.py
  │
  ├─ EcomRLVEOpenEnv.sample_prompt()
  │     └─ EcomRLVEEnv.reset(env_id, seed=episode_seed)
  │           ├─ 选题 / 难度 / persona / problem
  │           ├─ 创建 UserSimulator(...)
  │           └─ initial_message = UserSimulator.generate_initial_message()
  │                 ├─（优先）llm_backend._llm_generate
  │                 │         ├─ backend=ollama  → /api/chat
  │                 │         └─ backend=openai  → /v1/chat/completions（vLLM 等）
  │                 └─（失败）templates 渲染 / 各 env 的 verbalize_* fallback
  │           └─ Observation.conversation 含首条 user 消息
  │     └─ messages = [SYSTEM_PROMPT] + obs.conversation   → 写入 Dataset.prompt
  │
  └─ env_reward → EcomRLVEOpenEnv.evaluate_completion()
        └─ 再次 reset(同一 env_id, episode_seed) 复现同一用户首句与隐藏目标
        └─ env.step(completion)
              └─ 若未 done：UserSimulator.generate_response() 可能再生成用户回复
                    └─ 同样走 _llm_generate，失败则模板
```

要点：

1. **用户话术不在 `train_openenv.py` 里手写**；脚本只消费 `obs.conversation`。
2. **首句**在 `reset()` 时生成；**后续用户句**在 `step()` 中、当 agent 尚未 `done` 时由 `UserSimulator.generate_response()` 生成。
3. 当前 GRPO 评估路径（`evaluate_completion`）通常是「一次（或两次）step 即终止」，因此训练中更常见的是首句生成触发 LLM；完整多轮用户模拟在环境 `step` 循环更长时才会频繁出现。

### 6.3 使用 vLLM（OpenAI 兼容）作为用户模拟后端

用户模拟 LLM 与训练策略模型是**两个独立进程**。不要把 vLLM 装进本仓库的 `train` extra 来「同进程调用」；应单独起服务，训练侧只用 `requests` 访问 HTTP。

**建议与训练卡隔离**（混用同一张卡易 OOM）：

```bash
# 终端 A：用户模拟用 vLLM（示例占用 GPU 3）
CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen3-1.7B --port 8000

# 终端 B：训练（示例占用 GPU 0）
export ECOM_RLVE_LLM_BACKEND=openai
export ECOM_RLVE_LLM_BASE_URL=http://localhost:8000/v1
export ECOM_RLVE_LLM_MODEL=Qwen/Qwen3-1.7B
# 可选：export ECOM_RLVE_LLM_API_KEY=EMPTY
# 可选：export ECOM_RLVE_LLM_TIMEOUT=30
CUDA_VISIBLE_DEVICES=0 bash scripts/run_train_openenv.sh
```

环境变量说明：

| 变量 | 含义 | 默认 |
| --- | --- | --- |
| `ECOM_RLVE_LLM_BACKEND` | `ollama` 或 `openai` | `ollama` |
| `ECOM_RLVE_LLM_BASE_URL` | 服务根 URL | ollama: `http://localhost:11434`；openai: `http://localhost:8000/v1` |
| `ECOM_RLVE_LLM_MODEL` | 模型名 | `qwen3.5` |
| `ECOM_RLVE_LLM_TIMEOUT` | 超时秒数 | `30` |
| `ECOM_RLVE_LLM_API_KEY` | OpenAI 兼容可选 Bearer Token | `EMPTY` |

若 `ECOM_RLVE_LLM_BASE_URL` 未以 `/v1` 结尾，客户端会自动补上 `/v1/chat/completions` 与 `/v1/models`。

继续使用 Ollama 时无需改环境变量（保持默认即可），或显式：

```bash
export ECOM_RLVE_LLM_BACKEND=ollama
export ECOM_RLVE_LLM_BASE_URL=http://localhost:11434
export ECOM_RLVE_LLM_MODEL=qwen3.5
```

### 6.4 `train_openenv.py` 中的相关部分

#### （1）`EcomRLVEOpenEnv.sample_prompt`：把环境生成的用户句编进 prompt

对应 `scripts/train_openenv.py` 中的封装逻辑：

1. 轮询 `collection` 内的 `env_id`，计算确定性 `episode_seed`。
2. 调用 `self.env.reset(env_id=env_id, seed=episode_seed)`。此时环境内部完成出题，并生成**首条用户消息**写入 `obs.conversation`。
3. 组装 chat messages：先放脚本内的 `SYSTEM_PROMPT`（助手角色与工具协议），再追加 `obs.conversation` 中的内容（通常至少包含一条 `role=user`）。
4. 返回 `(messages, env_id, episode_seed)`，供 `build_dataset` 落盘为训练行。

因此：日志里看到的用户自然语言问句，来自环境 `reset()`，而不是 `SYSTEM_PROMPT`。

#### （2）`build_dataset`：只保存 prompt，不保存「标准用户话」标签

`build_dataset(openenv, tokenizer, n_prompts)` 循环调用 `sample_prompt`，每行写入：

| 字段 | 与用户话术的关系 |
| --- | --- |
| `prompt` | 已含 system + 环境生成的 user（等）消息 |
| `env_id` / `episode_seed` | 奖励阶段用来**复现同一首句与同一题** |

数据集本身不单独存一份 `user_utterance` 字段；用户句嵌在 `prompt` 的 messages 里。

#### （3）`evaluate_completion` / `env_reward`：复现同一用户首句再打分

`env_reward` 从 completion 抽出 JSON 后调用：

```text
_OPENENV.evaluate_completion(completion, env_id, episode_seed)
```

其内部再次：

```text
obs = self.env.reset(env_id=env_id, seed=episode_seed)
obs, reward, done, info = self.env.step(completion)
```

因为 `(env_id, episode_seed)` 与出题时相同，`reset()` 会确定性生成**同一条**初始用户消息（以及同一隐藏目标）。若模型输出未标记 `done`，封装会再 step 一次强制终答 JSON，以便拿到终端奖励。

在这一过程中，若 `step` 认为对话未结束，`EcomRLVEEnv` 仍可能调用 `UserSimulator.generate_response()`，从而再次请求配置的 LLM 后端——这就是训练 step 中反复出现 LLM 连接 WARNING 的直接原因之一。

#### （4）脚本不负责启动用户模拟 LLM

`train_openenv.py` / `run_train_openenv.sh` **不**拉起 Ollama 或 vLLM。是否使用 LLM 用户模拟，取决于环境变量指向的 HTTP 服务是否可达。

> 日志落盘：`scripts/run_train_openenv.sh` 默认开启 `--log_to_file`，训练日志会同时写入 `<output_dir>/train.log`。可通过环境变量临时关掉或换路径：
>
> - `LOG_TO_FILE=0 bash scripts/run_train_openenv.sh`：仅控制台输出。
> - `LOG_PATH=/tmp/ecomrlve.log bash scripts/run_train_openenv.sh`：写入自定义路径（隐含 `--log_path`，覆盖 `<output_dir>/train.log`）。
>
> 用户传入的 `"$@"` 追加在脚本默认参数之后，如果手动传了 `--log_to_file` / `--log_path`，以用户传入为准（`argparse` 以后者为准）。

### 6.5 库内实现位置（环境侧）

#### 可切换 LLM 客户端与各类 verbalize / 对话生成

**文件：** `src/ecom_rlve/simulator/llm_backend.py`

| 符号 | 职责 |
| --- | --- |
| `_llm_generate()` | 按 `ECOM_RLVE_LLM_BACKEND` 分发到 Ollama 或 OpenAI 兼容 API；失败返回 `None` |
| `_generate_ollama()` / `_generate_openai_compatible()` | 两后端的具体 HTTP 实现 |
| `is_llm_available()` | 探测当前后端是否可达（`is_ollama_available` 为其兼容别名） |
| `verbalize_constraints()` / `verbalize_with_strategic_omission()` | 商品发现等场景的约束口语化 / 策略性省略 |
| `verbalize_cart_request()` / `verbalize_return_request()` | 购物车 / 退货首句 |
| `generate_dialogue_response()` / `generate_clarification_response()` | 多轮继续说 / 澄清 |

设计原则：确定性 seed（尽力而为）、失败回退模板、LLM 只影响用户侧自然语言，**不影响**可验证奖励的判定逻辑。

#### 多轮用户模拟器

**文件：** `src/ecom_rlve/simulator/dialogue.py` → 类 `UserSimulator`

- `generate_initial_message()`：首条用户话（内部可走 llm_backend 或模板）
- `generate_response()`：根据助手回复、工具结果、进度信息生成后续用户话，并可能触发 ragequit 等行为
- 模板渲染依赖：`src/ecom_rlve/simulator/templates.py`

#### 各原子环境如何触发「首句 verbalize」

| 文件 | 典型调用 |
| --- | --- |
| `src/ecom_rlve/envs/product_discovery.py` | `verbalize_with_strategic_omission` / `verbalize_constraints` |
| `src/ecom_rlve/envs/cart.py` | `verbalize_cart_request` |
| `src/ecom_rlve/envs/returns.py` | `verbalize_return_request` |

这些在问题实例生成阶段把结构化约束变成自然语言用户请求。

#### 环境服务入口：创建模拟器并写入对话

**文件：** `src/ecom_rlve/server/openenv.py`（`EcomRLVEEnv`）

- **`reset()`**：构造 `UserSimulator(...)`，调用 `generate_initial_message()`，将结果 `append` 到 `state.conversation`，再打成 `Observation` 返回。这正是 `EcomRLVEOpenEnv.sample_prompt` 读到的用户句来源。
- **`step()`**：在 agent 未 `done` 时调用 `_user_sim.generate_response(...)`，把新的用户消息追加进对话，再决定是否终止并算奖励。

### 6.6 训练时看到 LLM WARNING 该如何理解

| 情况 | 含义 | 建议 |
| --- | --- | --- |
| `Connection refused` 到 `11434` | 默认 Ollama 后端但服务未开 | 可忽略（走模板）；或启动 Ollama；或改用 vLLM 并设 `ECOM_RLVE_LLM_BACKEND=openai` |
| `OpenAI-compatible LLM call failed` | vLLM/兼容服务不可达或模型名不对 | 检查 `BASE_URL` / `MODEL`、vLLM 是否在跑、是否与训练卡冲突 |
| `EmbeddingEngine: debug_mode=True` | 向量检索用确定性随机嵌入 | 与用户话术无关 |
| `env_reward ... reward=-0.625` | 策略输出被环境打了负分 | 早期训练常见，不等于 LLM 用户模拟失败 |

若只想稳定训策略、不依赖外部服务：保持 LLM 服务关闭即可，依赖模板 fallback。若希望用户模拟更接近真实对话：启动 Ollama 或 vLLM，并正确配置上表环境变量。

### 6.7 与数据生成文档的分工

- **本节**：用户话术的代码位置、模板 / Ollama / vLLM 路径，以及和 `train_openenv.py` 采样/奖励的接口关系。
- **[train_openenv_data_generation.md](./train_openenv_data_generation.md)**：prompt 数据集如何程序化采样、`(env_id, episode_seed)` 如何保证复现，更偏「数据行结构」。

---

## 7. 奖励函数设计

`GRPOTrainer` 同时注册三个奖励函数，最终信号为它们的组合（由 TRL 汇总；环境分在代码中额外放大）。

### 7.1 `format_reward`：动作 JSON 是否合法

从 completion 中抽取 JSON（支持去掉 markdown 代码块、截取首个 `{...}`），再 `parse_action`：

| 分数 | 条件 |
| --- | --- |
| `+1.0` | 合法且可解析为有效 action（含必需字段） |
| `-0.5` | 是 JSON 但字段不满足 action 协议 |
| `-2.0` | 找不到合法 JSON |

早期训练中该信号通常最先上升：模型先学会「把话说成 JSON」。

### 7.2 `tool_usage_reward`：工具名与结构是否合理

| 分数 | 条件 |
| --- | --- |
| `+1.0` | 存在 `tool_calls`，且名称均以合法前缀开头（`catalog.` / `cart.` / `order.` / `return.` / `policy.`） |
| `+0.5` | 无工具调用，但 `answer.done == true`（简单题可接受） |
| `-0.5` | 有工具调用但名称非法 |
| `-0.25` | 既无合法工具调用，也未给出 done 答案 |
| `-1.0` | 无法解析 |

### 7.3 `env_reward`：环境任务分（主信号）

流程：

1. 抽取 JSON；失败则记 `-1.0`。
2. 用该样本的 `env_id`、`episode_seed` 调用 `_OPENENV.evaluate_completion`。
3. 取环境返回的标量 `reward`（文档约定量级约 `[-1, 1]`），再 **乘以 5.0**，使任务完成度在总奖励中占主导。
4. 环境评估抛错时记 `-3.0`。

每隔约 10 次评估会打一条 info 日志（env、seed、reward、是否正确、终止原因、部分 breakdown）。评估过程中可能再次触发用户模拟器 / LLM 后端（见第 6 节）。

### 7.4 训练初期的预期现象

日志中脚本会提示：前约 **50–100 step** 可能主要在学 JSON 格式，环境奖励改善相对滞后。这是预期行为，不必过早判定训练失败。建议同时观察：

- format / tool 奖励是否先抬升；
- 随后 `env_reward`（×5 后）是否缓慢上升；
- checkpoint 冒烟生成是否开始出现可解析 JSON。

---

## 8. 输出产物

| 路径 | 内容 |
| --- | --- |
| `{output_dir}/checkpoint-*` | 按 `--save_steps` 保存的中间状态（由 Trainer 管理） |
| `{output_dir}/final/` | 训练结束后的 LoRA 适配器与 tokenizer/processor（`save_pretrained`） |

实验追踪取决于 `--report_to`。默认 `none`，仅依赖标准 logging。

> 日志落盘：默认仅输出到 stdout/控制台。若希望长期留存或离线分析，可使用 `--log_to_file`（默认写入 `<output_dir>/train.log`），或通过 `--log_path /path/to/train.log` 指定自定义路径。两种方式互不冲突——控制台与文件 sink 同时生效，日志格式与控制台一致（`时间戳 [级别] logger: 消息`）。同一进程内多次设置同一路径不会重复挂载 `FileHandler`。

---

## 9. 实现上的注意点与局限

下列行为来自当前脚本实现，使用与二次开发时需要知晓：

1. **全局环境单例**：`_OPENENV` 在 `main()` 中赋值，奖励函数通过全局变量访问。多进程 / 多卡数据并行时需自行评估线程安全与进程隔离问题。
2. **单步为主的评估形态**：`evaluate_completion` 对模型输出做一次（必要时两次）`step`，更接近「一次动作/一次终答」的打分路径；完整多轮交互式工具闭环若要更强，需要扩展采样与环境交互协议。
3. **`fast_inference` 默认关闭**：开启需要另行安装与当前 `transformers` 钉扎兼容的 vLLM；Qwen3.5 场景下通常保持默认关闭。
4. **显存占用**：若开启 vLLM 快速推理，还会按 `gpu_memory_utilization=0.6` 预留显存；OOM 时可减小模型、序列长度、`num_generations`，或下调该比例。
5. **首次运行会拉模型权重**：默认从 HuggingFace Hub 拉取；内网环境需预先缓存或改 `--model` 为本地路径。
6. **用户话术依赖环境库**：脚本不启动 Ollama/vLLM；未启动时模板 fallback，见第 6 节。
7. **单卡建议**：`scripts/run_train_openenv.sh` 默认设置 `CUDA_VISIBLE_DEVICES=0`，避免进程在所有可见 GPU 上创建 CUDA 上下文。

---

## 10. 常见问题排查

### 10.1 `ModuleNotFoundError: unsloth`

未安装 `train` extra。执行：

```bash
UV_HTTP_TIMEOUT=600 uv sync --extra train
uv run python scripts/train_openenv.py ...
```

### 10.2 Qwen3.5 报 `model_type qwen3_5` / transformers 过旧

需要 `transformers>=5.2.0`（当前 `train` extra 钉在 `>=5.2.0,<=5.5.0`）。重新：

```bash
uv lock
UV_HTTP_TIMEOUT=600 uv sync --extra train
```

### 10.3 `Qwen3VLProcessor` 没有 `encode`

已由 `resolve_text_tokenizer()` 解包内部 tokenizer。若仍报错，确认运行的是更新后的 `train_openenv.py`。

### 10.4 下载 `triton` 等大包超时

增大 HTTP 超时并重试；uv 通常会复用已下载缓存：

```bash
UV_HTTP_TIMEOUT=600 uv sync --extra train
```

### 10.5 `max_seq_length` 过小导致退出

日志会提示 prompt 长度与总长度冲突。增大 `--max_seq_length`，或显式给出更小的 `--max_prompt_length` / `--max_completion_length`。

### 10.6 训练很久格式分仍极低

检查采样温度是否过高、模型是否过小、completion 长度是否被截断导致 JSON 残缺。可先用更小的 `--max_steps` 与更强的基座做冒烟。

### 10.7 日志刷屏 `Ollama call failed` / `OpenAI-compatible LLM call failed`

见第 6.6 节：用户模拟 LLM 服务未开或配置错误。训练可继续（模板 fallback）；若要消除警告并提升用户话自然度，启动 Ollama 或 vLLM，并设置第 6.3 节中的环境变量。

### 10.8 看起来占用了多张 GPU

未设置 `CUDA_VISIBLE_DEVICES` 时，PyTorch/Unsloth 可能在所有可见卡上建上下文。使用：

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/run_train_openenv.sh
```

或依赖脚本默认的单卡绑定。

### 10.9 远程 IDE 在 `uv sync` 时卡顿

向 `.venv` 写入海量文件会触发远程文件监视与索引。可将 `**/.venv/**` 加入 Cursor/VS Code 的 `files.watcherExclude` / `search.exclude`，或在非 IDE 会话中完成安装后再连接。

---

## 11. 推荐工作流

1. 用 `scripts/train.py` 在目标 `--collection` 上跑若干 episode，确认环境奖励与难度机制正常。
2. （可选）启动 Ollama 或 vLLM，并配置 `ECOM_RLVE_LLM_*` 以启用 LLM 用户话术；否则接受模板 fallback。
3. `uv sync --extra train`，确认 `import unsloth` 与 GPU 可用。
4. 小规模冒烟：`CUDA_VISIBLE_DEVICES=0 bash scripts/run_train_openenv.sh --max_steps 20 --n_prompts 64 --num_generations 2`。
5. 正式训练：按显存与任务难度调整 collection、步数、LoRA rank 与组大小。
6. 在 `output_dir/final` 上做离线评测或接入后续 serving 流程。

---

## 12. 相关代码与文档索引

| 路径 | 说明 |
| --- | --- |
| `scripts/train_openenv.py` | 本说明对应的训练脚本（含 `EcomRLVEOpenEnv` / 奖励 / tokenizer 兼容） |
| `scripts/run_train_openenv.sh` | 推荐启动脚本（单卡 `CUDA_VISIBLE_DEVICES`、常用超参） |
| `scripts/train.py` | 环境 rollout / Dummy 基准脚本 |
| `docs/train_openenv_data_generation.md` | prompt 数据集生成与 `(env_id, episode_seed)` 对齐机制 |
| `src/ecom_rlve/server/openenv.py` | `EcomRLVEEnv`：`reset`/`step`、挂载 `UserSimulator` |
| `src/ecom_rlve/simulator/llm_backend.py` | 可切换用户模拟 LLM 后端（Ollama / OpenAI 兼容 vLLM） |
| `src/ecom_rlve/simulator/dialogue.py` | `UserSimulator` 多轮用户模拟 |
| `src/ecom_rlve/simulator/templates.py` | LLM 不可用时的模板渲染 |
| `tests/test_llm_backend.py` | 用户模拟 LLM 后端单测（mock HTTP） |
| `src/ecom_rlve/envs/product_discovery.py` 等 | 各原子环境触发首句 verbalize |
| `src/ecom_rlve/server/state.py` | `parse_action` 等状态/动作协议 |
| `src/ecom_rlve/training/collections.py` | Collection 定义（C1/C2/…） |
| `pyproject.toml` | `[project.optional-dependencies].train` 与 uv 索引配置 |

---

## 13. 版本说明

本文档依据仓库当前 `scripts/train_openenv.py`、用户模拟相关模块与 `pyproject.toml` 的实现整理。若后续调整默认超参、奖励权重、`fast_inference` 策略、用户模拟 LLM 后端或依赖约束，请以源码与 lock 文件为准，并同步更新本页。
