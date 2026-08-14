# `train_openenv.py` 训练数据生成机制说明

本文档说明 `scripts/train_openenv.py` 在 GRPO（Group Relative Policy Optimization）训练流程中，**训练数据如何产生、结构是什么、与环境奖励如何对齐**。该机制属于「在线环境采样 + 程序化出题」，而非从外部静态语料库直接加载带标注的问答对。

---

## 1. 总体定位

### 1.1 与常见监督学习数据的区别


| 对比项  | 传统 SFT 数据集        | 本脚本的训练数据                             |
| ---- | ----------------- | ------------------------------------ |
| 数据来源 | 人工标注或离线爬取的对话/指令对  | EcomRLVE-GYM 环境在线 `reset()` 采样       |
| 样本内容 | 通常同时包含输入与标准输出     | **仅包含输入 prompt**；输出由训练中模型生成          |
| 监督信号 | token 级交叉熵 / 模仿学习 | 多奖励函数打分（格式、工具使用、环境任务分）               |
| 可复现性 | 依赖固定文件            | 依赖 `(env_id, episode_seed)` 确定性复现同一题 |


换言之：本脚本构建的是 **GRPO 用的 prompt 数据集**，不是带 `completion` 标签的模仿学习语料。模型回复在训练步内由策略采样产生，再经奖励函数评估后更新策略。

### 1.2 在训练流水线中的位置

```text
┌─────────────────────────────────────────────────────────────────┐
│  1. 初始化 EcomRLVEEnv（默认合成 catalog）                         │
│  2. 加载 Unsloth 模型 + LoRA                                       │
│  3. build_dataset()  ← 本文档关注点：程序化生成 prompt 数据集        │
│  4. 配置 GRPOConfig / GRPOTrainer                                  │
│  5. trainer.train()：对每个 prompt 生成 G 条 completion → 算奖励   │
│  6. 保存 LoRA 适配器                                               │
└─────────────────────────────────────────────────────────────────┘
```

对应代码入口位于 `main()` 中调用 `build_dataset(_OPENENV, tokenizer, n_prompts=args.n_prompts)` 的阶段。

---

## 2. 数据生成主流程

训练数据由三层抽象协作完成：

1. `**EcomRLVEEnv**`（`src/ecom_rlve/server/openenv.py`）：电商对话 RL 环境本体，负责 catalog、难度、问题实例、用户首句消息。
2. `**EcomRLVEOpenEnv**`（`scripts/train_openenv.py`）：对上述环境的薄封装，负责采样 prompt，并在奖励阶段用同一种子复现 episode。
3. `**build_dataset()**`：循环采样，组装为 HuggingFace `datasets.Dataset`。

### 2.1 流程图

```text
EcomRLVEOpenEnv(collection, seed)
        │
        │  构造时：EcomRLVEEnv(collection=..., seed=...)
        │         └─ 未传入 catalog → generate_synthetic_catalog(n≈1000)
        │
        ▼
build_dataset(n_prompts)
        │
        │  for i in 1..n_prompts:
        │      sample_prompt(tokenizer)
        │         │
        │         ├─ 轮询选择 env_id（来自 collection）
        │         ├─ episode_seed = counter * 1000 + 42
        │         ├─ obs = env.reset(env_id, seed=episode_seed)
        │         │         │
        │         │         ├─ 采样/确定 difficulty（自适应区间）
        │         │         ├─ env.generate_problem(...) → 隐藏目标
        │         │         ├─ 采样 persona
        │         │         └─ UserSimulator → 首条 user message
        │         │
        │         └─ messages = [SYSTEM_PROMPT] + obs.conversation
        │
        ▼
Dataset 行：{ prompt, env_id, episode_seed }
```

### 2.2 核心函数职责

#### `build_dataset(openenv, tokenizer, n_prompts)`

- **作用**：生成固定规模的 prompt 集合。
- **默认规模**：`--n_prompts` 默认为 `1000`。
- **实现要点**：对每个索引调用一次 `sample_prompt()`，将结果追加为字典行，最终 `Dataset.from_list(rows)`。
- **日志**：会打印实际包含的 `env_id` 集合，便于确认 curriculum collection 是否生效。

#### `EcomRLVEOpenEnv.sample_prompt(tokenizer)`

- **环境选择**：按 `_episode_counter` 对 `env_ids` 取模，**均匀轮询** collection 内环境（例如 C4 会在 PD / SUB / CART / RETURN 间循环）。详见下一小节。
- **种子设计**：`episode_seed = _episode_counter * 1000 + 42`，保证：
  - 不同 prompt 对应不同问题实例；
  - 同一 `(env_id, episode_seed)` 在奖励阶段可再次 `reset()` 复现完全相同的隐藏目标与用户首句。
- **消息组装**：
  - 固定写入电商助手角色与工具协议的 `SYSTEM_PROMPT`；
  - 追加 `obs.conversation`（`reset()` 后通常仅有一条 `role=user` 的初始消息）。

#### `_episode_counter` 是什么

`_episode_counter` 是训练封装类 `EcomRLVEOpenEnv` 内部的**私有整型计数器**，在 `__init__` 中初始化为 `0`，每调用一次 `sample_prompt()` 就加 `1`。它本身**不会**写入 HuggingFace Dataset，只服务于采样循环中的环境轮询与种子派生。

在 `sample_prompt()` 中，该计数器被使用两次：

1. **轮询环境**
  `env_id = self.env_ids[self._episode_counter % len(self.env_ids)]`  
   例如 collection 为 C4（含 PD / SUB / CART / RETURN）时，计数 `0, 1, 2, 3, 4, …` 会依次落到 PD → SUB → CART → RETURN → PD → …，从而在 `n_prompts` 足够大时使各环境样本数近似均衡。
2. **生成确定性 `episode_seed`**
  先执行 `self._episode_counter += 1`，再计算：  
   `episode_seed = self._episode_counter * 1000 + 42`  
   因此第 1、2、3 条样本的 seed 分别为 `1042`、`2042`、`3042`…。奖励阶段（`env_reward` / `evaluate_completion`）使用同一 `episode_seed` 再次调用 `env.reset()`，即可复现同一道题（相同隐藏目标与用户首句）。

与 Dataset 的关系可以概括为：

- 写入数据集的是派生出的 `env_id` 与 `episode_seed`；
- `_episode_counter` 仅存在于内存中的采样过程，是连接「第几条 prompt」与「哪道题 / 哪个 seed」的内部状态。

**易混点：** `EcomRLVEEnv`（`src/ecom_rlve/server/openenv.py`）内部另有一个同名的 `_episode_counter`，用于环境侧统计已开启的 episode 次数。二者是**相互独立**的两套计数；本文流程图与本节描述的，均指训练封装类 `EcomRLVEOpenEnv` 中的计数器。

#### `EcomRLVEEnv.reset(env_id, seed=...)`

`reset()` 是「题目」真正被生成的地方，主要步骤包括：

1. 确定环境 ID（脚本侧已强制传入，不再随机）。
2. 通过自适应难度引擎在当前 `[low, high]` 区间采样难度，或使用强制难度。
3. 调用对应原子环境的 `generate_problem(difficulty, catalog, seed)`，得到 `ProblemParams`（隐藏目标：目标商品、约束、订单线索等，**不直接暴露给模型**）。
4. 初始化 episode 状态（购物车、Seen 集合、订单历史等按环境按需生成）。
5. 采样用户 persona 权重，初始化 `UserSimulator`。
6. 生成首条用户自然语言消息，写入 `conversation`。
7. 返回 `Observation` 供训练侧拼 prompt。

---

## 3. 数据样本结构

### 3.1 Dataset 列定义

每一行包含三列：


| 列名             | 类型                     | 含义                                                 |
| -------------- | ---------------------- | -------------------------------------------------- |
| `prompt`       | `list[dict[str, str]]` | Chat 消息列表，元素形如 `{"role": "...", "content": "..."}` |
| `env_id`       | `str`                  | 原子环境标识，如 `"PD"`、`"CART"`、`"RETURN"`                |
| `episode_seed` | `int`                  | 用于确定性复现该 episode 的种子                               |


**注意：**

- 数据集中**没有** `completion` / `label` / `answer` 列。
- `env_id` 与 `episode_seed` 会作为额外列保留；TRL `GRPOTrainer` 会将其经 `**kwargs` 传入奖励函数（尤其是 `env_reward`），以便复现同一题并调用环境 `step()`。

### 3.2 `prompt` 消息典型形态

```text
[
  {
    "role": "system",
    "content": "<SYSTEM_PROMPT：角色设定 + 可用工具列表 + JSON action schema>"
  },
  {
    "role": "user",
    "content": "<用户模拟器生成的首轮自然语言请求>"
  }
]
```

`SYSTEM_PROMPT` 要求模型输出合法 JSON，字段包括：

- `assistant_message`：对用户的自然语言回复；
- `tool_calls`：工具调用列表（如 `catalog.search`）；
- `answer`：最终作答结构（含 `env`、推荐商品 ID、`done` 等）。

因此，prompt 侧已经把「动作空间与协议」编码进输入；模型需要在 completion 中遵守该协议。

### 3.3 示例（示意）

以下为结构示意，具体文本随 seed 与难度变化：

```json
{
  "prompt": [
    {
      "role": "system",
      "content": "You are a helpful e-commerce shopping assistant. ..."
    },
    {
      "role": "user",
      "content": "我想找一款降噪蓝牙耳机，预算大概 100 美元以内。"
    }
  ],
  "env_id": "PD",
  "episode_seed": 1042
}
```

训练时，`GRPOTrainer` 会对该 `prompt` 采样 `--num_generations`（默认 4）条 completion，再分别打分。

---

## 4. Catalog：题目所依赖的商品世界

训练数据中的「用户意图」「目标商品」「可检索结果」都建立在环境持有的 catalog 之上。`train_openenv.py` 当前默认路径为：

```python
EcomRLVEEnv(collection=collection, seed=seed)
# catalog 参数为 None → 调用 generate_synthetic_catalog(...)
```

### 4.1 默认合成 Catalog

- **生成函数**：`generate_synthetic_catalog(n_products, seed)`（`src/ecom_rlve/data/catalog_loader.py`）。
- **默认规模**：配置项 `n_synthetic_products`，默认约 **1000** 个商品，并附带若干 `Variant`。
- **设计目的**：无需下载 Amazebay 真实目录即可完成端到端训练与调试；字段与真实数据的内部规范（canonical schema）对齐。

合成分布（实现中的典型设定）包括：

- 价格：对数正态分布（长尾）；
- 评分：Beta 分布映射到 [1, 5]；
- 发货天数：几何分布；
- 库存：约 10% 缺货（`stock_qty=0`），其余有货；
- 类目 / 品牌：从预定义列表均匀采样；
- 每商品通常生成 2–4 个颜色/尺码变体。

### 4.2 内部商品字段（Canonical `Product`）

无论合成或真实加载，内部统一为：


| 字段                                   | 说明                            |
| ------------------------------------ | ----------------------------- |
| `id`                                 | 商品 ID（真实数据常映射自 `parent_asin`） |
| `title` / `desc`                     | 标题与描述                         |
| `cat` / `brand`                      | 类目与品牌                         |
| `attrs`                              | 属性字典（颜色、材质、接口类型等）             |
| `price` / `rating` / `rating_count`  | 价格与评价信号                       |
| `ship_days` / `stock_qty`            | 物流与库存（真实 Amazebay 中常为合成补齐）    |
| `store` / `parent_asin` / `features` | 店铺、父 ASIN、卖点列表                |


检索工具返回轻量 `ProductCard`；完整详情需 `catalog.get_product`。这会影响模型在对话中「先搜后取详情」的工具使用模式，从而间接影响训练时奖励信号。

### 4.3 与真实目录的关系（当前脚本未默认启用）

项目支持通过 `load_catalog()` 加载 Amazebay 等真实目录，并可挂接预构建 FAISS 索引。但 `**train_openenv.py` 并未传入真实 `catalog` 或 `faiss_index_path**`，因此：

- 训练 prompt 中的商品世界默认是**合成目录**；
- 向量检索在合成场景下通常走 debug / mock 嵌入路径（由 `EcomRLVEEnv` 根据是否提供真实 catalog 自动选择），以保证无 GPU 大索引时也能跑通。

若需真实 2M catalog 训练，需要在环境初始化处显式注入 catalog 与索引配置（当前脚本未提供对应 CLI 开关）。

---

## 5. 环境集合（Curriculum）如何影响数据分布

`--collection` 决定 prompt 的任务类型分布：


| Collection | 包含环境                  | 训练数据侧效果                 |
| ---------- | --------------------- | ----------------------- |
| `C1`（默认）   | PD                    | 全部为商品发现 / 推荐类首轮对话       |
| `C2`       | PD, SUB               | 在发现基础上加入缺货替代场景          |
| `C4`       | PD, SUB, CART, RETURN | 加入购物车构建与退货相关出题          |
| `C8`       | 全部 8 个原子环境            | 覆盖状态查询、政策问答、套装规划、多意图旅程等 |


`sample_prompt()` 对 collection 内环境做 round-robin，因此在 `n_prompts` 足够大时，各环境样本数量近似均衡（差不超过 1）。

难度方面：`reset()` 使用自适应难度引擎在每个环境独立维护的区间内采样，因此即便同一 `env_id`，不同行的用户表述复杂度与约束数量也会随 difficulty 变化。

---

## 6. 用户消息如何生成（从隐藏目标到自然语言）

训练数据中的 `user` 文本不是人工写死的模板全集，而是：

1. 原子环境根据 catalog 与难度生成 **隐藏目标**（例如 PD 的目标商品与约束集合）；
2. 用户模拟器结合 persona、信息缺失率 `p_missing`、噪声率 `p_noise` 等难度参数；
3. 通过模板 / verbalization（在更完整配置下亦可接 LLM）生成首轮自然语言请求。

对模型可见的只有这句（或多轮后的）用户话；目标商品 ID、完整约束、订单真相等通常保留在 episode 的 `hidden_goal` 中，供终止时 verifier 打分。

这一设计使得：

- 数据集天然带有「部分可观察」特性（用户未必说全约束）；
- 同一意图可对应不同表述，增加策略需泛化的多样性；
- 奖励可基于隐藏目标做可验证评估，而不依赖另一模型当裁判（主任务分）。

---

## 7. 训练时如何消费这些数据

### 7.1 Prompt 侧

`GRPOTrainer` 读取 `prompt` 列，经 tokenizer chat template 转为模型输入，再采样 completion。

脚本还会根据首条样本估计 `max_prompt_length`（默认取样本 token 数 × 1.2），并在 `max_seq_length` 预算内分配 `max_completion_length`，避免上下文溢出。

### 7.2 Completion 与奖励侧（无标签，但可复现出题）

奖励函数并不从 Dataset 读标准答案，而是：

1. 从模型输出中提取 JSON；
2. `**format_reward**`：检查是否为合法 action JSON；
3. `**tool_usage_reward**`：检查工具名与结构是否合理；
4. `**env_reward**`：用该行的 `env_id` + `episode_seed` 再次 `reset()`，对 completion 执行 `step()`，取环境标量奖励（并 ×5 作为主信号）。

若模型未显式 `done`，评估逻辑会尝试强制构造终局 answer，以便仍能得到任务奖励。

因此，**数据生成阶段写入的 `episode_seed` 不是可有可无的元数据，而是连接「出题」与「判分」的关键键**。

### 7.3 为何只需「首轮 prompt」

当前实现中，每条训练样本对应 episode 起始观察。GRPO 在该设定下主要学习：

- 在首轮正确调用工具 / 组织 JSON；
- 或在简单题上直接给出合理 `answer`。

多轮完整轨迹的强化（持续 `step` 直到终止）在本脚本中被压缩为「对单次 completion 的环境评估」（必要时再补一步 done）。这是对 OpenEnv + GRPOTrainer「先生成、后奖励」管线的适配，也是理解数据形态时需要注意的建模选择。

#### 已知缺陷：单轮生成，弱化多步执行学习

**事实层面：** 可以认为当前训练过程中，**几乎所有训练样本的 prompt 都只包含一句用户消息**（外加固定的 `system` 协议），模型也是**从这一次上下文中一次性生成**整段 JSON completion，而不是在工具返回或多轮用户反馈后再继续决策。

具体表现如下：

1. **输入侧**  
   `sample_prompt()` 组装的 `prompt` 典型形态为 `[system, user]`，其中 `user` 仅为 `reset()` 后的首轮自然语言请求。Dataset 中不包含后续轮次的工具结果或用户追问。

2. **生成侧**  
   `GRPOTrainer` 对每个 prompt 只采样一次（组内多条）completion。训练循环**不会**把 `step()` 之后的观察（工具返回、下一轮用户话）拼回上下文，再让模型生成下一步动作。

3. **判分侧**  
   `evaluate_completion()` 的流程是：同种子 `reset` → 用该 completion **`step` 一次** → 若尚未 `done`，则由脚本**强制构造**一步终局 `answer` 再 `step`，以便拿到任务奖励。环境本身支持多轮对话，但本脚本没有把「观察 → 再生成」接到 GRPO 管线中。

因此需要区分两种「多步」能力：

| 能力 | 当前训练是否覆盖 |
|------|------------------|
| 单轮输出合法 JSON / 正确工具名 | 有（`format_reward` / `tool_usage_reward`） |
| 同一 JSON 内并列多个 `tool_calls` | 协议允许，可能学到「一次调用多个工具」 |
| 检索后根据工具结果再 rerank / 澄清 / 改答 | **基本没有**：生成时看不到工具返回 |
| 多轮根据用户反馈调整策略 | **基本没有** |

**影响层面：** 该设计会对依赖中间观察的**多步执行能力造成损害或使其学不到**。模型更容易学成「在未见检索结果的情况下，一次写出 `tool_calls` 乃至 `answer`」，或「先写工具、未 `done`，再由脚本强制收尾」（终局步并非模型根据中间观察自行推出）。这与真正的工具增强智能体（observe → act → observe → …）差距明显；在 CART、RETURN、JOURNEY 等更依赖迭代交互的环境上，问题通常更突出。

**缓解方向（需改训练管线，而非仅加大 `--n_prompts`）：** 例如每步让模型生成动作，将工具结果与用户反馈拼入下一轮 prompt，再对整条轨迹聚合奖励（多轮 / 轨迹级 RL），才能系统覆盖「看结果再决策」的能力。

---

## 8. 相关命令行参数

与数据生成直接相关的参数如下：


| 参数                        | 默认值    | 作用                                     |
| ------------------------- | ------ | -------------------------------------- |
| `--collection`            | `C1`   | 决定环境集合与任务类型分布                          |
| `--seed`                  | `42`   | 环境主种子；影响合成 catalog 与自适应难度初始状态等         |
| `--n_prompts`             | `1000` | 生成多少条 prompt 行                         |
| `--num_generations`       | `4`    | 每个 prompt 采样的 completion 数（GRPO 组大小 G） |
| `--max_prompt_length`     | 自动估计   | 限制输入长度；`None` 时由首条样本估算                 |
| `--max_completion_length` | `512`  | 生成长度上限（同时受 `max_seq_length` 约束）        |
| `--max_seq_length`        | `2048` | prompt + completion 总预算                |


示例：

```bash
python scripts/train_openenv.py \
  --collection C1 \
  --n_prompts 1000 \
  --seed 42 \
  --num_generations 4
```

---

## 9. 设计要点与局限

### 9.1 设计要点

1. **程序化、可无限采样**：数据量由 `--n_prompts` 控制，不依赖静态文件规模。
2. **种子可复现判分**：`(env_id, episode_seed)` 使 GRPO「生成—奖励」两阶段对齐同一问题实例。
3. **合成 catalog 降低启动成本**：无需先构建 2M FAISS 即可开训。
4. **协议内置于 system prompt**：把工具与 JSON schema 作为输入的一部分，降低动作空间歧义。

### 9.2 当前局限（阅读代码时需知情）

1. **默认非真实电商目录**：合成商品分布与真实 Amazebay 存在域差。
2. **样本仅为 episode 起点**：不是完整多轮轨迹离线数据集。
3. **单轮生成、弱化多步执行（重要）**：prompt 几乎只有一句用户消息；训练时一次生成、判分时最多「一步动作 + 强制终局」，模型看不到工具返回后再决策。单轮并行 `tool_calls` 不能替代多轮闭环，复杂任务（如 CART / RETURN / JOURNEY）受影响更大。详见上文 §7.3「已知缺陷：单轮生成，弱化多步执行学习」。
4. **无离线标准答案**：无法直接做纯 SFT；若需 SFT，应另用轨迹生成脚本（如 `generate_trajectories*.py`）另行产出。
5. **`tokenizer` 传入 `sample_prompt` 但组装消息时未使用**：chat template 应用发生在训练框架侧；采样阶段主要依赖环境与固定 system 文本。

---

## 10. 关键源码索引


| 模块 / 符号                               | 路径                                      | 职责             |
| ------------------------------------- | --------------------------------------- | -------------- |
| `build_dataset`                       | `scripts/train_openenv.py`              | 组装 HF Dataset  |
| `EcomRLVEOpenEnv.sample_prompt`       | `scripts/train_openenv.py`              | 采样单条 prompt    |
| `EcomRLVEOpenEnv.evaluate_completion` | `scripts/train_openenv.py`              | 奖励阶段复现 episode |
| `SYSTEM_PROMPT`                       | `scripts/train_openenv.py`              | 系统角色与 JSON 协议  |
| `EcomRLVEEnv.__init__` / `reset`      | `src/ecom_rlve/server/openenv.py`       | catalog 构建与出题  |
| `generate_synthetic_catalog`          | `src/ecom_rlve/data/catalog_loader.py`  | 合成商品世界         |
| `Product` / `Variant`                 | `src/ecom_rlve/data/schema.py`          | 商品规范字段         |
| `COLLECTIONS`                         | `src/ecom_rlve/training/collections.py` | 课程集合定义         |


---

## 11. 小结

`train_openenv.py` 的训练数据获取方式可以概括为：

> **在选定 curriculum collection 下，对合成（默认）商品目录反复执行确定性 `reset()`，将「系统协议 + 用户模拟器首轮话语」写成 GRPO prompt 行，并用 `env_id` / `episode_seed` 保证后续环境奖励可精确复现同一题。**

它服务的是「可验证电商对话 RL」设定：数据负责出题，奖励负责判分，模型负责在工具增强的动作空间中学习策略，而不是记忆固定标准回复。