# 为什么模型「看起来在做对的事」却无法有效学习

本文档记录一次具体的训练诊断：用户打开落盘的 `prompts.jsonl` 后观察到模型其实发出了带 `catalog.search` 工具调用的回复，但训练过程始终没有任何有效梯度。本文档把这些证据汇总成一份可复现的诊断报告，并给出**修复建议**与**验证步骤**。

适用版本：`scripts/train_openenv.py` + `src/ecom_rlve/server/openenv.py` + `src/ecom_rlve/tools/catalog.py`（默认 synthetic catalog、`collection="C1"`、`Qwen3-1.7B` 或 `Qwen3.5-4B` 均可复现）。

## 修复状态（摘要）

| 根因 | 状态 | 说明 |
|---|---|---|
| §4.1 工具 schema 对模型不可见 | **已选定并实现：方案 B** | 训练侧走 OpenAI/Qwen `tools=` 协议；模型用 XML `<tool_call>`；终局仍用 EcomRLVE JSON |
| §4.2 ProductCard 无 `cat` | **已修复** | `ProductCard` / `product_to_card` 现返回完整 category 路径 |
| §4.3 `r_eff` 激励过早 done | 仍开放 | 未在本次改动 |
| §4.4 `r_task` 无中间态 | 仍开放 | 未在本次改动 |

方案 B **不包含** vLLM guided decoding，也**不包含** GRPO 多轮 agent loop（当前 `evaluate_completion` 仍是单步 + 强制 done）。

---

## 1. 现象描述

### 1.1 控制台观察

训练过程中 reward、format、tool reward 迅速饱和到上限，但 `r_task`、`env_reward` 始终为负，`frac_reward_zero_std=1.0`（一个 GRPO group 内所有 completion 拿到的 reward 相同，**组内方差为零，因此没有梯度**）。

### 1.2 落盘观察（`outputs/ecomrlve_grpo/prompts.jsonl`）

1000 条 prompt + 1000×4 条 result（每 prompt 4 个 GRPO generation）共 ~5000 行 JSONL。逐行聚合后：

```
Total completions : 26
  with tools      : 26  (correct=0)
  without tools   : 0   (correct=0)
termination_reason counts: Counter({'agent_done': 26})

r_task stats: min=-0.900 max=-0.900 mean=-0.900 std=0.000
r_eff  stats: min=0.333 max=0.333 mean=0.333 std=0.000
```

即：

- **模型 100% 在用工具**（`catalog.search`），没有退化到纯 `agent_done`；
- **所有 completion 都在 turn=2 就强制终止**（reason=`agent_done`），从没进入多轮探索；
- `r_task` 与 `r_eff` 在所有数据点上**方差为 0**，GRPO 看不到任何信号。

直观结论是「模型学会了一件事：调一次工具，然后立刻 done」，但这件事**永远**得不到正解。

---

## 2. 复现路径

不需要重跑训练，直接复现单条 prompt 的工具交互：

1. 从 `prompts.jsonl` 里任取一条 `kind: "prompt"` 记录，记下 `episode_seed`；
2. 用 `EcomRLVEEnv.reset(env_id=prompt["env_id"], seed=prompt["episode_seed"])` 重新构造同一回合；
3. 取 `state.hidden_goal` 拿到「正确答案」（`target_product_ids` 与 `constraints`）；
4. 用 `completion_extracted` 里的工具调用参数调用 `catalog.search(...)`；
5. 检查返回结果是否包含 `target_product_ids`。

诊断脚本：`tests/diag_seed_543042.py`，可直接 `.venv/bin/python tests/diag_seed_543042.py` 运行。

---

## 3. 实证：种子 `543042` 的工具调用结果

`prompts.jsonl:1001` 指向的 seed：

```
hidden_goal.target_product_ids = ['syn_000032']
hidden_goal.constraints       = [
    {'attr': 'price', 'op': 'lte', 'value': 35.14},
    {'attr': 'cat',   'op': 'eq',  'value': 'electronics/mobile/tablets'},
]
target product: id=syn_000032 cat='electronics/mobile/tablets' price=29.35
                brand=BrightPath rating=4.5 ship_days=1
User message : "I need a electronics/mobile/tablets. I'd like it under $35.14 ."
```

模型为这条 prompt 发出的 4 次 `catalog.search`（4 个 GRPO 采样）：

| # | query | filters | 返回是否含 `syn_000032` |
|---|---|---|---|
| 1 | `electronics mobile tablets` | `{"price_max": 35.14, "category": ["electronics","mobile","tablets"]}` | True（召回成功） |
| 2 | `tablets` | `{"max_price": 35.14, "category": "electronics"}` | False |
| 3 | `tablets` | `{"category": "electronics", "subcategory": "mobile", "max_price": 35.14}` | False |
| 4 | `mobile tablets` | `{"price_max": 35.14, "category": "electronics"}` | False |

只有第一次的 query 凭文本相似度把目标召回来，但模型无法确认这一点（见 §4）。其余 3 次因为 filter key 全部无效，等价于"无过滤"地被向量召回"碰运气"地错过了 target。

**但即使目标进了 top_k，模型依然 reward=-0.9**。原因：模型在第一个 turn 调一次工具，第二个 turn 立刻 `done=True`，并没有继续 `catalog.get_product(target_id)` 校验或在 `answer.recommended_product_ids` 里写回 id。

---

## 4. 根因分析：四个相互叠加的缺陷

### 4.1 工具 schema 与 SYSTEM_PROMPT 不匹配（训练侧无 OpenAI `tools` 字段）

**事实（诊断时）**：

- `src/ecom_rlve/tools/catalog.py` 中合法 `FILTER_KEYS`（`_apply_filters` 实际匹配的）只有：`cat, brand, color, size, material, connector_type, price_min, price_max, rating_min, rating_max, ship_days_max, rating_count_min, store, wattage_min/max, weight_lbs_max, screen_size_inches_min/max`；
- 模型发出的 key 是 `category`（错的）、`subcategory`（错的）、`max_price`（错的，应为 `price_max`）；
- `cat` 的值必须是**完整的 category 字符串**（如 `electronics/mobile/tablets`），但模型拆成列表 `[...]` 后又被视为未知 key 静默丢弃。

**机制**：训练侧把 dataset 的 `prompt`（`[system, user]` 消息列表）交给 TRL `GRPOTrainer`，经 `tokenizer.apply_chat_template` 渲染后本地 `model.generate`。当时 **没有** 传入 OpenAI 风格的 `tools=` 参数；工具信息只写在硬编码 `SYSTEM_PROMPT` 里，且仅有一行简略签名：

```text
- catalog.search(query, filters, top_k): Search the product catalog
```

而环境侧其实已有完整 Pydantic schema（如 `CatalogSearchArgs`），以及 `ToolRegistry.list_tools()` / JSON Schema —— 但训练路径从未把它们注入模型可见的 prompt。模型只能凭猜补全 `filters` args。

**已选修复：方案 B（完整 tools 协议）** —— 见 §6.1。与「仅手工扩写 SYSTEM_PROMPT」（方案 A）的区别：schema 以 `ToolRegistry` 为单一事实源，经 chat template 的 `<tools>` 块注入；工具调用输出改为 Qwen 原生 XML `<tool_call>`，终局答案仍用 EcomRLVE JSON。

### 4.2 `ProductCard` 返回 dict 不含 `cat`

**事实（诊断时）**：

`catalog_search` 返回的 `ProductCard` 原先不含 `cat` 字段（只有 `product_id/title/price/rating/.../key_attrs`）。诊断脚本里 `card.get('cat')` 全部为 `None`，模型即使把目标搜出来，也无法从搜索结果直接核对 category。

**已修复**：`ProductCard` 增加必填字段 `cat`（完整 category 路径），`product_to_card` 从 `product.cat` 写入。回归测试见 `tests/test_product_card_has_cat.py`。

### 4.3 Reward shaping 激励"立刻 done"

**事实**：

- `r_eff`（效率奖励）在所有 26 条 completion 上**恒为 0.333**，对应 `details.effective_turns=2`；
- 看 `src/ecom_rlve/rewards/composer.py` 可知 `r_eff` 设计为"用的回合越少分越高"（典型在 0~1 之间单调递增到 `T_max` 之前达到饱和）；
- 反过来：**只要模型过早 done（且 answer 不对）也没有惩罚**。`is_correct=False` 完全靠 `r_task` 来体现，而 `r_task` 又因为 4.1/4.2/4.4 一直为 -0.9。

结果：

- 模型学会的最优策略 = "随便调一次 search（甚至不带 filter）→ 立刻 done"；
- 这种策略在 GRPO group 内 4 个 sample 上得到**完全相同**的 reward，组内方差 = 0，没有梯度；
- 训练停滞。

### 4.4 `r_task` 评分对"工具调用成功 + 错误答案"没有中间态

**事实**：

观察 26 条 result 的 `reward_breakdown.details`：
```
"r_task": -0.9, "r_eff": 0.333, "r_hall": 0.0,
"is_correct": 0.0,
"output_ids": [], "seen_ids_count": 20,
```

- `output_ids`（模型 `answer.recommended_product_ids`）恒为 `[]`，即模型从未把任何 id 写进最终答案；
- `seen_ids_count`（模型在工具返回里看到过的 product id 数量）从 2 到 20 不等，**有的 sample 里 target id 已经被工具返回过**（query#1），但模型没有把它写进 answer；
- `r_task` 在所有情况下都是 -0.9（漏掉 target 的固定惩罚），即便"差一点就答对"也无中间奖励。

**模型没机会**学会 "工具返回 → 校验 → 写入 answer" 这条链，因为它从未被奖励过任何「中间正确状态」（比如"看到 target id"奖励 +0.3，"漏掉 target"奖励 -0.9）。

---

## 5. 为什么 GRPO 看起来"饱和"却没有学习

### 5.1 GRPO 的优势函数

GRPOTrainer 对一个 prompt 的 G 个 completion 计算 group-internal advantage：

```
advantage_i = (reward_i - mean(reward_group)) / std(reward_group)
```

如果 4 个 sample 的 reward 全相等（哪怕全 +1），std=0，**所有 advantage = 0**，反向传播的 policy gradient = 0。

### 5.2 当前真实数据上的 advantage

| reward component | std across completions |
|---|---|
| `r_task` | 0.000 |
| `r_eff` | 0.000 |
| `r_hall` | 0.000 |
| `r_total` | 0.000 |
| `format_reward` | ≈ 0.000（4 个 sample 都能解析 JSON 时） |
| `tool_usage_reward` | ≈ 0.000 |

加上 `env_reward * 5.0` 之后：

- `mean_reward_per_step = +1.0 * 5.0 + 1.0 + 1.0 - 0.625 * 5.0 = +3.75`（典型组）
- 但 `std` 仍是 0。

控制台看到 `reward='3.7500'`，**这是均值，不是梯度**。GRPO 实际更新量为 0。`frac_reward_zero_std=1.0` 这一指标是直接证据。

---

## 6. 修复方案（按优先级）

### 6.1 已选定：方案 B — 完整 OpenAI / Qwen tools 协议（针对 §4.1）

**目标**：让模型在生成时看到与环境执行侧一致的工具 JSON Schema，并按 Qwen 原生格式发出工具调用。

**实现要点**：

1. **`ToolRegistry.to_openai_tools()`**（`src/ecom_rlve/tools/registry.py`）  
   把 `list_tools()` 的 name/description/parameters 包装成 OpenAI `{"type":"function","function":{...}}` 列表。

2. **强化 `CatalogSearchArgs.filters` 的 Field description**（`src/ecom_rlve/tools/catalog.py`）  
   在 schema 内写明合法 key、`cat` 必须是完整路径、`price_max` 而非 `max_price` —— 经 `tools=` 渲染后进入 `<tools>` 块。

3. **`ToolsAwareTokenizer`**（`scripts/train_openenv.py`）  
   TRL `GRPOTrainer` 调用 `apply_chat_template` 时不传 `tools=`；包装器默认注入 `to_openai_tools()` 结果，使 Qwen `chat_template.jinja` 渲染 `# Tools` / `<tools>...</tools>`。

4. **双格式 `parse_action`**（`src/ecom_rlve/server/state.py`）  
   - 含 `<tool_call>` → 解析 Qwen XML 为 `ToolCall`；  
   - 否则走原 EcomRLVE 终局 JSON。  
   `format_reward` / `tool_usage_reward` / `env_reward` 已适配。

5. **`SYSTEM_PROMPT` 重写**  
   删除简略工具签名列表；保留角色说明、**通用工作流**（理解目标 → 按 schema 调任意工具 → 读结果 → 证据充分再终局）、以及终局 JSON 协议。不写「仅针对 catalog.search」的专属步骤。

**相对方案 A（仅扩写 SYSTEM_PROMPT）**：schema 不再与执行侧漂移；新增/改工具时自动同步进 prompt。

**本次明确不做**：guided decoding；GRPO 多轮 tool→observation→再生成。

### 6.2 已修复：`ProductCard` 返回 `cat` 字段

在 `src/ecom_rlve/data/schema.py` 的 `ProductCard` 增加必填字段 `cat`，并由 `product_to_card` 写入 `product.cat`。`catalog.search` 返回的每张 card 现在都带完整 category 路径（如 `electronics/mobile/tablets`），模型可在搜索结果上直接核对约束，无需依赖缺失字段。

回归测试：`tests/test_product_card_has_cat.py`。
### 6.3 P1（仍开放）：重新设计 `r_eff`，避免激励"立刻 done"

- 把"回合数 ≤ 1 且 `is_correct=False`"作为一个明确的负向项（如 -0.5），确保"瞎猜 done"比"老实调工具"差；
- 只有在 `is_correct=True` 时 `r_eff` 才奖励"快"；
- 或者把 `r_eff` 改为"使用的工具回合数 / 命中目标需要的最小回合数"，低于 1.0 也扣分。

### 6.4 P2（仍开放）：给 `r_task` 加中间态奖励

当前 `r_task = -0.9` 是「漏 target 的固定惩罚」。可以拆成：

- 把 target 写进 `answer.recommended_product_ids` → +1.0；
- 在工具历史里至少**看到** target id（出现在 `seen_product_ids` 中）→ +0.2（中间态奖励，给梯度）；
- 既没看到 target 也没写 → -0.9（保持原惩罚）。

这样 GRPO 至少能区分"完全瞎猜"和"差一点答对"。

### 6.5 P3（仍开放）：bonus — 显式奖励「验证后再答」

仅当 answer 中的 id 出现在 `state.seen_product_ids` 中（即模型确实通过工具看过这个 id）时，才把 `r_task` 计为正确；否则视作 hallucination 扣分。

### 6.6 P3（仍开放）：`seen` ∩ `target` 中间信号

`r_task` 评分函数可以读取 `state.seen_product_ids` 与 `target_product_ids` 的交集大小，把交集 > 0 但 answer 为空作为一个明确的"差一点"信号。

---

## 7. 验证步骤（修复后如何确认有效学习）

### 7.1 针对方案 B（§4.1 / §6.1）

1. **单元测试**（已落地）：
   - `tests/test_openai_tools_export.py`
   - `tests/test_parse_qwen_tool_call.py`
   - `tests/test_tools_aware_tokenizer.py`
2. **prompt 侧**：训练启动日志应出现 `Injected N tools into chat template`；dump 的 `kind: "prompt"` 行含 `"tools_injected": true`。
3. **completion 侧**：新跑的 `prompts.jsonl` 中 `completion_raw` 应出现 `<tool_call>` **或** 合法终局 JSON；`catalog.search` 的 filters 不应再大量出现 `category` / `subcategory` / `max_price`。
4. **解析侧**：XML tool_call 能被 `parse_action` 转成 `ActionSchema` 并进入 `env.step`。

### 7.2 针对整体有效学习（仍依赖 §6.2–6.4）

1. **重跑 30 step 训练**，打开 `prompts.jsonl`，重新执行 §3 的聚合脚本：
   - `r_task stats` 的 `std` 应 > 0；
   - `r_eff stats` 的 `std` 应 > 0；
   - 至少出现一次 `is_correct=True`。
2. **验证 ProductCard**：`catalog.search` 响应中每张 card 的 `cat` 应为非空完整路径（已由 `tests/test_product_card_has_cat.py` 覆盖）。
3. **验证 GRPO 有梯度**：`frac_reward_zero_std` 显著 < 1.0。

---

## 8. 关键文件与行号速查

| 关注点 | 位置 |
|---|---|
| `SYSTEM_PROMPT`（角色 + 通用工作流 + 终局 JSON） | `scripts/train_openenv.py`（`SYSTEM_PROMPT` 常量） |
| `ToolsAwareTokenizer` | `scripts/train_openenv.py`（`ToolsAwareTokenizer` 类） |
| `ToolRegistry.to_openai_tools` | `src/ecom_rlve/tools/registry.py` |
| `CatalogSearchArgs.filters` 描述 | `src/ecom_rlve/tools/catalog.py` |
| `_apply_filters` 合法 key | `src/ecom_rlve/tools/catalog.py`（`_apply_filters`） |
| `parse_qwen_tool_calls` / 双格式 `parse_action` | `src/ecom_rlve/server/state.py` |
| `ProductCard` / `product_to_card` | `src/ecom_rlve/data/schema.py` |
| `r_eff` / `r_task` 计算 | `src/ecom_rlve/rewards/composer.py` |
| 训练 dump | `scripts/train_openenv.py`（`--dump_prompts`） |
| 落盘路径 | `outputs/ecomrlve_grpo/prompts.jsonl` |

---

## 9. 一句话总结

**模型在「调一次 search + 立刻 done」上达到局部最优，组内 reward 方差为零 → GRPO 无梯度。** §4.1（看不到真实工具 schema）已通过**方案 B**修复；§4.2（搜索结果无 `cat`）已通过 `ProductCard.cat` 修复。要真正出现可学习信号，仍需继续处理 §4.3–4.4（reward shaping）。
