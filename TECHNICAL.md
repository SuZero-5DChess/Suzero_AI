# TECHNICAL.md — 5dc_ml 技术文档

## 项目概述

本项目是 5D 国际象棋的 AlphaZero 训练框架，包含：
- 游戏引擎（C++，不可修改）
- AlphaZero 训练框架（Python/PyTorch）
- Web 前端界面

## 2026-07-25 变更：训练步数增强

### 问题

`_training_phase` 中一个 epoch 只做一次 `sample(batch_size)`，本质上是一个 gradient step。10 个 epoch 只有 10 个梯度步。对 572K 参数的 Transformer 来说严重不足。此外 scheduler 的 `T_max` 也是基于旧的 epoch 数，与新步数不匹配。

### 修复：epoch 内多层梯度步 + 动态步数调整

**1. 新增 `steps_per_epoch` 配置**

**文件：** `alphazero/config.py`

在 `TrainConfig` 中新增 `steps_per_epoch: int = 10`，解耦 epoch 语义与 batch 步数。现每个 epoch 执行 `steps_per_epoch` 次独立梯度步。

**2. 重构 `_training_phase` 为双层循环**

**文件：** `alphazero/train.py`

```python
for epoch in range(self.cfg.epochs_per_iteration):
    for _ in range(effective_steps):
        samples = self.replay_buffer.sample(self.cfg.batch_size)
        ...
```

每 iteration 梯度步数 = `epochs_per_iteration × steps_per_epoch` = 10 × 10 = 100 步，提升 10 倍。

**3. 动态步数调整**

当 replay buffer 较小时自动降低有效步数，避免在小数据集上过拟合：

```python
effective_steps = min(
    self.cfg.steps_per_epoch,
    max(1, buffer_size // self.cfg.batch_size),
)
```

- buffer=512 → effective_steps = min(10, 2) = 2
- buffer=2560 → effective_steps = min(10, 10) = 10
- buffer 增长后自动放大到全量 10 步

**4. 同步修正 `T_max`**

**文件：** `alphazero/train.py`

```python
self._total_steps = cfg.num_iterations * cfg.epochs_per_iteration * cfg.steps_per_epoch
```

LR scheduler 的余弦退火周期现在匹配实际的总梯度步数。

---

## 2025-07-26 变更

### C++ 自对弈修复

#### 1. urgency 计算对齐 Python

**文件：** `src/tools/az_selfplay_onnx.cpp`

**问题：** C++ 端 `play_game` 中 `urgency = exp(-0.2 * boards_remaining)`，但 Python 端传的是原始剩余棋盘数 `boards_remaining`。网络内部在 `BoardTokenizer` 中会用 `a*r + (1-a)` 做特征变换，C++ 传了已经衰减过的值导致双重衰减。

**修复：** 改为 `urgency = static_cast<float>(boards_remaining)`，同时移除不再使用的 `kUrgencyAlpha` 常量。

#### 2. 二进制格式清理

**文件：** `src/tools/az_selfplay_onnx.cpp`、`alphazero/self_play.py`

**问题：** C++ 二进制写入 `action_to_squares`、`action_delta_t`、`action_delta_l` 三个字段，但 Python 网络已不再使用这些字段（`score_legal_actions` 签名中已移除），Python 读取时只能跳过。

**修复：** 从 C++ `BinarySampleWriter::write_game` 中移除这三个字段的写入，同时从 Python `_load_games_from_binary` 中移除对应的 skip 行。二进制版本号从 2 升至 3。

#### 3. ONNX 导出修复

**文件：** `alphazero/export_onnx.py`、`alphazero/network.py`

**问题：** `export_onnx.py` 的 `OnnxActionWrapper.forward` 调用不存在的 `self.network.score_legal_actions_batched_flat`，ONNX 导出路径完全断裂。

**修复：**
- `OnnxActionWrapper.forward` 改为内联 scoring 逻辑（直接调用 `self.network.policy_head(board_out)` 后做索引），不再依赖网络方法
- `network.py` 中移除 `score_legal_actions`、`predict`、`predict_actions` 三个方法，各调用方已内联对应逻辑
- 修复 `urgency.reshape(batch_size, 1)` 的形状错误（网络期望 `[B]` 而非 `[B, 1]`）
- 导出方式从 `dynamo=True` + `dynamic_shapes` 改为标准导出 + `dynamic_axes`，避免未使用的 ONNX 输入被剪枝
- 未使用的 3 个 ONNX 输入（`action_to_squares`、`action_delta_t`、`action_delta_l`）通过 no-op 计算链保留在图中，C++ 端发送 14 个输入不会报错

### 当前状态

C++ 自对弈的 ONNX 导出路径已修复，数据结构已对齐。但 C++ 端 MCTS 是 semimove 级别的（与 Python 的 halfstep 级别 MCTS 架构不同），`pending_from` 标记在 C++ 上下文中不适用。

## 2025-07-25 变更

### 清理

#### 1. `score_legal_actions` 移除死参数

**文件：** `alphazero/network.py`、`alphazero/train.py`

`score_legal_actions` 的签名中 `action_to_squares`、`action_delta_t`、`action_delta_l` 三个参数在函数体中从未被引用，是历史遗留的产物（来自之前某版 destination-aware scoring 设计）。已从签名和调用方 (`compute_loss` in train.py) 中移除。

注意：这批字段仍在 `HalfstepRecord` / `collate_samples` / C++ 二进制协议中保留，只清理了 Python 训练侧的无效参数传递。

#### 2. MCTS `select_action` 修复：one-hot → 完整 visit 分布

**文件：** `alphazero/mcts.py`

**问题：** `select_action` 返回的 `policy_probs` 是 one-hot 向量（只有选中的那个 semimove 为 1），而不是标准的 AlphaZero 完整 MCTS visit 分布。这导致训练时 policy loss 的信息量极少——网络只学到"哪个格子被选中了"，没学到"MCTS 认为哪些格子也不错"。

**另一个 bug：** submit 分支的 `policy_probs` 用的长度是 `len(actions)`（源节点数），但 `action_entries` 是 semimove 级别的，长度不同。submit 样本在训练中会被 `abi.numel() != len(pt)` 检查静默跳过，policy loss 完全没学到 submit。

**修复：**
- 新增 `_build_full_policy_distribution()` 方法，从两级 MCTS 树构建完整的 semimove 级别概率分布
- 每个合法 semimove 的 policy 概率 =（源节点 visit 占比）×（目标节点 visit 占比）
- 修正 submit 分支，使用与 `action_entries` 对齐的长度和概率
- 保持动作选择逻辑不变（temperature-based 两级采样决定实际走哪步）

#### 3. destination 节点编码 pending_from 到 board_planes

**文件：** `alphazero/mcts.py`

**问题：** 之前 `encode_state` 不包含 `pending_from` 信息，网络在 source 节点和 destination 节点收到完全相同的输入。网络不知道自己是在"选棋子"还是在"选落点"，只能产出通用的 square saliency 分数，由 MCTS 在外部用不同读法来弥补。

**修复：** 在 `_expand_node` 中，编码后如果 `state.pending_from` 不为空（destination 节点），在 `board_planes` 中将已选中棋子所在位置的**所有通道取反**（1 → -1）。这样网络输入在两种节点下不同，可以学到区分语义的 saliency。

#### 4. MCTS 重构为标准 AlphaZero MCTS

**文件：** `alphazero/mcts.py`、`alphazero/self_play.py`、`alphazero/smoke_test.py`

**彻底重写 MCTS，消除所有非标准的"瞎设计"。**

**MCTSNode：**
- 节点持有 `state`（`HalfstepState`）——标准 MCTS 中每个节点代表一个游戏状态
- 没有 source/destination 节点类型区分——节点是统一的，状态本身决定合法动作
- 标准 PUCT 选择

**MCTS：**
- 标准四阶段：Selection（PUCT）→ Expansion（网络评估+创建子节点）→ Evaluation（网络价值）→ Backup（回传，SUBMIT 处翻转）
- 每步展开时把 `pending_from` 标记到 `board_planes`（前一条的修复）
- 保留转置表缓存（TT），避免相同状态重复展开
- 移除所有 semimove 级别的概念：`_build_semimove_entries`、`capture_action_entries`、`action_entries` 全部删除

**select_action：**
- 接口简化：返回 `(action, policy_probs, root_value)`，不再返回 `action_entries`
- `policy_probs` = root 的 halfstep 级别 visit 分布（标准 AlphaZero 方式）
- 两级采样（source → destination）选择具体走法

**self_play.py：**
- `HalfstepRecord` 移除死字段：`action_to_squares`、`action_delta_t`、`action_delta_l`
- 使用 `_build_root_action_arrays` 从 root 节点构建 action metadata
- C++ 二进制读取跳过遗留字段

**修复：** `export_onnx.py` 的 `OnnxActionWrapper.forward` 改为内联 scoring 逻辑，不再调用网络方法。同时修复了 `urgency.reshape(batch_size, 1)` 导致的形状错误（网络期望 `[B]` 而非 `[B, 1]`），并将导出从 `dynamo=True` 改为标准导出（`dynamic_axes`），避免未使用的 ONNX 输入被剪枝。

## 2025-07-24 变更

### 新增功能

#### 1. 优雅中断机制

**文件：** `alphazero/train.py`

- 在 `Trainer.__init__()` 中注册 SIGINT/SIGTERM 信号处理器
- 收到中断信号后设置 `self._interrupted = True`，当前迭代完成后保存 checkpoint 并退出
- 第二次收到中断信号时强制退出
- 重新运行训练可自动从最新 checkpoint 恢复

**用法：** 运行训练后按 Ctrl+C，训练将在当前迭代完成后优雅退出，所有数据已保存。

#### 2. 持续训练模式

**文件：** `alphazero/train.py`

- 新增 `--continuous` CLI 参数
- 在 `Trainer.train()` 外层包一层 `while not self._interrupted` 循环
- 每完成 `num_iterations` 轮迭代后，若 `continuous=True` 且未中断，继续下一轮
- 日志输出标明当前是第几轮连续训练

**用法：** `python -m alphazero.train --continuous`

#### 3. 实时 Web 看板

**文件：** `serve_ui.py`、`ui/training.html`

**`serve_ui.py` 升级：**
- 添加 `/api/training/tail?path=...&after=...` API 端点
  - 读取指定 JSONL 文件，返回 `iteration > after` 的所有条目
  - 返回 `{ entries, current_iteration, total_games, total_samples }`
  - 文件不存在时返回空数组，不报错
- 添加 COOP/COEP 响应头：
  - `Cross-Origin-Opener-Policy: same-origin`
  - `Cross-Origin-Embedder-Policy: require-corp`
- 保持静态文件服务能力不变

**`ui/training.html` 升级：**
- 自动轮询：每 5 秒调用 `/api/training/tail` 获取新数据
- 增量更新：只追加新数据，不重新创建整个图表
- 状态栏：显示迭代次数、对局数、样本数、白胜/黑胜/和棋数
- 最后更新时间戳
- 自动刷新开关（可暂停/恢复）

#### 4. 终局类型统计与饼图

**文件：** `alphazero/train.py`、`ui/training.html`

**数据记录：**
- `Trainer._self_play_phase()` 中存储 `self._current_terminal_reasons`
- `Trainer._log_metrics()` 将 `terminal_reasons` 和 `outcomes` 写入 JSONL 条目

**JSONL 新增字段：**
```json
{
  "terminal_reasons": {"capture_king": 28, "material": 22},
  "outcomes": {"white_wins": 28, "black_wins": 20, "draws": 2}
}
```

**饼图：**
- 使用 Chart.js doughnut 图展示终局类型分布
- 支持的终局类型：吃王、材料分、将杀、逼和、无合法动作、回合上限、异常终止
- 每个扇形显示类型名、数量、百分比
- 实时更新：每次 poll 到新数据时重绘

#### 5. 启动脚本更新

**文件：** `run_train.bat`

- 使用 `serve_ui.py` 替代 `python -m http.server`
- 添加 `--continuous` 参数
- 移除训练结束后的 `copy` 命令（实时看板直接从源路径读取）
- 启动后自动打开浏览器指向实时看板

## 数据流

```
run_train.bat
  ├── start serve_ui.py (后台) ── 静态文件 (ui/) + API (/api/training/tail)
  │
  └── python -m alphazero.train --continuous
        │  每轮迭代:
        │    ├── _self_play_phase() → 记录 terminal_reasons
        │    ├── _training_phase() → 计算 loss
        │    └── _log_metrics() → 写入 JSONL (含 terminal_reasons)
        │
        ▼  Ctrl+C 时:
        └── _save_checkpoint() → 优雅退出
              ▲
              └── 重新运行 → _load_checkpoint() → 恢复

serve_ui.py
  ├── GET / → 静态文件 (ui/training.html)
  ├── GET /api/training/tail → 读取 JSONL 返回新条目
  └── COOP/COEP headers → SharedArrayBuffer 支持

ui/training.html
  ├── setInterval 5s → fetch /api/training/tail
  ├── 增量更新图表 (loss, stats)
  └── 饼图: 终局类型分布
```

## JSONL 日志格式

文件位置：`alphazero/logs/<variant>/training_log.jsonl`

每个条目包含：
```json
{
  "iteration": 5,
  "timestamp": "2025-07-24T12:00:00",
  "total_games": 250,
  "total_samples": 12500,
  "buffer_size": 12500,
  "iter_time": 45.23,
  "lr": 0.00019,
  "terminal_reasons": {"capture_king": 28, "material": 22},
  "outcomes": {"white_wins": 28, "black_wins": 20, "draws": 2},
  "total_loss": 0.85,
  "value_loss": 0.12,
  "policy_loss": 0.73
}

## 2026-07-25 变更

### 裁剪

#### 1. Transformer 网络参数量大幅精简

**文件：** `alphazero/config.py`、`alphazero/network.py`

**问题：** 默认配置的 6 层 Transformer（d_model=256, d_ff=512, 8 头）共 3.28M 参数，对 Very Small 变体（4×4 棋盘）严重过参数化，96.4% 的参数集中在 Transformer 部分。训练和推理的计算开销大于实际需要的容量。

**变更：**

| 参数 | 原值 | 新值 |
|------|------|------|
| d_model | 256 | 128 |
| n_layers | 6 | 4 |
| n_heads | 8 | 4 |
| d_ff | 512 | 256 |
| 总参数量 | 3,280,201 | **572,041** |

**影响：**
- 参数量减少 **83%**，训练和推理速度显著提升
- 每头注意力维度保持 32（128/4），为标准配置
- 模型架构不变，所有维度通过 `NetworkConfig` 动态适配，无硬编码
- 已有 checkpoint 需要重新训练（维度不兼容）

**变更文件：**
- **`alphazero/config.py`**：更新 `NetworkConfig` 默认值
- **`alphazero/network.py`**：更新 BoardTokenizer 输出维度注释（253→125, 256→128）

---

### 清理

#### 2. SinusoidalPositionEncoding 改为静态预计算表

**文件：** `alphazero/network.py`

**问题：** 之前的实现包含 `nn.Linear(256→256)` 可学习投影层，且每次 forward 都重新计算 sin/cos。正弦编码的本意是固定、无参数的位置信息注入，不应有可学习层。

**修复：**
- 移除了 `self.proj` 线性投影层
- 在 `__init__` 中预计算一张 `[2*max_pos, half]` 编码表（`register_buffer`），L 和 T 共享同一张表
- 表覆盖 `[-max_pos, max_pos)` 范围，forward 时通过 `idx + max_pos` 偏移正确索引负坐标
- `forward` 时直接索引切片，`cat(pe_l, pe_t)` → `[d_model]`，零计算开销
- 现无可学习参数（`sum(p.numel()) == 0`），d_model 偶数时 `half*2 == d_model`，无需投影

#### 2. SinusoidalPositionEncoding 支持负坐标（2026-07-25 补修）

## 2026-07-25 第二波修复

### Bug 修复

#### 1. S1 — MCTS 选择阶段跨回合边界取负 Q 值

**文件：** `alphazero/mcts.py`

**问题：** `_backup` 穿过 SUBMIT 边时取负值（`value = -value`），因为回合边界切换行棋方。但 `best_child` 直接用 `q_value` 做最大化，导致 submit 子节点的 Q 值（对手视角）被当做己方视角最大化。

**修复：** `best_child` 中比较子节点和父节点的行棋方，不同时对 Q 取负：

```python
def best_child(self, c_puct: float) -> "MCTSNode":
    def key(c):
        q = c.q_value
        if c.state.current_player() != self.state.current_player():
            q = -q
        exploration = c_puct * c.prior * math.sqrt(self.visit_count) / (1 + c.visit_count)
        return q + exploration
    return max(self.children, key=key)
```

#### 2. S2 — pending_from 编码训练/搜索对齐

**文件：** `alphazero/env.py`, `alphazero/mcts.py`, `alphazero/self_play.py`

**问题：** MCTS 搜索时 `_expand` 在 destination 节点把 pending 棋子所在格子的 12 个通道全部 `*= -1.0`，但训练样本编码时 `env.encode_state()` 不接受 pending_from 参数。训练数据中 destination 节点与 source 节点编码完全相同，网络在搜索时遇到 -1.0 的分布外输入。

**修复：**
1. `env.py:encode_state()` 加 `pending_from` 参数，当不为 None 时对选中棋子所在格子的所有通道做 `*= -1.0`
2. `mcts.py:_expand()` 改为调用 `encode_state(pending_from=state.pending_from)`，移除手动修改 board_planes 的代码
3. `self_play.py:play_game()` 中编码训练样本时传 `pending_from=game.pending_from`

注意：C++ 后端（`cpp_onnx`）的二进制格式中 `pack_board_planes` 用 `> 0.5f` 二值化，-1 会被变成 0。为此在二进制中新增 `pending_[x/y/t/l]` 四个 int32 字段存储已选中棋子坐标（source 节点用 -1 哨兵），Python 加载器读取后还原 -1 标记到 board_planes。二进制版本号从 3 升至 4。C++ 编译后需将 `Release/az_selfplay_onnx.exe` 复制到 `build_onnx_selfplay/` 根目录。

#### 3. S3 — Python 并行自对弈 payload 不匹配

**文件：** `alphazero/self_play.py`

**问题：** `generate_games()` 构造 9 元 payload（末尾有 `str(self.device)`），但 `_play_games_worker` 解包 8 元。`--sp-workers >1` 时报错。

**修复：** 删除 payload 中的 `str(self.device)` 第 9 个元素。

#### 4. S4 — 缺失 `import engine` 导致 PGN 永远为空

**文件：** `alphazero/self_play.py`

**问题：** `:358` 使用 `engine.SHOW_CAPTURE | engine.SHOW_PROMOTION`，但 `engine` 未导入。`except Exception` 吞掉 `NameError`，PGN 永远为空。

**修复：** 在函数内部延迟导入 engine，捕获 ImportError 后回退到 `show_pgn()` 默认参数。

#### 5. S5 — board_idx == -1 静默回绕

**文件：** `alphazero/self_play.py`, `alphazero/train.py`

**问题：** `_build_root_action_arrays` 中未找到精确 `(l,t,c)` 匹配时 `bi` 保持 -1。`train.py:117` 用 `abi[non_submit]` 索引，-1 在 Python 负索引中取到最后一个有效 board 的 logit。

**修复：**
- `_build_root_action_arrays` 中增加与 `mcts.py._find_board_index` 相同的颜色忽略兜底逻辑
- `compute_loss` 中加 `valid_mask` 防护，对 -1 索引赋予极小 logit（-1e8）

### 改进

#### 6. G1 — Policy target 使用原始访问分布

**文件：** `alphazero/mcts.py`

**问题：** `select_halfstep` 返回的 `policy_probs` 是温度锐化后的分布（`T=0.1` 时近似 one-hot），但标准 AlphaZero 训练目标应该用原始访问计数分布。

**修复：** `select_halfstep` 返回原始访问计数分布（`visits / total_visits`）用于训练，保留温度锐化分布用于动作采样。

#### 7. G2 — 树内各叶子独立计算 urgency

**文件：** `alphazero/mcts.py`

**问题：** `search()` 只在开始时算一次 `urgency = max(0.0, float(board_limit - board_count))`，然后传给所有 `_expand`。树内深层叶子可能有不同的 `board_count`（新开局面的时序线改变了板数）。

**修复：** 把 `board_limit` 传给 `_expand`，在 `_expand` 内从 `node.state.env.board_count` 重新计算 urgency，确保每个叶子用自己真实的剩余棋盘数。

**文件：** `alphazero/network.py`

**问题：** 预计算表只覆盖 `[0, max_pos)`，当 L/T 坐标出现负值时，PyTorch 负索引语义静默地映射到表尾，编码完全错误。

**修复：**
- 表从 `[max_pos, half]` 扩大为 `[2*max_pos, half]`，覆盖 `[-max_pos, max_pos)` 范围
- 新增 `self.pe_offset = max_pos`，forward 中索引加偏移：`pe[l_coords + self.pe_offset]`，`l=-1` → `pe[49]` 取到 `sin(-1·ω)` 的正确编码

### 神经网络输入重构：通道精简 + 行棋方编码

#### 背景

**问题：** 神经网络输入有 14 个通道，其中通道 12 (unmoved flag) 和 通道 13 (occupied mask) 是冗余的：
- unmoved flag 只在王车易位和过路兵中用到，在 5D 棋中基本不需要
- occupied mask 等效于把 12 个棋子通道 OR 起来，网络可以自己学

更重要的是，**网络没有直接获知当前行棋方（side to move）的通道**，只能靠从棋子布局和轮次坐标间接推断。在 5D 棋的时间分叉中这很不靠谱。

#### 变更内容

**输入通道精简：**

| 改动前 | 改动后 |
|--------|--------|
| 14 通道：6 白 + 6 黑 + unmoved + occupied | **12 通道**：6 白 + 6 黑 |

- **`alphazero/variants.py`**：`piece_channels: 14 → 12`
- **`alphazero/config.py`**：更新注释
- **`alphazero/env.py`**：`encode_state()` 移除通道 12/13 的读取和填充
- **`src/tools/az_selfplay_onnx.cpp`**：`kDefaultPieceChannels: 14 → 12`，移除 unmoved/occupied 编码

**行棋方编码：使用 [CLS] token 代替额外通道：**

- 不再用可学习的 `nn.Parameter` CLS token
- [CLS] token 现在编码行棋方：白方走 → **全 0**，黑方走 → **全 1**
- 在 `d_model` 维超平面上，这是两个顶点，每个 transformer 层都可以通过注意力机制读到

**改动文件：**
- **`alphazero/network.py`**：`forward()` 新增 `side_to_move` 参数，删除 `self.cls_token`，用 `side_to_move` 构造 CLS
- **`alphazero/mcts.py`**：`_expand_node` 从 `encoded["current_player"]` 获取行棋方，传给 `predict_actions`
- **`alphazero/self_play.py`**：`HalfstepRecord.player` 记录行棋方，`collate_samples` 返回 `side_to_move` 张量
- **`alphazero/train.py`**：`compute_loss` 将 `batch['side_to_move']` 传给网络
- **`alphazero/export_onnx.py`**：`OnnxActionWrapper` 新增 `side_to_move` 输入
- **`src/tools/az_selfplay_onnx.cpp`**：`EncodedState.current_player`、`BatchRequest.side_to_move`、ONNX `input_names_` 加入 `"side_to_move"`

## 网络结构：`AlphaZeroNetwork`

**文件：** `alphazero/network.py`，**配置：** `alphazero/config.py` (`NetworkConfig`)

### 整体架构

纯 Transformer 编码器架构（非 CNN），默认配置（`very_small` variant，4×4 棋盘）：

| 参数 | 值 |
|------|-----|
| d_model | 128 |
| n_heads | 4 |
| n_layers | 4 |
| d_ff | 256 |
| dropout | 0.1 |
| 棋盘尺寸 | 4×4 = 16 格 |
| 总参数量 | **572K**（2026-07-25 精简，原 3.28M） |

### 各网络块

#### 1. BoardTokenizer
- **输入：** `[B, N, 12, 16]` — B 个局面，每局面 N 个棋盘，每棋盘 12 个平面 × 16 格
- 12 个平面 = 6 种白棋 + 6 种黑棋（已移除 unmoved 标记和 occupied 标记）
- **结构：** `Flatten(192) → Linear(192→125) → LayerNorm → GELU`
- **输出：** `[B, N, 125]`，预留 3 维给 urgency

#### 2. Urgency 特征注入
- **输入：** 标量 `r` = 距离强制终局还剩多少棋盘
- 用三个不同斜率的线性变换 `a*r + (1-a)` (`a = -0.4, -0.05, -0.2`) 编码为 3 维
- `r=1` 时三个值均为 1；`r` 越大以不同速率向 0 衰减
- 拼接在 BoardTokenizer 输出后 → `[B, N, 128]`

#### 3. SinusoidalPositionEncoding（静态预计算表）
- 对每个棋盘的 `L`（时间线）和 `T`（回合）坐标分别做正余弦编码
- 各 64 维 → concat → `[128]`，直接加到 token 上（**无线性投影**）
- 在 `__init__` 中预计算一张 `[2*max_pos, 64]` 编码表（`register_buffer`），覆盖 `[-max_pos, max_pos)` 范围，支持负坐标
- `forward` 时通过 `idx + max_pos` 偏移索引，零 sin/cos 开销，无可学习参数

#### 4. [CLS] Token
- **不是可学习参数**，而是由 `side_to_move` 输入决定：
  - 白方行棋 (`side_to_move=0`)：[CLS] = **全 0**
  - 黑方行棋 (`side_to_move=1`)：[CLS] = **全 1**
- 在 `d_model` 维超平面上，全 0 和全 1 是两个顶点。每层 transformer 通过注意力机制聚合全局信息时，同时读到了当前行棋方
- Value Head 和 Submit Head 只从 [CLS] 位置读取

#### 5. TransformerEncoder（4 层）
- 每层：多头自注意力 (4 头) + FFN (d_ff=256)，GELU 激活
- 支持 `padding_mask` 处理不等长序列

### 输出头

| 头 | 输入 | 结构 | 输出 | 用途 |
|----|------|------|------|------|
| Value Head | [CLS] `[B, 128]` | `Linear(128→64) → GELU → Linear(64→1) → Tanh` | `[-1, 1]` 标量 | 当前走棋方胜率估值 |
| Submit Head | [CLS] `[B, 128]` | `Linear(128→1)` | 标量 logit | 提交（结束回合）倾向 |
| Policy Head | board tokens `[B, N, 128]` | `Linear(128→64) → GELU → Linear(64→16)` | `[B, N, 16]` | 逐格显著性 |

### Policy Head 的核心设计

**Policy Head 输出的 `raw_logits[b, n, s]` 不是"走子选择 logit"，而是每个格子的显著性分数（square saliency）。** MCTS 根据节点类型用不同的方式索引这份 logits：

- **Source 节点**（`pending_from is None`）：读 `raw_logits[board, from_sq]`，表示"这个格子作为棋子来源有多好"
- **Destination 节点**（`pending_from` 已设置）：读 `raw_logits[board, to_sq]`，表示"这个格子作为落点有多好"

两次使用的是**同一份 `raw_logits`、同一个网络前向传播**，只是索引的格子不同。网络没有分别输出"source logits"和"destination logits"，而是学到了一个通用的逐格偏好映射，由 MCTS 根据上下文按需读取。

`score_legal_actions` 是仅供诊断使用的辅助函数，不参与 MCTS 搜索流程。实际的 MCTS 展开逻辑见 `alphazero/mcts.py` 的 `_expand_node` 方法。
```

## 2026-07-25 变更（续）

### 新增：Playout Cap Randomization

**来源：** David J. Wu, *Accelerating Self-Play Learning in Go*, arXiv:1902.10565, 2020

**原理：** 在自对弈中，对每步棋以概率 `p` 执行高质量搜索（`N` 次模拟 + Dirichlet 噪声），以概率 `(1-p)` 执行快速搜索（`n << N` 次模拟、无噪声）。**只有高质量搜索步才记录训练样本**，快速搜索步只推进游戏，不产生样本。这样在相同计算量下可以生成更多对局，同时训练目标仍来自高质量搜索。

**配置项（`MCTSConfig`）：**

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `use_playout_cap_randomization` | `True` | 是否启用 |
| `playout_cap_randomization_p` | `0.25` | 全搜索步的比例 |
| `playout_cap_randomization_fast_sims` | `50` | 快速搜索的模拟次数 |
| `num_simulations` | `800` | 全搜索的模拟次数（原 `200`） |

**改动文件：**

- **`alphazero/config.py`**：`MCTSConfig` 新增上述三个字段
- **`alphazero/mcts.py`**：`MCTS.search()` 新增 `use_full_search` 参数
  - `use_full_search=True`：使用 `cfg.num_simulations` 次模拟 + Dirichlet 噪声（标准行为）
  - `use_full_search=False`：使用 `cfg.playout_cap_randomization_fast_sims` 次模拟，跳过 Dirichlet 噪声
- **`alphazero/self_play.py`**：`SelfPlayWorker.play_game()` 中，每步以概率 `p` 决定 `use_full_search`，仅在全搜索步记录 `HalfstepRecord`（`samples.append`），移动历史记录不受影响

## 2026-07-25 变更：移除 dead data — last_move_markers

### 问题

`last_move_markers`（标记上一手棋的 from/to 位置）在整个 pipeline 中被精心计算、存储、序列化、collate，但 **从未传入网络**：

- `env.py:encode_state()` 计算了 `last_move_markers`
- `HalfstepRecord` 存储了 `last_move_markers` 字段
- `collate_samples()` 将其打包为 `batch['last_move_markers']` tensor
- C++ 自对弈端也计算、打包、写二进制、作为 ONNX 输入喂入
- 但 `AlphaZeroNetwork.forward()` 的签名中 **根本没有 `last_move_markers` 参数**
- `compute_loss()` 调用 `network()` 时也没有传
- ONNX 包装器 `OnnxActionWrapper.forward()` 接收但不使用，属于 ONNX 图中的死输入

在 5D 象棋中，上一步的移动位置是关键的上下文信息（影响了哪些时间线被激活、哪些棋子被移动）。但这个功能的代码从未生效，属于 AI 生成代码遗留的假连接。

### 修复：移除所有 `last_move_markers` 相关死代码

删除的数据流：
- **`alphazero/env.py`**：`encode_state()` 中移除 `last_move_markers` 的初始化、计算（遍历 `last_semimove` 标记 +1/-1）和返回
- **`alphazero/self_play.py`**：`HalfstepRecord` 移除 `last_move_markers` 字段；Python 和 C++ 二进制反序列化中移除读取；`collate_samples()` 移除打包
- **`alphazero/export_onnx.py`**：`OnnxActionWrapper.forward()` 移除 `last_move_markers` 入参；`_build_dummy_inputs()` 移除对应 dummy 输入；ONNX input_names 和 dynamic_shapes 中移除
- **`alphazero/smoke_test.py`**：`HalfstepRecord` 构造调用中移除 `last_move_markers=` 参数
- **`alphazero/test_mcts_unit.py`**：编码字典中移除 `"last_move_markers"` 条目
- **`alphazero/test_cpp_selfplay.py`**：移除所有 `last_move_markers` 的断言
- **`src/tools/az_selfplay_onnx.cpp`**：移除 `EncodedState` 中的字段；移除 `pack_last_move_markers()` 函数；移除 `encode_state()` 中的计算；移除二进制写入；移除 ONNX 批预测中的 tensor 拷贝和输入（减少一个 ONNX 输入）；`input_names_` 从 14 减为 13；自对弈数据版本从 4 升级到 5