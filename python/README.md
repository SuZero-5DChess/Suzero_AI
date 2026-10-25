# suzero_train · 游戏无关训练核心 + PyTorch 训练器

从 **KataGo** 的训练代码（`python/train.py`）中提取训练方法，但**不依赖围棋、
不依赖 KataGo 的模型或数据格式**。分两层：

1. **抽象层**（`suzero_train/*.py`）：只定义契约与调度，不导入 PyTorch。
2. **PyTorch 后端**（`suzero_train/pytorch/`）：把任意 `torch.nn.Module`、任意
   objective 和任意数据集接入这些契约。

搜索核心仍是仓库根目录的 C++ `suzero_mcts`。本目录的 Python 代码不参与
CMake 构建；唯一的例外是可选的 pybind11 扩展 `_5dchess`，它由根目录
CMake 的 `SUZERO_BUILD_PYTHON_BINDINGS` 生成到
`suzero_5dchess/`。

## 抽象层

| 模块 | 内容 |
|---|---|
| `types.py` | `TrainingBatch`（不透明数据 + `sample_count`）、`TrainStepResult`、`TrainingReport` |
| `data_source.py` | `TrainingDataSource`：`begin_epoch()` / `next_batch()` / `is_exhausted()` / `validation_batches()` |
| `trainer.py` | `Trainer`：`train_step()` / `validate()` / 保存与加载 checkpoint / 计数器 |
| `loop.py` | `TrainingLoop` + `LoopConfig`：epoch、步数上限、验证/日志/checkpoint 节奏 |
| `metrics.py` | `MetricsSink` 及 logging / composite / null 实现 |

`TrainingLoop` 不读取张量内容，只消费 `TrainingBatch.sample_count`，因此调度逻辑与
数据格式、模型结构解耦。

## PyTorch 训练器

```python
import torch
import torch.nn.functional as F

from suzero_train import LoopConfig, TrainingLoop
from suzero_train.pytorch import InMemoryDataSource, TorchTrainer, TorchTrainerConfig


def objective(model, data):
    prediction = model(data["inputs"])
    # KataGo 的反向 loss 是 batch 总和，不是 batch 平均。
    loss = F.mse_loss(prediction, data["targets"], reduction="sum")
    return loss, {"loss_sum": loss}


def collate(samples):
    return {
        "inputs": torch.stack([s[0] for s in samples]),
        "targets": torch.stack([s[1] for s in samples]),
    }


source = InMemoryDataSource(
    samples, batch_size=256, collate_fn=collate,
    validation_samples=validation_samples,
)
trainer = TorchTrainer(
    model,
    objective,
    config=TorchTrainerConfig(device="cuda", warmup_samples=2_000_000),
)
report = TrainingLoop(trainer, source, LoopConfig(epochs=10)).run()
```

`objective(model, data)` 负责前向与 loss，返回 `(标量 loss, metrics)`。训练器负责
autocast、反向、梯度裁剪、优化器调度、指标累计与 checkpoint。模型、batch 的字段
含义、loss 组合全部由调用者决定，因此适用于任意棋类或非棋类游戏。

需要精确复现 KataGo 的 autocast 范围时，可传入 `forward=forward_fn`，让
`forward(model, data)` 只做前向、`objective(model, data, outputs)` 在 autocast
之外计算 loss 与指标（KataGo 只在模型前向使用 autocast）。不传 `forward` 时整个
objective 都在 autocast 内运行。

### 从 KataGo 提取的训练方法

- **per-sample 学习率**：`3e-5 * learning_rate_scale`；AdamW 额外乘 `1.33`；
  再乘 warmup、`(batch / 256) ** batch_lr_exponent` 与参数组 `lr_scale`。
  `batch_lr_scaling` 只控制 LR 的 batch 缩放；exponent 默认 AdamW 为 0.5，
  SGD 为 0（不缩放）。
- **weight decay 独立缩放**：`weight_decay_batch_scaling` 单独控制 WD 的 batch
  缩放，exponent 默认 AdamW 为 0.5、SGD 为 1.0；再乘参数组
  `weight_decay_scale`。另支持 `weight_decay_lr_power`：
  `wd *= (learning_rate_scale * warmup) ** weight_decay_lr_power`，默认 0，
  KataGo 的 bnorm 家族取 0.75。注意：旧版本中 `batch_lr_scaling=False` 会
  同时关闭 WD 缩放；现在两者独立，要关闭 WD 缩放需显式设置
  `weight_decay_batch_scaling=False`。SGD 的 LR 默认 exponent 也从 1.0 改为
  0.0，以匹配上游“SGD 不按 batch 缩放 LR”的行为。
- **分段 warmup**：把 KataGo 前 200 万样本的 8 段曲线（1/20 … 1/1.4）按
  `warmup_samples` 等比缩放；`0` 关闭。
- **参数组**：`ParamGroupSpec` 指定 `lr_scale` 与 `weight_decay_scale`，用于
  输入/主干/输出头等不同学习率或权重衰减。
- **LR/WD 刷新节奏**：默认前 200,000,000 样本每 5 个 step 刷新一次，之后每
  50 个 step，均从每个 epoch 的起点计数；每个 epoch 开始时强制刷新。
- **梯度范数裁剪**：`grad_norm_cap * grad_norm_clip_scale` 乘以
  `sqrt(batch / 256)`，再除以 `sqrt(learning_rate_scale)`，与上游一致；`0` 时
  只记录范数不裁剪（此时直接测量 gnorm，不会用 `max_norm=inf` 调用裁剪，避免
  `inf / inf` 产生 NaN 梯度）。非有限的 gnorm（如 fp16 跳步）会被丢弃，同时
  丢弃由它派生的 `exgnorm_sum`，避免污染运行指标。
- **AMP**：`"none" | "bf16" | "fp16"`，fp16 使用 `GradScaler`；可用
  `forward=` 把前向与 fp32 的 loss 计算分开。
- **loss 约定**：objective 返回的标量 loss 用于反向，默认按 batch 总和理解
  （`loss_reduction="sum"`）。指标缺少 `loss_sum` 时，训练器按该约定补出逐样本
  的 `loss`（`"sum"` 除以样本数，`"mean"` 直接作为逐样本值）。`loss_reduction`
  只影响指标推导，不改变反向所用的 loss；若反向 loss 是 batch mean，应显式设为
  `"mean"`，并相应重新标定 LR 与梯度裁剪上限。
- **指标**：`_sum` 视为 batch 总和、按样本平均；`_batch` 视为已按 batch
  归一化；其余键（如 `nsamp`）按 running sum 累计，与 KataGo 的
  `accumulate_metrics` 一致。运行指标可按 `metrics_decay`（默认
  0.995）指数衰减，并在每个 epoch 开始时清理非有限键。衰减后的运行均值通过
  `TrainStepResult.running_metrics` 暴露给 sink。`_batch` 窗口每
  `batch_metric_window_steps`（默认 100）个 step 按 0.001 收缩一次，与日志频率
  无关；`_sum` 的累计值不受影响。
- **验证 loss**：每个验证 batch 的标量 loss 先按 `loss_reduction` 归一为逐样本
  量，再按样本数加权平均；缺少 `loss_sum` 时自动补齐，因此同一轮验证中部分
  batch 提供 `loss_sum` 不会报错。
- **checkpoint**：原子写入 model / optimizer / scaler / running metrics /
  train_state / last_val_metrics / trainer_config，可从任意 checkpoint 续训。
  若旧 checkpoint 没有 `nsamp` 指标，会用 `global_step_samples` 补齐样本计数。
  加载时校验 optimizer param group 的名称顺序，并重新应用当前
  `ParamGroupSpec` 的 `lr_scale` / `weight_decay_scale`，不会静默沿用
  checkpoint 中的旧 scale。

### `TorchTrainerConfig` 常用字段

| 字段 | 默认值 | 说明 |
|---|---|---|
| `device` | `"auto"` | CUDA → MPS → CPU |
| `optimizer` | `"adamw"` | `"adamw"` 或 `"sgd"` |
| `base_learning_rate` | `3e-5` | KataGo 的 per-sample 基准 |
| `learning_rate_scale` | `1.0` | 全局 LR 缩放，也影响裁剪上限 |
| `batch_reference` | `256` | batch 缩放的参考值 |
| `batch_lr_scaling` | `True` | 是否按 batch 缩放 LR（只影响 LR） |
| `batch_lr_exponent` | `None` | `None` 时 AdamW=0.5、SGD=0.0 |
| `weight_decay_batch_scaling` | `True` | 是否按 batch 缩放 WD（独立于 LR） |
| `batch_wd_exponent` | `None` | `None` 时 AdamW=0.5、SGD=1.0 |
| `weight_decay_lr_power` | `0.0` | WD 随 `learning_rate_scale * warmup` 的幂次；bnorm 家族用 0.75 |
| `warmup_samples` | `0` | warmup 总样本数；0 关闭 |
| `schedule_update_interval_early` | `5` | 早期 LR/WD 刷新间隔（step） |
| `schedule_update_interval_late` | `50` | 后期 LR/WD 刷新间隔（step） |
| `schedule_update_early_until_samples` | `200_000_000` | 使用早期间隔的样本上限 |
| `schedule_batch_size` | `None` | LR/WD/裁剪使用的固定 batch；`None` 用当前 batch |
| `loss_reduction` | `"sum"` | 标量 loss 的解释：`"sum"` 或 `"mean"` |
| `batch_metric_window_steps` | `100` | `_batch` 窗口收缩间隔（step）；0 关闭 |
| `weight_decay` | `0.0` | 基础 weight decay，按组与 LR 幂次再乘系数 |
| `grad_norm_cap` | `0.0` | 大于 0 时启用全局梯度裁剪 |
| `grad_norm_clip_scale` | `1.0` | 裁剪上限的额外倍率（须为正） |
| `metrics_decay` | `0.995` | 运行指标衰减 |
| `amp` | `"none"` | `"none"` / `"fp16"` / `"bf16"` |

KataGo 按 normalization 类型（fixup/bnorm 等）选择 weight decay 的完整表以及
adaptive-norm 因子仍未移植；bnorm 家族可把 `weight_decay_lr_power` 设为 `0.75`
复现其 LR 幂次。请通过 `weight_decay` 与 `ParamGroupSpec.weight_decay_scale`
自行设置各组的 WD 系数。

### 训练循环与数据源

- `TrainingLoop` 在 `max_steps` 恰好落在 epoch 末尾时仍会完成并验证该 epoch；
  判定依赖 `TrainingDataSource.is_exhausted()`，无法廉价判定的数据源会额外
  探测一次 `next_batch()`（此时本次 run 即将结束，最多消耗一个 batch）。
- 数据源在某个 epoch 不产生任何 batch 时训练停止；本次 run 未执行任何 step
  时不会写最终 checkpoint。
- `InMemoryDataSource`：内存样本 + `collate_fn`，每个 epoch 可复现地打乱，
  可带独立验证集。`drop_last=True`（默认）且样本数不足一个 batch 时构造即
  抛出 `ValueError`；`drop_last` 只作用于训练集，验证集总会保留最后一个不完整
  batch。`validation_collate_fn` 可为验证集单独指定 collate；`start_epoch`
  用于续训时复现相同 epoch 的 shuffle 顺序（种子为 `seed + epoch`）。
- `IterableDataSource`：用两个工厂函数包装任意可迭代的 batch 流，适合自对弈
  数据生成或流式读取；其 `is_exhausted()` 只在 `next_batch()` 返回 `None`
  后精确。

## 5D 象棋 AlphaZero（`suzero_5dchess`）

在游戏无关训练核心之上，`suzero_5dchess` 实现了完整的
AlphaZero 自对弈训练闭环。它**不复制**参考项目的代码，只复用同一套
`smallstep_movekey_v1` 协议与 `cls_action_mlp` 网络规格。

| 模块 | 职责 |
|---|---|
| `protocol.py` | 协议常量、`very_small` / `standard` / `standard_turn_zero` 几何与 PGN |
| `env.py` | C++ `Chess5DGameState` 的 Python 包装（`GameEnv`） |
| `sample.py` | 单样本校验、batch padding、SUBMIT 列对齐 |
| `network.py` | PyTorch 网络与 200,000 参数上限检查 |
| `mcts.py` | PUCT 搜索、批量接口、强制 SUBMIT 快捷路径 |
| `self_play.py` | 一盘自对弈与多盘生成（可选线程并发） |
| `replay.py` | FIFO 回放缓冲区与 `TrainingDataSource` 适配 |
| `objective.py` | 掩码策略交叉熵 + 价值 MSE |
| `export_onnx.py` | 动态 B/N/q 的 ONNX 导出与 sidecar |
| `evaluation.py` | 原有训练期 checkpoint 评估路径 |
| `match.py` | 独立 C++ 异构对战，支持 ONNX / checkpoint、每模型搜索参数与换色统计 |
| `pipeline.py` | 训练迭代、checkpoint/续训、JSONL/TensorBoard |
| `cli.py` | `train` / `export` / `evaluate` / `match` 子命令 |

### 构建绑定

```bash
cmake -S .. -B ../build -G "Visual Studio 17 2022" -A x64 \
  -DSUZERO_BUILD_5DCHESS=ON \
  -DSUZERO_BUILD_PYTHON_BINDINGS=ON \
  -DSUZERO_5DCHESS_ENGINE_ROOT="../5dchess_engine-main" \
  -DSUZERO_ONNXRUNTIME_ROOT="C:/path/to/onnxruntime"
cmake --build ../build --config Release
```

绑定是薄适配层：只做 Python 对象与 C++ 状态之间的转换，不含搜索或训练
逻辑。构建前需在当前 Python 环境安装 `pybind11`。构建后
`_5dchess.pyd` 与 `onnxruntime.dll` 直接位于本包内，无需额外设置
`PATH`。

### 网络与参数预算

`d_model` 固定为 128（参考宽度），`n_heads=4`、`n_layers=1`、
`d_ff=256`、`action_mlp_hidden=64`：

- 棋盘 tokenizer：`Linear(192 → 125) + LayerNorm + GELU`
- 每个 token 拼接 3 维 urgency 特征与 (l, t) 正弦位置编码
- CLS token 的值初始化为 `side_to_move`，不额外增加位置参数
- 动作输入：源/目标位置编码 + 源/目标格编码（416 维），经动作 MLP 打分
- 输出：`value`（tanh）、`submit_logit`、`other_logits`

参数量为 **192,058**（`very_small`）。`build_network()` 在构造时检查
`max_params`（默认 200,000），超限直接抛异常，避免训练开始后才发现。
`standard_turn_zero` 使用 8×8 棋盘，参数量 **270,202**；训练 CLI 对该变体
默认使用 **300,000** 预算，显式 `--max-params` 可覆盖。Python API 需显式传
`NetworkConfig(max_params=300_000)`，默认宽度仍为 128。

### 训练流程

每个 iteration：

1. 用当前网络自对弈 `num_games` 盘，按棋盘上限区间、温度和
   playout-cap randomization 采样；强制 SUBMIT 的局面不调用网络、不产生样本。
   每局不设动作数上限，由棋盘上限或吃王、无合法动作等自然终局结束；任一动作使
   棋盘数达到或超过上限时立即按物质分终局，同一小步吃王时仍以吃王为先。
2. 只把完整搜索（未走 fast path）的位置写入 FIFO 回放缓冲区。
3. 从缓冲区均匀采样，做若干 epoch 的掩码策略交叉熵 + 价值 MSE 训练。
4. 按间隔保存原子 checkpoint、导出 ONNX、可选地与参考模型对弈。

策略目标是根节点**原始访问次数**，SUBMIT 固定放在最后一列；训练时
`policy_mask` 屏蔽 padding，因此不同动作数的局面可以混在同一个 batch。
价值目标是当前行棋方视角，支持吃王（±1）、物质分（tanh）等连续终局值。

checkpoint 额外保存 `format`、动作编码、iteration、累计步数/样本数、
变体、网络几何、训练配置与评估状态；回放缓冲区独立保存在每个训练目录的
`replay_buffer.pt` 中并原子覆盖，因此所有 `agent_iter_*.pt` 都不含缓冲区副本。
续训会校验变体与网络几何，并恢复优化器与该目录中的缓冲区；缓冲区文件缺失时
只告警并以空缓冲区继续。

回放样本使用紧凑记录（`sample_codec.py`，`gn5d_replay_zlib_v1`）：棋盘按
“每格一个通道编码”存为 uint8 棋子码，而不是 12 通道 float32 one-hot；
每步动作打包成 6 字节（超长棋局回退 int16），策略与价值目标用 float16，
整条记录再经 zlib 压缩。读取训练 batch 时才展开成 ONNX 所需的 float32
`[B, N, C, S]`，因此标准 8×8 变体的样本从约 60 KiB 降到约 0.5–0.9 KiB
（取决于局面占格数），且解码后策略会重新归一化。

ONNX 导出的三个轴（batch、棋盘数、动作数）都是动态的，sidecar JSON
记录协议、几何、架构与参数量。导出函数可用 `verify=True` 对比
ONNX Runtime 与 PyTorch 的输出。

### 命令行

```bash
python -m pip install -e ".[chess5d]"

# Very Small 小预算测试：2 局、2 simulations、固定 board limit 4
python -m suzero_5dchess train --debug \
  --games 2 --sims 2 --min-board-limit 4 --max-board-limit 4
# TurnZero 小预算测试：2 局、2 次模拟、固定 board limit 4、1 个训练 step
# TurnZero 初始有 2 张棋盘，上限 4 会很快产生正常 material 终局。
python -m suzero_5dchess train --variant standard_turn_zero --debug \
  --device cpu --games 2 --sims 2 \
  --min-board-limit 4 --max-board-limit 4 \
  --batch-size 1 --min-replay-size 1 --steps-per-epoch 1 --record-games
python -m suzero_5dchess train --variant standard_turn_zero  # 正式 T0 训练
python -m suzero_5dchess train --resume latest
python -m suzero_5dchess train --record-games  # 每轮 JSON + 每局 .gn5d 精简棋谱
python -m suzero_5dchess export \
  --checkpoint runs/very_small/cls_action_mlp/latest.pt --output m.onnx
python -m suzero_5dchess evaluate \
  --checkpoint-a runs/very_small/cls_action_mlp/agent_iter_0010.pt \
  --checkpoint-b runs/very_small/cls_action_mlp/agent_iter_0020.pt

# 与训练期评估独立的新入口：每模型不同参数 / ONNX 或 checkpoint
python -m suzero_5dchess match \
  --model-a model-a.onnx --model-b model-b.pt \
  --sims-a 100 --sims-b 200 --games 20 \
  --output match-results.json
```

`match` 由 C++ 对战核心执行：双方独立输入编码、动作映射、ONNX Session 和搜索参数；
`--games` 必须为正偶数，自动交换白黑。独立的 `--max-moves` 默认为 `0`（禁用）；
显式正值才会按每个小步（含 SUBMIT）计数并在达到上限时截断。截断不计作胜、负或和，
不进入胜率统计，效用为 `None`；完整逐步记录仅写入显式指定的 `--output` 新文件。
checkpoint 会独立导出到临时目录，不修改训练产物。详见
[对战接口与使用说明](../docs/heterogeneous-model-match.md)。

`--debug` 等价于 `debug_train_config()`：1 轮、1 epoch、2 step、batch 8、
2 局、2 simulations、固定 board limit 4、关闭 TensorBoard 与 playout cap，输出目录
`runs/debug/`（`very_small`）；其他变体调试输出到 `runs/debug/<variant>/`。
自对弈没有 `max_game_length` 配置或 CLI 参数，每局只由棋盘上限或自然终局结束。
TurnZero 的正式训练目录为 `runs/standard_turn_zero/cls_action_mlp/`，
续训时使用 `train --variant standard_turn_zero --resume latest`。
Windows 可在项目根目录运行 `run_train_turnzero.bat`，复用共享启动器的配置，
与原有 Very Small 训练目录、历史快照和棋谱相互隔离。

## 安装与测试

```bash
cd python
python -m pip install -e .                  # 只依赖 torch
python -m pip install -e ".[chess5d]"  # 5D 训练额外依赖
python -m pytest tests -q                   # 需要 pytest
```

测试用一个微型回归网络覆盖训练、验证、学习率/warmup、参数组、
`reference_group`、loss 约定、梯度裁剪、数据源 collate/epoch 顺序以及
checkpoint 保存与恢复。5D 训练测试另外覆盖参数预算、batch 的 SUBMIT
对齐、自对弈样本合法性、线程并发一致性、ONNX 多形状动态轴、
`run_training()` 的完整调试闭环与续训、评估间隔与对局，以及 MCTS 在
换方边上的价值翻转和真实访问计数。没有编译 C++ 绑定或缺少
`onnx`/`onnxruntime` 时，相关用例自动跳过。

## 未移植的上游功能

- DDP / 多进程多 GPU、跨 rank 指标归并
- Muon / NorMuon / Aurora 优化器
- SWA、lookahead、`torch.compile`、flex attention
- LR cycle / `-lr-schedule` / `-lr-scale-auto2`
- KataGo 按 `norm_kind` 与参数组名选择的完整 weight decay 表，以及
  adaptive-norm 权重衰减因子
- KataGo 的围棋模型、Board 编码、`.npz` 自对弈数据格式与导出流程

这些功能要么依赖围棋专用结构，要么依赖分布式环境，需要时可在此基础上增量补齐。

## 许可证

训练方法的公式与调度源自 KataGo（MIT），但 Python 代码是重新实现的，未复制
上游文件。完整声明见仓库根目录 [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)。
本目录新增代码以 MIT 发布，见根目录 [`LICENSE`](../LICENSE)。
