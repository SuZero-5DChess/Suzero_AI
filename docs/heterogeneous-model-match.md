# 异构模型对战

## 当前状态

已实现独立于训练、自对弈的模型对战路径：

- 游戏无关的 `IMatchGame`、单局 `MatchRunner`；
- 可配置游戏线程数的 `runMatchSeries`，按完整换色配对统计胜、负、和、截断；
- 5dc 的 `Chess5DMatchGame`，用完整 MoveKey / SUBMIT 同步动作；
- 通过 `IChess5DModelInput` 解耦 ONNX evaluator 与具体游戏适配类；
- C++ `runOnnxMatchSeries`、Python `play_model_match` 和独立 `match` CLI；
- 两个 ONNX 模型或两个可信本地 checkpoint 可使用不同网络结构、输入坐标偏移、动作排列、搜索预算和执行设备。

系列对战支持**多线程并局**，默认单独 `match` 命令仍为串行；训练期定期对战默认
20 个游戏线程。并局时每个模型独立使用一个批量推理器和 ONNX Session，不混合不同权重。
不提供 gate 提前判定或训练期自动晋级。
原有 `evaluate` 命令、`--evaluation` 训练评估与 `SelfPlayRunner` 不改用本模块。
新增独立的训练期定期对战选项，默认关闭；`run_train.bat` 不再设置相关旋钮，
需要时通过 `--model-match-*` 显式开启（见下文）。

## 范围与边界

两个游戏适配类必须表示同一套逻辑规则。通用对战核心不理解具体游戏、神经网络张量或 ONNX 协议。

- 支持不同 `IEvaluator`、`SearchParams`、游戏适配类、模型编码与局部动作排列；
- 只支持双人零和效用，`terminalUtility()` 必须从 `currentPlayer()` 视角返回；
- 支持同方连续小步，不假设玩家严格交替；
- 不接入围棋，不引入 Go 的 Board、贴目、劫、ownership、SGF 或计分；
- 不要求两个模型共享输入协议或批量评估器；
- **内置 5dc ONNX 入口**仍限定为 `smallstep_movekey_v1`，非 SUBMIT 候选在前、最多一个 SUBMIT 且位于最后。其他模型协议可以自行实现游戏适配器与 evaluator，调用通用核心。

## 快速使用

先按项目 README 构建 C++ 适配层和 Python 扩展，并安装 `python/` 包。
ONNX 模型旁必须有对应的 `<model>.json` sidecar。

```bash
python -m suzero_5dchess match \
  --model-a models/5dchess/very_small_latest_fp32.onnx \
  --model-b runs/debug/models/very_small_latest_fp32.onnx \
  --name-a reference --name-b candidate \
  --sims-a 100 --sims-b 200 \
  --games 20 \
  --output match-results.json
```

- `--workers N` 设置同时进行的 C++ 对局数，默认 1，实际不超过 `--games`；双方各用独立批量推理器；
- `--games` 必须为正偶数；每两局交换 A/B 的白黑归属，模型及其编码不交换；
- `--max-moves` 是独立的可选小步上限，默认 `0` 表示禁用；显式正值时每次小步和
  SUBMIT 都计数，不同于旧 Python 评估入口的半步上限；
- `--sims-a/b`、`--c-puct-a/b`、`--max-depth-a/b`、`--temperature-a/b` 分别设置；默认温度为 0，无根噪声；
- `--provider-a/b cpu|cuda` 和 `--cuda-device-a/b` 分别选择设备。CUDA 不可用时明确报错，不静默切换为 CPU；
- `--line-shift-a/b` 是模型编码覆盖项，默认读取各自 sidecar 的 `line_shift`，不参与规则兼容性判定；
- `--reverse-moves-a/b` 倒序本地非 SUBMIT 候选，可验证动作排序独立性；
- `--board-limit`、`--material-scale` 是双方共同的规则参数；任一动作后棋盘数达到或
  超过上限都会立即按物质分终局，同一小步吃王时 `capture_king` 优先；
- stdout 输出汇总 JSON，stderr 逐局输出进度；`--output` 保存包含每步共享动作及双方局部 ID 的完整 JSON，拒绝覆盖已有文件；
- Ctrl+C 在局间响应，未完成的系列不会作为完整配对结果返回。

`--model-a/b` 也接受本项目的 `.pt` / `.pth` checkpoint：各自独立加载网络结构并导出到临时目录，不改写 checkpoint 或训练目录。checkpoint 加载应只用于可信本地文件。

也支持参考项目 `5dc_ml` 的 `smallstep_v1` LSTM checkpoint，包括
`very_small_old_version_v1.pt`。推理适配层严格转换 Transformer 权重键，保留原始位置编码表、
board projection、LSTM 与 value/submit/action head；在网络内部将 MoveKey 转为旧版
`[B,q,N,S]` 起点 -1、终点 +1 标记。LSTM 取最后一个真实棋盘的输出，排除 padding。
仅用于推理，不改变本项目的训练网络或参数预算，不依赖参考项目的 Python 包。
缺少/错误动作协议、变体/棋盘几何不匹配、非 LSTM 参考 checkpoint 均拒绝加载。

### 训练期定期对战

显式传入 `--model-match-interval` 后，按该全局训练轮次间隔执行，每次分别对战：

1. **恰好 20 轮前**的模型（`--model-match-lookback`，默认 20）；
2. 固定旧模型 `<path/to/very_small_old_version_v1.pt>`（`--model-match-reference`；
   仓库不包含 checkpoint，需从 `5dc_ml` 项目导出或自行准备）。
   未设置该参数时只进行历史对战。

每个对手默认 20 局，使用 **20 个 C++ 游戏线程并发**，实际线程数不超过局数；
双方模型各自使用独立批量推理器（一个 ONNX Session / server thread），不为每局重复加载模型。
历史对手和固定对手的系列仍依次测量，不同时叠加两套系列的线程/显存预算。
每步 simulations 由 `--model-match-sims`（默认 50）控制；对战棋盘上限由
`--model-match-max-boards`（默认 25）控制，达到或超过上限立即按物质分结束；采样温度固定 0.1（非 argmax，按局种子可复现）、无根噪声、成对换色，
双方预算相同。推理设备由 `--model-match-provider` 选择，默认 `cpu`。
共同规则使用训练配置的 `max_board_limit` 和 `material_scale`，**不使用对手 checkpoint
中保存的自对弈棋盘限制或搜索预算**。这是测量，不是 gate，不自动晋级或回滚权重。

旋钮通过 CLI 传入：`--model-match-interval`、`--model-match-lookback`、
`--model-match-games`、`--model-match-workers`、`--model-match-sims`、
`--model-match-max-boards`、`--model-match-provider`、`--model-match-reference`。
`--model-match-workers` 独立于自对弈的
`SELFPLAY_WORKERS` / `NUM_SERVER_THREADS` / `SELFPLAY_PROCESSES`，设为 1 恢复串行。
例如：

```bat
run_train.bat very_small cls_action_mlp --model-match-interval 10 --model-match-lookback 30 --model-match-games 8 --model-match-workers 8 --model-match-sims 100
run_train.bat very_small cls_action_mlp --model-match-interval 10 --model-match-provider cuda
```

直接使用 Python CLI 时默认关闭，显式传入：

```bash
python -m suzero_5dchess train --resume latest \
  --model-match-interval 20 --model-match-lookback 20 \
  --model-match-games 20 --model-match-workers 20 --model-match-sims 50 \
  --model-match-max-boards 25 --model-match-provider cpu \
  --model-match-reference "/path/to/very_small_old_version_v1.pt"
```

- 按全局轮数取模触发，例如 interval=20 在第 20、40、60 轮执行；续训/连续训练跨批次不重置。
- 开启时在初始/恢复轮次及每轮训练后保存 `matches/checkpoints/agent_iter_XXXX.pt` 轻量快照，
  不包含 optimizer/replay，独立于 `--save-interval`；目前保留全部快照，需按磁盘容量管理。
- 历史目标先查轻量快照，再查原有 `agent_iter_XXXX.pt`。目标轮次为负或不存在时明确记录
  skipped，不用最近 checkpoint 代替；固定对战照常执行。启动时会预检固定 checkpoint。
- 完整结果与逐步动作写入 `logs/matches/iter_XXXX_previous_<id>.json` 和
  `iter_XXXX_fixed_<id>.json`，唯一后缀避免续训或重复测量覆盖。每局还会写成可由
  原生界面重放的 GN5D 棋谱，按终局原因存入 `logs/matches/games/<terminal_reason>/`；
  若配置了 `--gn5d-data-dir`，定期对战 JSON 与棋谱改存到该数据目录的
  `<variant>/cls_action_mlp/matches/`。汇总字段 `gn5d_records` 给出本系列全部棋谱路径。
- 每轮 `training_log.jsonl` 的 `model_matches` 记录双方总计、分颜色统计、对手路径/历史轮次、
  耗时与完整结果路径；最近结果随训练 checkpoint 保存。TensorBoard 使用
  `ModelMatch/previous/*`、`ModelMatch/fixed/*` 分开记录。
- 训练控制台使用多行摘要明确显示当前模型、对手、胜/负/和/截断局数及统计口径。
  控制台只显示正常终局的 `win_rate = wins / 正常终局数`，不再显示容易与材料分混淆的
  `score_rate`；后者仍作为兼容字段保存在结果中。截断局不进入正常终局胜率的分母。
  分颜色胜率会写成当前模型或对手 `as White` / `as Black`，避免 `W/B` 同时可能表示
  胜负或颜色的歧义。
- 终局原因为 `material` 的对局使用其连续材料效用计算平均分；控制台显示参与统计的
  `material` 对局数、当前模型视角的 `material_average_score`，并说明正数代表当前模型领先、
  负数代表落后。该值也写入 TensorBoard。`capture_king` 等其他终局不进入材料平均分。
  多局按 `--model-match-workers` 并发，不复用自对弈并发预算。汇总和完整棋局仍按预定换色顺序保存，
  不按完成先后重排，摘要中的 `workers` 记录实际线程数。
- `--model-match-interval 0` 关闭整个新路径；旧 `--evaluation` 仍可独立显式启用。
  `--debug` 不继承脚本的对战预算，除非另行显式指定这些参数。

单独对固定旧模型对战仍使用原有 `match` 命令：

```bash
python -m suzero_5dchess match \
  --model-a runs/very_small/cls_action_mlp/latest.pt \
  --model-b "/path/to/very_small_old_version_v1.pt" \
  --games 20 --sims-a 50 --sims-b 50 --output old-model-match.json
```

### Python API

```python
from suzero_5dchess.match import MatchPlayer, play_model_match

result = play_model_match(
    MatchPlayer("a.onnx", name="A", simulations=100),
    MatchPlayer("b.pt", name="B", simulations=200, provider="cuda"),
    games=20,
    workers=8,
    max_moves=0,  # 默认：不设小步上限
    board_limit=25,
    seed=20261023,
)
print(result["totals"])     # 按输入 A/B 顺序排列
print(result["by_color"])   # 每个模型各自的 white / black 统计
```

### 结果语义

每个模型的统计均包含 `wins`、`losses`、`draws`、`truncated`：

- 终局效用正、负、零分别计胜、负、和；这是连续物质效用的**符号**，不是将效用当作胜率；
- 只有显式设置正数小步上限且达到上限时，仍非终局的对局才记为截断；效用为
  `None` / `std::nullopt`，单独累计 `truncated`，不计胜、负、和，也不进入胜率或
  材料分统计；达到棋盘上限立即结束；
- 恰在最后一个允许动作结束的游戏仍是正常终局；
- 单局记录保留终局当前玩家、其效用，以及按 A/B 输入索引换算的两个效用；
- 独立 `match` 命令的控制台摘要额外给出按 A/B 顺序排列的 `material_average_scores`，只统计
  `terminal_reason == "material"` 的棋局；
- `moves[*].participant_index` 和 `local_actions` 始终按原始 A/B 顺序，不因换色而改变含义；
- 不兼容规则、动作无法映射或状态分叉直接抛异常，不伪装成某一模型负局。

## 通用接口与实现

### `IMatchGame : public IGameState`

声明位于 [match.h](../include/suzero/mcts/match.h)。除原游戏状态接口外提供：

| 方法 | 契约 |
|---|---|
| `rulesKey()` | 包含真实规则及效用尺度，不包含纯模型输入设置；双方必须一致 |
| `canonicalStateKey()` | 不依赖模型编码或局部动作排列的逻辑局面身份，包含规则相关历史 |
| `toMatchAction(ActionId)` | 将本地稠密动作 ID 转为不透明共享字符串 |
| `fromMatchAction(MatchAction)` | 转回当前局面的本地 ID；无法表示时返回 `kInvalidAction` |
| `terminalReason()` | 可选原因；双方均提供非空值时必须一致 |

`clone()` 必须保留 `IMatchGame` 和原具体适配类型。模型前的检查包装会在内部搜索 clone 丢失类型时拒绝推理，避免把另一种编码对象送入模型。

### `MatchParticipant` 与 `MatchRunner`

每个参与方持有名称、玩家 ID、独立状态工厂、evaluator 和搜索参数。
状态工厂每次必须返回全新、独立持有的状态；双方初始逻辑局面必须相同。
每次搜索至少配置一个正的访问数、playout 数或时间上限。

单局执行流程：

1. 校验规则、逻辑局面、当前玩家及终局信息；
2. 为双方建立独立 `PuctSearch`，按实际 `currentPlayer()` 选择行动参与方；
3. 使用其搜索参数选动作，转换为共享动作，再分别映射到双方；
4. 验证动作合法性及往返映射；
5. 优先 `advanceRoot` 复用子图，未展开时 clone / apply / reset；若进入零候选动作终局，直接保留最终快照，不向搜索传入无须再搜索的根；
6. 检查推进后的两份逻辑状态一致，继续或返回终局 / 截断。

### 配对系列

`MatchSeriesConfig::pairs` 控制完整配对轮数，实际局数为 `pairs * 2`。
`numGameThreads` 控制并局数（默认 1，实际不超过局数）；并发时工厂与 evaluator 必须线程安全。
5dc ONNX 入口自动为每个模型创建独立 `BatchedEvaluator`，各有一个推理服务线程和 Session，
单批上限为 `min(numGameThreads, pairs * 2, 128)`，不会为每局复制模型。
`runMatchSeries` 只交换参与方 `PlayerId`，工厂、evaluator、编码和参数仍归属原模型。
每局及双方搜索由基础 seed 和预定局索引确定性派生随机种子，不依赖线程调度；使用时间预算时不保证运行时序确定。

结果 `games` 始终按预定换色顺序排列，`totals[participant]` 和 `bySeat[participant][seat]` 保持输入顺序。
通用核心的 seat 0/1 代表输入参与方 0/1 最初分配的玩家 ID，**不一定是实际先行动者**；5dc 的固定玩家 0/1 分别为白/黑，因此 Python 对外命名为 `by_color`。

可选 `MatchGameCallback` 在调用线程上按完成顺序串行执行，参数为原始局索引，因此并发时索引可能乱序。
Python `progress(done, game)` 的 `done` 是单调递增的完成数，`game["index"]` 保留原始局索引。
回调或工作线程异常会停止派发新局、等待在途对局退出后向上传播，不返回不完整系列冒充完整结果。

## 5dc 接入

- [match_game.h](../include/suzero/5dchess/match_game.h)：composition 包装现有状态，各自持有独立引擎实例；
- `rulesKey` 包含完整变体 PGN、棋盘尺寸、board limit、material scale；
- `canonicalStateKey` 来自底层 5dc 的精确逻辑状态键，先触发惰性终局判定；
- `stateKey` 额外区分编码配置及局部动作顺序；
- `MatchAction` 序列化完整八坐标 MoveKey 或 SUBMIT，不直接共享本地动作 ID；
- `IChess5DModelInput` 只提供 `actions()` 和 `encodePosition()`，单条与批量 ONNX 路径均不再硬转换为 `Chess5DGameState`；
- [match.h](../include/suzero/5dchess/match.h) 的 `runOnnxMatchSeries` 给双方创建独立 ONNX Session；
- Python `MatchPlayer` 支持不同网络架构、模型路径、设备与搜索设置，checkpoint 导出仅发生在新临时目录。

## 测试与后续

已增加：

- `suzero_match_tests`：两个不同具体玩具适配类、不同排列、独立 evaluator、连续行动、终局视角、截断、错误协议及配对统计；不依赖 5dc 或 ONNX；
- `suzero_5dchess_match_tests`：共享动作同步、SUBMIT、编码差异、第二具体适配类、单条 / 批量 ONNX 和终局系列；
- `python/tests/test_model_match.py`：真实导出两个不同宽度网络、不同输入坐标偏移、独立搜索参数、checkpoint 临时转换、配对统计、截断及 CLI。

后续扩展仍保持独立模块：

1. 每模型可配置的批大小与推理服务线程数；
2. 显式配置平局处理的 gate 判定；
3. 其他参考 checkpoint 架构或其他 5dc 输入 schema 的专用适配器 / evaluator。

## KataGo 参考来源

参考目录：`5dc_ml` 项目内的 `katago/KataGo-master`。
借鉴 `cpp/program/play.h` / `play.cpp` 的 BotSpec、MatchPairer 和 GameRunner，
`cpp/command/match.cpp` 的配对换色思想，以及 `gatekeeper.cpp` 按模型颜色归属统计的方式。
核心原则是每方独立搜索与 evaluator，同一逻辑动作同步到双方状态；没有复制围棋专用规则或计分代码。
