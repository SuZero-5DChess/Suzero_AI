# SuZero AI · 5D 国际象棋 AlphaZero

SuZero AI 是一个面向 **5D 国际象棋（5D Chess with Multiverse Time Travel）** 的
AlphaZero 风格 AI。规则适配、PUCT 图搜索、自对弈、PyTorch 训练、ONNX 导出、
模型对战和棋谱回放都在本仓库内打通，目标是把这一种游戏从训练到对弈做成完整闭环。

仓库内部的搜索核心仍保持与游戏无关的接口设计，便于独立测试、替换规则和复用组件；
但本项目只服务 5D 国际象棋，不打算成为通用棋类框架。

> **本项目大量借鉴并派生自 [KataGo](https://github.com/lightvector/KataGo)。**
> 搜索侧直接派生自 KataGo `cpp/search/`：PUCT 选择、FPU、根节点 Dirichlet 噪声、
> 访问权重聚合、图搜索（置换合并、边访问追赶、成环处理、换根复用）等；训练侧的
> per-sample LR、warmup、参数组、梯度裁剪、指标累计和 checkpoint 组织，
> 重新实现了 KataGo `python/train.py` 的方法。KataGo 使用 MIT 许可证，上游版权
> 声明保留在 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，逐项对应关系见
> [docs/katago-mapping.md](docs/katago-mapping.md)。本项目新增代码同样以 MIT 发布，
> 见 [LICENSE](LICENSE)。

- 规则引擎：项目内固定的 [5dchess_engine](https://github.com/ftxi/5dchess_engine)
  源码树（BSD-2-Clause），通过外层适配器调用，不修改引擎核心
- 搜索与推理：C++17 搜索核心 + C++23 5D 适配层 + ONNX Runtime（CPU / CUDA）
- 训练：Python 3.10+ / PyTorch；自对弈可使用进程内 PyTorch 或 C++ ONNX 后端
- 构建：CMake ≥ 3.16；主要开发环境为 Windows 11 + Visual Studio 2022

## 规则变体与模型规模

| `--variant` | 规则 | 棋盘 | `cls_action_mlp` 参数量 | 说明 |
|---|---|---|---|---|
| `very_small` | Very Small - Open | 4×4 | **192,058** | CLI 默认；200,000 参数预算 |
| `standard_turn_zero` | Standard - Turn Zero | 8×8 | **270,202** | `run_train.bat` 默认；训练预算 300,000，自对弈默认 1000 sims |
| `standard` | Standard | 8×8 | **270,202** | 需显式 `--max-params 300000` 或更高 |

`standard` 与 `standard_turn_zero` 共用 8×8 网络几何，区别只在开局状态：
Turn Zero 开局包含 T0 黑方历史棋盘和 T1 白方棋盘。不同变体的 checkpoint 不能
直接互相 `--resume`。

---

## 1. 能力概览

### 1.1 5D 规则与搜索

- **规则适配**：`Chess5DGameState` 封装 5dchess_engine 的 Very Small - Open、
  Standard 和 Standard - Turn Zero 规则。
- **动作空间**：动作是小步（8 元 `MoveKey`）或 `SUBMIT`，候选由当前局面生成；
  `SUBMIT` 固定排在最后。只有 `SUBMIT` 一个候选时走强制快捷路径，既不调用网络
  也不运行搜索。
- **终局判定**：吃王（`capture_king`）、任一动作后棋盘数达到或超过上限的物质分
  （`material`）、无合法动作（`no_legal_action`）。同一小步既吃王又触及棋盘上限时，
  吃王优先。
- **KataGo 风格 PUCT 图搜索**：按状态键合并置换节点，边只引用子节点；根节点不入
  节点表，换根后不会出现经过根的环。
- **搜索细节**：FPU、utility 标准差缩放、根策略温度（含早期/后期插值）、
  策略形状 Dirichlet 噪声、3 自由度 Student-t 坏子降权、非 LCB 回顾式着法权重、
  subtract/prune、边访问追赶与路径成环检测。
- **搜索控制**：访问数 / playout 数 / 时间 / 深度限制，以及外部停止条件接口；
  支持 `advanceRoot` 复用子树。
- **批量推理与多局并行**：`BatchedEvaluator` + `SelfPlayRunner`，每局独占搜索、
  共享评估器；对齐 KataGo 的 `numGameThreads` / `NNEvaluator` 模型。
- **异构模型对战**：双方可以使用不同的游戏适配器、evaluator、搜索参数和模型格式，
  通过共享逻辑动作同步；支持完整换色配对、分颜色统计和截断计数。

### 1.2 自对弈与训练

- **完整 AlphaZero 流水线**：自对弈 → 回放缓冲区 → 训练 → 导出 ONNX → 评估。
- **网络架构 `cls_action_mlp`**：棋盘 tokenizer、(l, t) 正弦位置编码、transformer、
  价值/SUBMIT/动作头；纯 PyTorch 实现。
- **自对弈策略**：Dirichlet 根噪声、温度调度、playout-cap randomization、
  强制 `SUBMIT` 不调用网络；训练目标为掩码策略交叉熵 + 价值 MSE。
- **两种自对弈后端**：
  - `python`：进程内 PyTorch MCTS，适合调试和 CPU；
  - `cpp_onnx`：每轮导出 ONNX，由 C++ 多线程 PUCT + ONNX Runtime 执行，
    支持 CPU / CUDA，并可多进程扩展。
- **训练器**：沿用 KataGo 的 per-sample LR、warmup、参数组、梯度裁剪、AMP、
  指标累计、原子 checkpoint 与断点续训。
- **训练辅助**：TensorBoard、JSONL、定期模型对战、参考旧模型兼容、协作式 Ctrl+C。
- **训练核心可复用**：`suzero_train` 抽象层不依赖 PyTorch，`suzero_train.pytorch`
  可接收任意 `torch.nn.Module` 与 objective；它与 C++ 目标完全解耦，
  详见 [python/README.md](python/README.md)。

### 1.3 对弈、导出与棋谱

- **ONNX 协议**：`smallstep_movekey_v1`（7 输入 / 3 输出）；评估器校验 sidecar、
  IO 名称/类型/秩、几何尺寸，并拒绝旧协议模型。
- **导出与评估**：`export` 导出动态 batch / 棋盘数 / 动作数的 ONNX；
  `evaluate` 对弈两个 checkpoint；`match` 进行独立 C++ 异构模型对战。
- **GN5D 棋谱**：紧凑 v2 JSON，只存初始 FEN、动作序列、结果和元信息，
  其余由原生引擎重放推导；兼容旧 `5dc.capture-king` v1 和 `.5dck`。
- **回放 UI**：`run_ui.bat` 启动原生 5dchess_engine UI，导入 `.gn5d` 逐步回放，
  不另写棋盘或棋规。

### 1.4 未实现或不在范围内

- 单局内多线程搜索（virtual loss、原子统计）；并行度来自多对局线程
- 评估缓存（KataGo 的 eval cache）
- 人类 SL 模型、mirror 对抗、visit cap、时间控制策略
- 模型热切换与跨进程自对弈训练循环（KataGo `SelfplayManager` 部分）
- 从 KataGo 移植的围棋专用逻辑（目数、领地、ownership、贴目、劫、对称性、
  停一手）

这些是后续可独立扩展的方向。搜索算法与 KataGo 的逐项映射、以及刻意未移植的
部分，见 [docs/katago-mapping.md](docs/katago-mapping.md)。

---

## 2. 快速开始

### 2.1 依赖

- Windows 11 + Visual Studio 2022（C++23 支持）
- CMake ≥ 3.16
- Python 3.10+，并安装 `torch`、`numpy`、`onnx`、`onnxruntime`
- ONNX Runtime 预编译 SDK（CPU 即可；CUDA 需要 `onnxruntime-gpu`、CUDA 12.x、
  cuDNN 9）
- 构建 Python 扩展需要 `pybind11`（`python -m pip install pybind11`）

### 2.2 构建 C++ 核心、5D 适配层与 Python 扩展

```bat
set SUZERO_ONNXRUNTIME_ROOT=C:\path\to\onnxruntime-win-x64-1.24.3
configure.bat
cmake --build build --config Release
```

`configure.bat` 会自动打开 `SUZERO_BUILD_5DCHESS` 和
`SUZERO_BUILD_PYTHON_BINDINGS`，并从脚本自身位置推导全部项目内路径，整个目录
移动后直接重跑即可。

- 不把 `SUZERO_5DCHESS_ENGINE_ROOT` 写进缓存（全新配置自动选中项目内的
  `5dchess_engine-main`）。
- 检测到 `build/` 属于其他源码路径时自动追加 `cmake --fresh`。
- ONNX Runtime SDK 也可解压到 `extern/onnxruntime-win-x64-1.24.3/`
  （该目录不入库），脚本会自动发现。
- 扩展输出到 `python/suzero_5dchess/`；导入时由
  `suzero_5dchess/_ort_runtime.py` 把已安装 `onnxruntime` 包的 `capi` 目录加入
  Windows DLL 搜索路径。需要指向其他运行时时设置
  `SUZERO_ORT_RUNTIME_DIR`（优先于已安装的包）。
- `cpp_onnx` 的 `cuda` 模式还需要 CUDA 12.x 与 cuDNN 9；`torch/lib` 中的
  cuDNN 会被 `import torch` 自动加入 DLL 搜索路径。

也可以手动指定构建树：

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
  -DSUZERO_BUILD_5DCHESS=ON ^
  -DSUZERO_BUILD_PYTHON_BINDINGS=ON ^
  -DSUZERO_ONNXRUNTIME_ROOT=C:\path\to\onnxruntime-win-x64-1.24.3
cmake --build build --config Release
```

### 2.3 安装 Python 包

```bash
cd python
python -m pip install -e ".[chess5d]"
```

### 2.4 冒烟测试与正式训练

```bash
# 冒烟测试：2 局、2 simulations、固定 board limit 4、1 轮训练，CPU 即可
python -m suzero_5dchess train --debug \
  --games 2 --sims 2 --min-board-limit 4 --max-board-limit 4

# 正式训练（默认 very_small）
python -m suzero_5dchess train --iterations 100 --games 64 --sims 800

# C++ 自对弈后端：CUDA 推理 + 3 个对局线程 + 2 个评估服务线程
python -m suzero_5dchess train --selfplay-backend cpp_onnx \
  --cpp-onnx-provider cuda --workers 3 --num-server-threads 2

# 从 latest 续训
python -m suzero_5dchess train --resume latest

# 连续训练：每 20 轮为一批，自动从 latest 续训，直到 Ctrl+C
python -m suzero_5dchess train --iterations 20 --continuous
```

`--debug` 使用 `very_small`、1 轮、1 epoch、2 step、batch 8、2 局、2 simulations，
并把每局棋盘上限固定为 4；在 `runs/debug/` 下生成 checkpoint 与可被 C++ 评估器
直接加载的 `models/very_small_latest_fp32.onnx`。自对弈不设动作数截断，每局由
棋盘上限或其他自然终局结束。

训练中按一次 Ctrl+C 会等当前轮（自对弈 + 训练 + 评估）结束后保存并退出，再按一次
立即中止。按键后控制台会立刻打印“已收到中断信号”，`cpp_onnx` 后端整批自对弈期间
也一样。

---

## 3. 训练

### 3.1 训练循环

每轮训练包含：

1. 用当前网络自对弈 `--games` 局；
2. 把样本写入回放缓冲区（默认上限 50,000，至少 512 条后才开始训练）；
3. 训练 `--epochs-per-iteration` 个 epoch、每 epoch `--steps-per-epoch` 个 step；
4. 按间隔保存 checkpoint、导出 ONNX，并可选运行定期模型对战。

默认 `TrainConfig`：100 轮、64 局/轮、800 simulations、batch 256、
10 epoch × 10 step、学习率 2e-4、weight decay 1e-4、梯度裁剪 1.0。
`run_train.bat` 会覆盖其中一批默认值（见 3.4）。

checkpoint 保存网络几何，回放缓冲区独立存于训练目录的 `replay_buffer.pt`，
续训会校验变体与 `d_model` 等配置。checkpoint 写入是原子的。

### 3.2 自对弈后端

| 后端 | 执行位置 | 适用场景 |
|---|---|---|
| `python` | 进程内 PyTorch MCTS，`--workers` 个 Python 线程 | 调试、CPU、小规模 |
| `cpp_onnx` | 每轮导出 ONNX，C++ PUCT + ONNX Runtime，`--workers` 个游戏线程 | 正式训练、CUDA、大规模 |

`--num-server-threads` 控制 `cpp_onnx` 后端 C++ ONNX 评估器的服务线程数
（默认 1）。它与 `--workers`（并发对局数）相互独立：每个服务线程各自持有一个
独立的 ONNX Session，因此增加线程会带来额外的内存占用，CUDA 模式下还会增加
显存占用；实际增量取决于模型与会话配置，不保证与线程数严格成比例。
增加服务线程并不保证提速，是否需要取决于实际负载与设备。

`--selfplay-processes` 选择自对弈工作进程数，默认 `1`，保持原有的单进程路径。
大于 `1` 时只能配合 `--selfplay-backend cpp_onnx`，且 `--workers`、
`--num-server-threads`、`--games` 都必须不小于进程数。`--workers` 与
`--num-server-threads` 始终是总预算，由各进程分摊，不要求整除，也不会自动翻倍
（例如 50 workers / 2 processes 可分为 25+25；3 server threads / 2 processes
可分为 2+1）。例如：

```bash
python -m suzero_5dchess train --selfplay-backend cpp_onnx \
  --cpp-onnx-provider cuda --workers 50 --num-server-threads 2 \
  --games 300 --selfplay-processes 2
```

这是训练入口内的可选多进程自对弈；`python/experiments/multiprocess_selfplay/`
下的 benchmark 是独立性能基准，不属于训练路径。默认 `1` 即回退到原路径。

`cpp_onnx` 每轮把当前网络导出到
`runs/<variant>/cls_action_mlp/selfplay/model_iter_XXXX.onnx`，只保留最近 2 份。

### 3.3 规则变体

#### `very_small`（CLI 默认）

Very Small - Open，4×4 棋盘。`d_model=128` 时参数 192,058，构建时强制不超过
200,000。训练产物默认在 `runs/very_small/cls_action_mlp/`。

#### `standard_turn_zero`（`run_train.bat` 默认）

对应引擎 `[Board "Standard - Turn Zero"]`：8×8 棋盘，开局保留 T0 黑方历史棋盘和
T1 白方棋盘，并非普通 `standard` 开局。游戏引擎、搜索算法和训练目标不变。

- CLI 选择该变体时默认把参数预算设为 **300,000**，保留 `d_model=128`，
  实际参数量 **270,202**；显式 `--max-params` 仍优先。
- 自对弈默认每步 **1000** 次完整模拟（`--debug` 小预算预设和显式 `--sims` 仍优先）；
  快速搜索仍由 `fast_simulations`（默认 50）控制。
- checkpoint：`runs/standard_turn_zero/cls_action_mlp/latest.pt`，再次运行自动续训。
- ONNX：同目录的 `models/standard_turn_zero_latest_fp32.onnx`（及 `.json` sidecar）。
- 调试目录：`runs/debug/standard_turn_zero/`，不会误续训其他变体的模型。
- 不自动使用 Very Small 的固定旧模型；可对战同变体的历史快照。变体不同的
  checkpoint 即使同为 8×8，也不能直接 `--resume`。

小预算测试命令（已安装 Python 包；或先在项目根目录设置 `PYTHONPATH=python`）。
只跑 **2 局、2 次模拟、固定棋盘上限 4、1 轮训练**，完成后退出，不会继承批处理
脚本的连续训练开关。TurnZero 初始有 2 张棋盘，因此上限 4 可快速触发正常的
`material` 终局：

```bash
python -m suzero_5dchess train --variant standard_turn_zero --debug \
  --device cpu --games 2 --sims 2 \
  --min-board-limit 4 --max-board-limit 4 \
  --batch-size 1 --min-replay-size 1 --steps-per-epoch 1 --record-games
```

#### `standard`

Standard 8×8 开局，网络几何与 `standard_turn_zero` 相同（270,202 参数），但
超过默认 200,000 参数预算，必须显式放宽：

```bat
run_train.bat standard cls_action_mlp --max-params 300000
```

### 3.4 `run_train.bat`

Windows 下的训练入口，默认配置：

| 旋钮 | 默认值 | 说明 |
|---|---|---|
| `VARIANT` | `standard_turn_zero` | 位置参数 1 可覆盖 |
| `ARCHITECTURE` | `cls_action_mlp` | 位置参数 2，当前唯一支持值 |
| `CONTINUOUS` | `1` | 每批结束后自动从 `latest.pt` 续训，直到 Ctrl+C |
| `SELFPLAY_BACKEND` | `cpp_onnx` | 或 `python` |
| `CPP_ONNX_PROVIDER` | `cuda` | 或 `cpu` |
| `SELFPLAY_WORKERS` | `100` | 并发对局/线程总数 |
| `NUM_SERVER_THREADS` | `4` | C++ ONNX 评估器服务线程总数 |
| `SELFPLAY_PROCESSES` | `4` | 自对弈工作进程数 |
| `SELFPLAY_GAMES` | `1000` | 每轮自对弈局数 |
| `MIN_BOARD_LIMIT` / `MAX_BOARD_LIMIT` | `55` / `90` | 每局采样的棋盘数范围 |
| `--iterations` | `20` | 每批轮数 |
| `--sims` | `1000` | 每步完整模拟数 |
| `EVALUATION_INTERVAL` | `2` | 每 N 轮运行一次定期模型对战；0 关闭 |
| `MATCH_LOOKBACK` | `2` | 对手恰好早 N 轮 |
| `MATCH_GAMES` | `20` | 每个对手的对局数 |
| `MATCH_WORKERS` | `20` | 每个对手的并发 C++ 对局数 |
| `MATCH_SIMS` | `1000` | 对战 simulations |
| `MATCH_MAX_BOARDS` | `50` | 对战棋盘上限 |
| `MATCH_PROVIDER` | `cuda` | 或 `cpu` |
| `MATCH_REFERENCE` | 空 | 可选的固定对手 `.pt` / `.onnx` 路径（仅 `very_small`） |

脚本顶部的值可直接修改，命令行追加的参数会覆盖默认值：

```bat
REM 默认变体与默认配置
run_train.bat

REM 切回 very_small，修改 simulations 与轮数
run_train.bat very_small cls_action_mlp --sims 400 --iterations 5

REM 使用 CPU
run_train.bat very_small cls_action_mlp --device cpu

REM 调试预设
run_train.bat very_small --debug

REM 两个自对弈进程
run_train.bat standard_turn_zero cls_action_mlp --selfplay-processes 2

REM 关闭定期对战
run_train.bat standard_turn_zero cls_action_mlp --model-match-interval 0
```

服务线程数与并发对局数相互独立，每个服务线程各持有一个 ONNX Session，调高会增加
内存（CUDA 下还有显存）占用，并不保证提速，需按实际设备实测。

`run_train.bat` 会自动设置 `PYTHONPATH`、创建控制台日志目录、检测 Python 与
`_5dchess` 扩展；若 `runs/<variant>/cls_action_mlp/latest.pt` 存在，自动附加
`--resume latest`。每次运行结束后会保留窗口并打印日志与 checkpoint 位置。

### 3.5 训练中定期模型对战

`run_train.bat` 按 `EVALUATION_INTERVAL` 指定的轮次间隔，分别对战恰好
`MATCH_LOOKBACK` 轮前的模型与 `MATCH_REFERENCE` 指定的固定旧模型（仅
`very_small` 且显式设置时）。每个对手 `MATCH_GAMES` 局换色对战，使用
`MATCH_WORKERS` 个并发 C++ 游戏线程，双方各自独立批量推理。

间隔和历史跨度独立设置：

```bat
run_train.bat very_small cls_action_mlp --model-match-interval 10 --model-match-lookback 30 --model-match-games 8
REM 关闭定期对战：
run_train.bat very_small cls_action_mlp --model-match-interval 0
```

`MATCH_PROVIDER` 可独立于自对弈和训练设备选择 `cpu` / `cuda`。该路径独立保存
每轮轻量快照，不依赖训练 checkpoint 保存间隔；历史缺失时只跳过历史对战。
完整棋局 JSON 写入训练日志目录的 `matches/`，每局同时生成可重放的 `.gn5d`，
保存到 `matches/games/<terminal_reason>/`；设置 `--gn5d-data-dir` 时，这两类文件
位于该数据目录下的 `<variant>/cls_action_mlp/matches/`。汇总中的 `gn5d_records`
列出对应棋谱路径。

对战汇总进入 `training_log.jsonl` 的 `model_matches` 和 TensorBoard 的
`ModelMatch/previous/*`、`ModelMatch/fixed/*`。控制台会打印总胜率，以及当前模型
和对手分别持白、持黑的胜率；这些颜色胜率只统计对应颜色的正常终局。截断与和棋
分开统计；没有自动晋级或权重回滚。直接使用 Python CLI 时需显式开启
`--model-match-interval`。参数、历史选择和旧模型兼容范围见
[定期对战说明](docs/heterogeneous-model-match.md#训练期定期对战)。

---

## 4. 模型导出、评估与异构对战

### 4.1 导出与评估

```bash
# 导出 checkpoint 为 ONNX
python -m suzero_5dchess export \
  --checkpoint runs/very_small/cls_action_mlp/latest.pt --output model.onnx

# 两个 checkpoint 互相对弈
python -m suzero_5dchess evaluate \
  --checkpoint-a runs/very_small/cls_action_mlp/agent_iter_0010.pt \
  --checkpoint-b runs/very_small/cls_action_mlp/agent_iter_0020.pt \
  --games 20

# 独立 C++ 模型对战：每方可使用不同搜索参数、ONNX 或 checkpoint
python -m suzero_5dchess match \
  --model-a model-a.onnx --model-b model-b.pt \
  --sims-a 100 --sims-b 200 --games 20 \
  --output match-results.json
```

`export`、`evaluate`、`match` 都支持 `--variant`，例如
`--variant standard_turn_zero`。ONNX 导出支持动态 batch、棋盘数和动作数；
sidecar 与 C++ 评估器会校验协议版本和几何尺寸。

### 4.2 `match` 语义

- 要求正偶数局，自动换色。
- 独立的 `--max-moves` 默认为 `0`（不限制）；显式设为正数时，统计所有小步
  （包括 `SUBMIT`），达到上限但未终局单独计为 `truncated`，不进入胜、负、和及
  胜率统计。
- 双方可使用不同的模型格式（ONNX / `.pt`）、搜索参数和设备。
- 不替换原有 `evaluate` 或训练期的 `--evaluation` 评估。
- 新增 `--model-match-interval` 独立接入训练期定期对战，并支持参考项目的旧 LSTM
  checkpoint。

更多接口、结果语义及当前限制见
[异构模型对战](docs/heterogeneous-model-match.md)。

### 4.3 模型协议

- 协议名：`smallstep_movekey_v1`；7 个输入：`board_planes`、`l_coords`、
  `t_coords`、`used_board_counts`、`urgency`、`side_to_move`、`actions`。
- 3 个输出：`value`、`submit_logit`、`other_logits`。
- `OnnxEvaluator` 会校验 sidecar JSON、IO 名称/类型/秩、几何尺寸，并拒绝旧协议
  模型。`Evaluation::policy` 与 `Chess5DGameState::actions()` 一一对应。
- 旧 LSTM checkpoint 由 `python/suzero_5dchess/legacy_model.py` 适配，
  共享本项目编码器。

---

## 5. 棋谱与回放

### 5.1 GN5D 棋谱

`run_train.bat` 已默认传入 `--record-games`；直接使用 CLI 时需显式开启：

```bash
python -m suzero_5dchess train --record-games
```

- 每轮自对弈完成后，保留 `logs/games_iter_XXXX.json` 汇总，并逐局保存到
  `runs/<variant>/cls_action_mlp/logs/games/<terminal_reason>/`。自定义
  `--log-dir` 时使用该目录下的 `games/`。
- 文件名为 `iter0001_game000001_<uuid>.gn5d`，按吃王、物质分等终局原因分类，
  每种终局原因最多保留最新 5000 盘，超出时自动删除最旧棋谱；累计局号随 checkpoint
  续训，UUID 防止重跑覆盖旧存档。
- 使用极简 v2 结构：`format="gn5d"`、`version=2`，只保存单个 `initial` FEN、
  `[x1,y1,t1,l1,x2,y2,t2,l2]` / `"submit"` 行动、结果和五个元信息字段。棋盘宽高、
  开局快照、当前棋盘、时间线状态、步数和最终 FEN 均由原生引擎重放推导，不落盘；
  文件使用紧凑 JSON。旧 `5dc.capture-king` v1 和 `.5dck` 仍可读取。
- 支持 Python、C++ 单进程和 C++ 多进程自对弈。存档与训练样本分离；开启
  playout-cap randomization 也不会漏掉快速着法。
- 更新代码后需重新构建 Python 扩展：
  `cmake --build build --config Release --target suzero_5dchess_py`。
  旧扩展请求存档会明确报错，不会静默写空棋谱。
- 历史上 `moves` 为空的 JSON 无法还原完整棋谱；此功能只保证新生成的对局。
  存档在每轮结束后写出，强制中止尚未完成的自对弈批次不会保存未返回的棋局。

完整规范见 [GN5D 记谱法](docs/gn5d.md)。

Python 读取和原生校验（需已安装本项目 Python 包和编译扩展）：

```python
from suzero_5dchess.game_record import load_record, replay_record

record = load_record("path/to/game.gn5d")  # 从动作重放并校验终局
final_state = replay_record(record)          # 恢复后的原生状态，含分支棋盘历史
```

### 5.2 引擎自带 UI 读取棋谱

Windows 运行 `run_ui.bat`，浏览器打开 `http://127.0.0.1:8080/`。页面直接使用
项目内 `5dchess_engine-main/ui/` 的原版 HTML、Canvas、SVG 棋子、主题和导航控件。
点击 **Import → Open File…** 选择 `.gn5d` 或旧 `.5dck`，再点击 **Load**
（也支持粘贴 JSON 或拖入文件）。

- **← / →（Undo / Redo）**：按单个小步或 `SUBMIT` 后退、前进；
- **↑ / ↓（Previous / Next）**：按 `SUBMIT` 边界跳回合，保留末尾未提交的吃王着法；
- 复用原生 UI 的无限画布、缩放拖拽、棋盘翻转、时间线和移动箭头；
- **Export** 下载完整原始棋谱，保留 `.gn5d` / `.5dck` 格式，不覆盖训练存档。

该入口是**只读原生回放模式**：服务端用本项目 C++ 适配层的同一套规则校验并逐步
恢复棋谱，`gn5d-worker.js` 仅把快照适配到引擎 UI 的消息协议，不另写棋盘或棋规。
不需要预编译 WASM，也不需要其他项目目录；启动时显示原生 Standard - Turn Zero
开局。局面编辑、分支删除、Hint 和 `SUBMIT` 已禁用，导入失败保留原局面并在原生
导入框显示错误。请求体上限 16 MiB（原生 UI 文件选择器为 10 MiB），重放帧累计
JSON 上限 128 MiB。

也可从项目根目录直接启动（需已安装本项目 Python 包，或将 `python/` 加入
`PYTHONPATH`）：

```bash
python -m suzero_5dchess.ui_server --port 8080
```

`run_ui.bat` 支持 `PYTHON`、`UI_PORT` 环境变量；有日志目录且已安装 TensorBoard 时，
仍会独立启动 TensorBoard。缺少日志或 TensorBoard 不影响棋谱 UI。服务仅监听本机
`127.0.0.1`；关闭服务窗口或按 Ctrl+C 停止。

普通 PGN 分析仍使用引擎独立构建的 WASM 入口，构建方法见
[引擎构建文档](5dchess_engine-main/docs/usage/building.md#webassembly)。

---

## 6. 架构与设计

```
              ┌──────────────────────────────┐
              │  SelfPlayRunner              │  numGameThreads 个对局线程
              └──────────────┬───────────────┘
                             │ 每局一个
              ┌──────────────▼───────────────┐
              │      PuctSearch              │  图搜索，不含棋规
              │  (single-threaded)           │
              │  StateKey -> SearchNode      │
              └───────┬──────────────┬───────┘
                      │              │
          ┌───────────▼──┐   ┌───────▼──────────┐
          │  IGameState  │   │   IEvaluator     │
          │  (抽象)      │   │   (抽象)         │
          └──────────────┘   └───────▲──────────┘
                                     │
                         ┌───────────┴───────────┐
                         │   BatchedEvaluator    │  请求队列 + server 线程
                         └───────────▲───────────┘
                                     │ 每线程一个
                         ┌───────────┴───────────┐
                         │     IBatchBackend     │  evaluateBatch()
                         └───────────▲───────────┘
                                     │
          ┌──────────────────────────┴───────────┐
          │ Chess5DGameState / OnnxEvaluator     │  5D 规则 + ONNX 推理
          └──────────────────────────────────────┘
```

训练侧是独立的一条链路：

```
自对弈（python 或 cpp_onnx）
        │ 样本：状态编码 + 访问分布 + 行棋方 + value
        ▼
回放缓冲区 ──► TorchTrainer（KataGo 风格调度）──► checkpoint
                                                        │
                                                        ├─► ONNX 导出
                                                        ├──► evaluate / match
                                                        └──► 下一轮自对弈
```

### 6.1 内部搜索核心契约

搜索核心只定义抽象接口，不包含棋规；5D 适配层实现这些契约：

1. **动作空间**：`IGameState::actionCount()` 是**当前状态**的候选动作数，动作 id 为
   `[0, actionCount())` 的稠密整数，且是状态局部的。它必须是状态的确定函数：
   `stateKey()` 相等的两个状态必须返回相同数量、并以相同顺序枚举相同动作。
2. **状态身份**：`IGameState::stateKey()` 相等 <=> 两个状态对搜索完全等价：
   同一行棋方、相同合法动作、相同评估结果、相同终局状态与终局效用。游戏若有
   重复/循环规则，必须把规则所依赖的历史上下文并入键。这是 KataGo `GraphHash` +
   `graphSearchRepBound` 的抽象对应物。
3. **价值视角**：`Evaluation::utility` 与 `terminalUtility()` 都是**当前行棋方**
   （`currentPlayer()`）的效用。搜索在备份时只在**行棋方真正改变**时取负，因此
   支持同一方连续行动的游戏。
4. **策略**：`Evaluation::policy` 长度必须等于 `actionCount()`；不可用动作填负值
   （或 NaN）。合法但概率为零的动作填 `0`，仍会进入搜索（探索项为 0）。正值会被
   重新归一化。
5. **状态克隆**：每次 playout 从根状态 `clone()` 一次，沿路径原地
   `applyAction()`。因此 `applyAction()` 只在合法动作上调用。
6. **效用尺度**：整张图使用统一尺度。若效用落在 `[-1, 1]`，一般将
   `utilityRadius` 设为 `1`。
7. **步数（可选）**：`IGameState::moveCount()` 用于根策略温度与无参
   `chooseAction()` 的早期温度插值；默认返回 `0`（始终使用 early 端）。

为什么保留抽象接口？规则、搜索、评估和训练可以分别测试与替换；5D 适配层不需要
修改搜索核心。游戏侧（`IGameState`、`IEvaluator`、`IStopCriterion`）全部是纯虚
接口，算法侧（`PuctSearch`）与具体游戏解耦。

### 6.2 5D 适配层要点

- `Chess5DGameState`：`IGameState` 子类，封装 5dchess_engine 的规则、吃王终局、
  棋盘数触发的物质分终局、小步/`SUBMIT`。
- `OnnxEvaluator`：`IEvaluator` 子类，封装 `smallstep_movekey_v1` 策略/价值网络。
- `Chess5DAgent`：把游戏、网络与 PUCT 连接起来，含强制 `SUBMIT` 快捷路径、
  子树复用和参考 runner 的温度调度。
- 同一方连续小步时价值视角不变；只有 `SUBMIT` 换方后，搜索备份才翻转符号。

```cpp
#include "suzero/5dchess/agent.h"

using namespace suzero::chess5d;

AgentConfig config;
config.model_path = "models/5dchess/very_small_latest_fp32.onnx";
config.search = DefaultSearchParams();  // 200 playouts, cpuct 2.0
config.seed = 1;

Chess5DAgent agent(config);
while (!agent.state().isTerminal()) {
  const Decision decision = agent.selectAction();  // 温度 1.0 -> 0.1
  agent.applyAction(decision.actionId);
}
```

### 6.3 多对局批量自对弈（C++）

```cpp
#include "suzero/5dchess/onnx_batch_backend.h"
#include "suzero/mcts/selfplay.h"

suzero::mcts::BatchEvaluatorConfig batch;
batch.maxBatchSize = 128;
batch.numServerThreads = 1;
auto evaluator =
    suzero::chess5d::makeBatchedOnnxEvaluator(model_config, batch);

suzero::mcts::SelfPlayConfig config;
config.numGameThreads = 64;
config.maxGames = 64;
config.searchParams.maxPlayouts = 200;

suzero::mcts::SelfPlayRunner runner(
    config,
    [] { return std::make_shared<suzero::chess5d::Chess5DGameState>(); },
    evaluator);
for (const auto& game : runner.run()) {
  // game.samples: state + 稠密访问分布 + 行棋方 + value
}
```

`BatchedEvaluator::evaluate()` 会阻塞到请求被批量服务，因此 `state` 在调用期间
必须保持有效。每个 server 线程独占一个 `IBatchBackend`；batch 只取当前队列里
已有的请求，从不等待凑满。

---

## 7. 构建与测试

### 7.1 仅构建搜索核心

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Linux / macOS：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

核心测试使用 `tests/` 下的玩具游戏（减法取数、单步胜负、同方连续行动），
不依赖真实棋类引擎或神经网络。

### 7.2 构建 5D 适配层

```bat
set SUZERO_ONNXRUNTIME_ROOT=C:\path\to\onnxruntime-win-x64-1.24.3
configure.bat
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

| CMake 选项 | 默认值 | 说明 |
|---|---|---|
| `SUZERO_BUILD_5DCHESS` | `OFF` | 是否构建 5D 适配层与验收测试 |
| `SUZERO_BUILD_PYTHON_BINDINGS` | `OFF` | 构建 pybind11 扩展 `_5dchess`（需要 Python 开发环境） |
| `SUZERO_5DCHESS_ENGINE_ROOT` | 空 | 为空时使用项目内 `5dchess_engine-main`；也可显式指定其他源码树 |
| `SUZERO_ONNXRUNTIME_ROOT` | 空 | ONNX Runtime SDK；为空时尝试 `extern/onnxruntime-win-x64-1.24.3` |
| `SUZERO_ONNXRUNTIME_RUNTIME_DIR` | 自动 | 运行时 DLL 来源目录；默认取 `onnxruntime/capi`（若存在 CUDA provider）或 SDK 的 `lib` |
| `SUZERO_ORT_STAGE_CUDA` | `OFF` | 是否把约 299 MB 的 `onnxruntime_providers_cuda.dll` 也硬链接到 C++ 测试输出目录（测试默认走 CPU，通常不需要） |
| `SUZERO_5DCHESS_MODEL` | `models/5dchess/very_small_latest_fp32.onnx` | 模型路径，sidecar 为 `<model>.json`。模型文件不入库 |

测试目标：

- `suzero_mcts_tests`：搜索核心、图搜索、置换合并、成环、换根、自对弈调度。
- `suzero_5dchess_tests`：状态/动作枚举、克隆、`SUBMIT` 换方、物质分终局、
  编码、ONNX 协议、确定性、搜索与整局循环、批量评估、多对局自对弈与
  `runSelfPlay` 桥接。
- `suzero_5dchess_match_tests`：两种具体适配类型、`MoveKey` / `SUBMIT` 同步、
  不同编码和动作顺序、单条/批量 ONNX 推理及双模型对战。
- Python：`cd python && pytest`，覆盖训练抽象层、PyTorch 训练器、5D 训练闭环、
  存档与对战。

`--probe [plies]` 会先走若干小步，再打印当前局面的候选动作键、策略概率、价值和
urgency，便于与参考编码器逐项对比：

```bash
./build/Release/suzero_5dchess_tests.exe --probe 4
```

---

## 8. 目录结构

```
include/suzero/mcts/
  types.h           动作/玩家/常量
  game_state.h      IGameState（抽象，支持状态相关动作空间）
  evaluator.h       IEvaluator / Evaluation（抽象）
  search_params.h   搜索参数
  search_node.h     节点、边、统计量
  puct_search.h     PuctSearch / SearchResult / IStopCriterion
  random.h          确定性 RNG（PCG32、Gamma）
include/suzero/5dchess/
  game_state.h      Chess5DGameState（5dchess_engine 适配）
  onnx_evaluator.h  OnnxEvaluator（smallstep_movekey_v1）
  agent.h           Chess5DAgent（游戏 + 网络 + 搜索）
src/                搜索核心实现
src/5dchess/        5D 适配层实现
5dchess_engine-main/  固定的规则引擎源码树（BSD-2-Clause）
models/5dchess/     导出的 very_small ONNX 模型与 sidecar（不入库）
tests/              玩具游戏、核心单测与 5D 验收测试
python/suzero_train/          训练抽象层（不依赖 PyTorch）
python/suzero_train/pytorch/  游戏无关的 PyTorch 训练器与数据源
python/suzero_5dchess/        5D 象棋 AlphaZero：绑定、网络、MCTS、
                              自对弈、回放、ONNX 导出、评估与 CLI
python/tests/       Python 训练测试
docs/               算法映射、GN5D、异构对战与设计说明
TECHNICAL.md        当前实现状态、关键不变量与变更历史
THIRD_PARTY_NOTICES.md
run_train.bat       Windows 训练入口
run_ui.bat          Windows 棋谱回放入口
configure.bat       可重定位的 CMake 配置脚本
```

---

## 9. 来源、借鉴与许可

### 9.1 KataGo（MIT）

本项目**搜索核心派生自 KataGo 的搜索模块**，不是从零实现的独立算法：

- 上游参考路径：`cpp/search/search.h`、`search.cpp`、`searchnode.h`、
  `searchparams.h`、`searchhelpers.cpp`、`searchexplorehelpers.cpp`、
  `searchupdatehelpers.cpp`、`searchresults.cpp`、`searchnodetable.*`、
  `cpp/game/graphhash.*`。
- 借鉴内容：PUCT / FPU 公式、根节点 Dirichlet 噪声、访问权重聚合、
  3 自由度 Student-t 坏子降权、非 LCB 回顾式着法权重、图搜索节点表、边访问追赶、
  成环处理、根节点提升与子树复制、最终着法采样与温度语义。
- Python 训练器的 per-sample LR、warmup、参数组、梯度裁剪、指标累计与 checkpoint
  结构，重新实现了 KataGo `python/train.py` 的方法；未复制其 Python 源文件。
- KataGo 采用 MIT 许可证。上游版权声明与完整许可证原文保留在
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，逐项公式和映射见
  [docs/katago-mapping.md](docs/katago-mapping.md)。

### 9.2 其他第三方组件

- **5dchess_engine**（BSD-2-Clause，Copyright (c) 2024-2026, ftxi）：
  `src/5dchess/` 通过公开 API 使用其棋规与走法生成；`run_ui.bat` 直接使用其原生
  UI。`5dchess_engine-main/LICENSE` 与 UI 资源各自的许可证随源码保留。
- **ONNX Runtime**（MIT，Microsoft）：构建时链接用户提供的预编译 SDK，
  不复制其源码。
- **very_small ONNX 模型**：由 [5dc_ml](https://github.com/ftxi/5dc_ml)
  （BSD-2-Clause）导出的检查点生成，仅用于集成与验收测试。
- **旧 LSTM checkpoint 兼容**：`legacy_model.py` 按参考项目的网络拓扑重新实现
  推理适配，固定 `very_small_old_version_v1.pt` 由使用者从本地提供。

### 9.3 本项目

新增代码以 **MIT** 发布，见 [LICENSE](LICENSE)。`tests/` 下的玩具游戏仅用于验证
抽象接口和搜索算法；5D 适配层是独立构建目标，不影响测试核心时对引擎的零依赖。
