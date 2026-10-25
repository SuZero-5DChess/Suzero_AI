# SuZero-02 · 游戏无关 PUCT 图搜索核心

从 **KataGo** 的搜索模块中提取、重写出的游戏无关 PUCT **图搜索**（graph search，
即带置换合并的 DAG 搜索）库。核心只包含 **抽象接口 + 与游戏无关的搜索算法**；
另有一个可选的 **5D 象棋（Very Small - Open）适配层**，用外部棋规引擎和
ONNX 神经网络把 `IGameState`、`IEvaluator` 与 `PuctSearch` 连接起来。

- 语言：核心 C++17；5D 适配层因上游引擎要求使用 C++23
- 依赖：核心仅标准库；适配层需要 5dchess_engine 与 ONNX Runtime
- 许可证：MIT（KataGo 派生部分保留其 MIT 版权声明，见
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)）
- 构建：CMake ≥ 3.16

---

## 1. 当前范围

### 已实现

- KataGo 风格的 PUCT 选择（含 child weight / edge visits 语义）
- First Play Urgency（FPU）：`fpuReductionMax` 等参数、基于已访问策略质量的父值混合
- Utility 标准差对探索系数的缩放（`cpuctUtilityStdev*`）
- 根节点策略温度（含早期/后期插值）与 KataGo 式策略形状 Dirichlet 噪声
- 带权值聚合的备份：3 自由度 Student-t CDF、KataGo 的 `valueWeightExponent`
  坏子节点降权，以及剪枝前总权重归一化
- 终局价值、非交替行棋（同一方连续行动）的价值视角处理
- **图搜索**：按 `IGameState::stateKey()` 合并置换节点，边只引用子节点
- 根节点不入节点表，保证换根后不会出现经过根的环
- 边访问追赶（`graphSearchCatchUpLeakProb`）与路径成环检测；
  无环游戏可用 `graphSearchCycleCheckEnabled = false` 省去路径集合的哈希与查找
- 访问数 / playout 数 / 时间 / 深度限制，以及外部停止条件接口
- 根节点子树复用（`advanceRoot`，复制子图并重建节点表）
- 最终着法权重与采样：KataGo 非 LCB 回顾式降权、温度（含早期插值）、
  subtract/prune（`chosenMove*`）
- **状态相关动作空间**：`actionCount()` 允许随状态变化（5D 象棋的候选
  小步由当前位置生成），动作 id 在状态内稠密，先验按需为每个节点构建
- **异构模型对战**：`IMatchGame` / `MatchRunner` 使用双方独立游戏适配器、
  evaluator 和搜索参数，通过共享逻辑动作同步；支持完整换色配对、分颜色统计，
  截断单独计数且不进入胜、负、和及胜率统计。5dc 提供独立 `match` CLI，详见
  [对战接口与使用说明](docs/heterogeneous-model-match.md)
- **5D 象棋适配层**（可选构建）：
  - `Chess5DGameState`：`IGameState` 子类，封装 5dchess_engine 的
    Very Small - Open 规则、吃王终局、棋盘数触发的物质分终局、小步/SUBMIT
  - `OnnxEvaluator`：`IEvaluator` 子类，封装 `smallstep_movekey_v1`
    ONNX 策略/价值网络
  - `Chess5DAgent`：把游戏、网络与 PUCT 连接起来，含强制 SUBMIT 快捷路径、
    子树复用和参考 runner 的温度调度
- **训练子系统（可选，`python/`）**：
  - 游戏无关抽象层：`Trainer` / `TrainingDataSource` / `TrainingLoop` /
    `MetricsSink`，不依赖 PyTorch 或具体游戏
  - PyTorch 后端：`TorchTrainer` 接收任意 `torch.nn.Module` 与 objective，
    沿用 KataGo 的 per-sample LR、warmup、参数组、梯度裁剪、AMP、
    checkpoint 与指标累计；`InMemoryDataSource` / `IterableDataSource`
    提供与游戏无关的数据接入
  - **5D 象棋 AlphaZero 流水线**（`suzero_5dchess`）：
    - C++ 适配层通过 pybind11 暴露为 `_5dchess`，Python 侧负责
      自对弈、MCTS、回放缓冲区与训练
    - `cls_action_mlp` 架构的纯 PyTorch 重实现：棋盘 tokenizer、
      (l, t) 正弦位置编码、transformer、价值/SUBMIT/动作头
    - 自对弈（Dirichlet 根噪声、温度调度、playout-cap randomization、
      强制 SUBMIT 不调用网络）、均匀回放、掩码策略交叉熵 + 价值 MSE
    - 两种自对弈后端：`python`（进程内 PyTorch MCTS）与 `cpp_onnx`
      （导出 ONNX 后由 C++ 多线程 PUCT + ONNX Runtime 执行，可选 CUDA）
    - 协作式 Ctrl+C：第一次按下完成当前轮并保存退出，第二次立即中止
    - 原子 checkpoint 与断点续训、ONNX 动态 batch/棋盘数/动作数导出、
      模型对模型评估
    - 默认 `d_model=128` 时参数总量 **192,058**，构建时强制不超过
      200,000；默认变体为 `very_small`

### 明确未实现

- 单局内多线程搜索（virtual loss、原子统计）；并行度来自多对局线程
- 评估缓存（KataGo 的 eval cache）
- Go 专用逻辑：目数效用、领地、ownership、贴目、劫、对称性、停一手
- 人类 SL 模型、mirror 对抗、visit cap、时间控制策略
- 模型热切换与跨进程自对弈训练循环（KataGo `SelfplayManager` 部分）

这些是后续可独立扩展的方向。详见
[docs/katago-mapping.md](docs/katago-mapping.md)。

---

## 2. 设计

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
          │ 具体游戏 / 引擎适配层（5dchess 可选）│
          └──────────────────────────────────────┘
```

### 核心契约

1. **动作空间**：`IGameState::actionCount()` 是**当前状态**的候选动作数，
   动作 id 为 `[0, actionCount())` 的稠密整数，且是状态局部的。固定动作
   空间的游戏可在所有状态返回同一值；5D 象棋这类按局面生成合法动作的游戏
   可以返回不同的数量。它必须是状态的确定函数：`stateKey()` 相等的两个
   状态必须返回相同数量、并以相同顺序枚举相同动作。
2. **状态身份**：`IGameState::stateKey()` 相等 <=> 两个状态对搜索完全等价：
   同一行棋方、相同合法动作、相同评估结果、相同终局状态与终局效用。游戏若有
   重复/循环规则（超级劫、循环判和、步数计数等），必须把规则所依赖的历史上下文
   并入键。键的具体表示由游戏决定，任意字节串均可。这是 KataGo `GraphHash` +
   `graphSearchRepBound` 的抽象对应物。
3. **价值视角**：`Evaluation::utility` 与 `terminalUtility()` 都是
   **当前行棋方**（`currentPlayer()`）的效用。搜索在备份时只在
   **行棋方真正改变**时取负，因此支持同一方连续行动的游戏。
4. **策略**：`Evaluation::policy` 长度必须等于 `actionCount()`；不可用动作填
   负值（或 NaN）。合法但概率为零的动作填 `0`，仍会进入搜索（探索项为 0）。
   正值会被重新归一化。
5. **状态克隆**：每次 playout 从根状态 `clone()` 一次，沿路径原地
   `applyAction()`。因此 `applyAction()` 只在合法动作上调用。
6. **效用尺度**：整张图使用统一尺度。若效用落在 `[-1, 1]`，一般将
   `utilityRadius` 设为 `1`。
7. **步数（可选）**：`IGameState::moveCount()` 用于根策略温度与无参
   `chooseAction()` 的早期温度插值；默认返回 `0`（始终使用 early 端）。

### 为什么不把“抽象类”做成空接口？

游戏侧（`IGameState`、`IEvaluator`、`IStopCriterion`）全部是纯虚接口；
算法侧（`PuctSearch`）与具体游戏完全解耦。这样抽象边界明确、可直接单测，
又不会得到一个无法运行的纯声明集合。接入引擎时只需实现这两个接口，
不需要修改搜索核心。

---

## 3. 快速使用

```cpp
#include <memory>

#include "suzero/mcts/puct_search.h"

using namespace suzero::mcts;

class MyState final : public IGameState {
 public:
  std::unique_ptr<IGameState> clone() const override;
  int actionCount() const override { return kNumActions; }
  bool isLegal(ActionId action) const override;
  void applyAction(ActionId action) override;
  PlayerId currentPlayer() const override;
  bool isTerminal() const override;
  double terminalUtility() const override;  // 当前行棋方视角

  // 图搜索的身份键：必须包含行棋方、以及影响合法性/终局的全部历史上下文。
  StateKey stateKey() const override;
};

class MyEvaluator final : public IEvaluator {
 public:
  Evaluation evaluate(const IGameState& state) override;
};

SearchParams params;
params.maxPlayouts = 1600;
params.rootNoiseEnabled = true;                 // 自对弈时常用
params.rootDirichletNoiseWeight = 0.25;
params.chosenMoveTemperature = 0.0;             // 0 = 选访问最多的

auto search = std::make_unique<PuctSearch>(
    std::make_shared<MyState>(initial_state),
    std::make_shared<MyEvaluator>(),
    params,
    /*seed=*/1234);

const SearchResult result = search->run();       // 搜索
const ActionId move = search->chooseAction();    // 采样/取最优
const std::vector<double> target =
    result.denseVisitDistribution(MyState::kNumActions);  // 训练目标
```

对局推进时可复用子树：

```cpp
// 假设对手实际走了 move
if (!search->advanceRoot(move)) {
  search->resetRoot(std::make_shared<MyState>(next_state));
}
```

图搜索按 `stateKey()` 合并不同落子顺序到达的同一状态；同一 playout 的路径上
再次遇到已访问节点时只计入边访问并停止继续下探，因此带环游戏不会无限递归。
`params.maxDepth` 仍可作为额外的硬上限。

### 5D 象棋（可选适配层）

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

- 动作是**小步**（8 元 `MoveKey`）或 `SUBMIT`。同一方连续小步时价值视角
  不变；只有 `SUBMIT` 换方后，搜索备份才翻转符号。
- 候选动作由当前局面生成，`SUBMIT` 固定排在最后。只有 `SUBMIT` 一个候选时
  `Decision::forced == true`，既不调用网络也不运行搜索。
- 终局：吃王（`capture_king`）、任一动作后棋盘数达到或超过上限的物质分
  （`material`）、无合法动作（`no_legal_action`）。若同一小步既吃王又触及棋盘上限，
  吃王优先。
- 模型协议为 `smallstep_movekey_v1`（7 输入 / 3 输出）；评估器会校验
  sidecar JSON、IO 名称/类型/秩、几何尺寸，并拒绝旧协议模型。
- `OnnxEvaluator` 可单独使用：`Evaluation::policy` 与
  `Chess5DGameState::actions()` 一一对应。

### 训练器（Python，可选）

```python
from suzero_train import LoopConfig, TrainingLoop
from suzero_train.pytorch import (
    InMemoryDataSource,
    TorchTrainer,
    TorchTrainerConfig,
)

trainer = TorchTrainer(
    model,
    objective,                       # (model, data) -> (loss, metrics)
    config=TorchTrainerConfig(device="cuda"),
)
source = InMemoryDataSource(samples, batch_size=256, collate_fn=collate)
report = TrainingLoop(trainer, source, LoopConfig(epochs=10)).run()
```

抽象层只定义 `Trainer` / `TrainingDataSource` / `TrainingLoop` / `MetricsSink`
契约，且不导入 PyTorch；`suzero_train.pytorch` 用任意模型、任意 objective 和
任意数据集实现这些契约，训练方法（per-sample LR、warmup、参数组、
梯度裁剪、AMP、指标累计、checkpoint）提取自 KataGo，但不依赖围棋或
KataGo 的模型/数据格式。训练子系统与 CMake 目标相互独立，不改变 C++ 核心
的依赖。详见 [python/README.md](python/README.md)。

### 5D 象棋训练（Python）

先把 C++ 适配层编译成 Python 扩展（构建一次即可）：

```bat
set SUZERO_ONNXRUNTIME_ROOT=C:\path\to\onnxruntime-win-x64-1.24.3
configure.bat
cmake --build build --config Release
```

`configure.bat` 从脚本自身位置推导全部项目内路径，整个目录移动后直接重跑即可。
它不把 `SUZERO_5DCHESS_ENGINE_ROOT` 写进缓存（全新配置会自动选中项目内的
`5dchess_engine-main`）；检测到 `build/` 属于其他源码路径时会自动追加
`cmake --fresh` 重新生成。ONNX Runtime SDK 也可解压到
`extern/onnxruntime-win-x64-1.24.3/`（该目录不入库），脚本会自动发现。

扩展会输出到 `python/suzero_5dchess/`，但不再随包复制 ONNX Runtime DLL。
导入时由 `suzero_5dchess/_ort_runtime.py` 把已安装 `onnxruntime` 包的
`capi` 目录加入 Windows DLL 搜索路径；若该环境装的是 `onnxruntime-gpu`，
CUDA provider DLL 本来就在该目录中，无需额外部署。需要指向其他运行时时，
设置环境变量 `SUZERO_ORT_RUNTIME_DIR`（优先于已安装的包）。
构建需要当前 Python 环境已安装 `pybind11`（`python -m pip install pybind11`）。
`cpp_onnx` 后端的 `cuda` 模式还需要 CUDA 12.x 与 cuDNN 9（`torch/lib`
中的 cuDNN 会被 `import torch` 自动加入 DLL 搜索路径）。
然后：

```bash
cd python
python -m pip install -e ".[chess5d]"

# 冒烟测试：2 局、2 simulations、固定 board limit 4、1 轮训练，CPU 即可
python -m suzero_5dchess train --debug \
  --games 2 --sims 2 --min-board-limit 4 --max-board-limit 4

# 正式训练（very_small 为默认变体）
python -m suzero_5dchess train --iterations 100 --games 64 --sims 800

# C++ 自对弈后端：CUDA 推理 + 3 个对局线程 + 2 个评估服务线程
python -m suzero_5dchess train --selfplay-backend cpp_onnx \
  --cpp-onnx-provider cuda --workers 3 --num-server-threads 2

# 从 latest 续训
python -m suzero_5dchess train --resume latest

# 连续训练：每 20 轮为一批，自动从 latest 续训，直到 Ctrl+C
python -m suzero_5dchess train --iterations 20 --continuous

# 导出 checkpoint 为 ONNX，或让两个 checkpoint 对弈
python -m suzero_5dchess export \
  --checkpoint runs/very_small/cls_action_mlp/latest.pt --output model.onnx
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

`match` 要求正偶数局，自动换色；独立的 `--max-moves` 默认为 `0`（不限制）。
显式设为正数时，它统计所有小步（包括 SUBMIT）；达到上限但未终局单独计为
`truncated`，不进入胜、负、和及胜率统计。它不替换原有 `evaluate` 或
`--evaluation` 评估。
新增 `--model-match-interval` 独立接入训练期定期对战，并支持参考项目的旧 LSTM checkpoint。
更多接口、结果语义及当前限制见 [异构模型对战](docs/heterogeneous-model-match.md)。

`--num-server-threads` 控制 `cpp_onnx` 后端 C++ ONNX 评估器的服务线程数
（默认 1）。它与 `--workers`（并发对局数）相互独立：每个服务线程各自持有
一个独立的 ONNX Session，因此增加线程会带来额外的内存占用，CUDA 模式下
还会增加显存占用；实际增量取决于模型与会话配置，不保证与线程数严格成比例。
增加服务线程并不保证提速，是否需要取决于实际负载与设备。默认值 1 对旧扩展
保持兼容；只有显式请求非默认线程数、且旧扩展不接受 `num_server_threads`
时才会报错并提示重新构建（见上文构建步骤）。

`--selfplay-processes` 选择自对弈工作进程数，默认 `1`，保持原有的单进程路径。
大于 `1` 时只能配合 `--selfplay-backend cpp_onnx`，且 `--workers`、
`--num-server-threads`、`--games` 都必须不小于进程数。`--workers` 与
`--num-server-threads` 始终是总预算，由各进程分摊，不要求整除，也不会自动
翻倍（例如 50 workers / 2 processes 可分为 25+25；3 server threads / 2
processes 可分为 2+1）。例如：

```bash
python -m suzero_5dchess train --selfplay-backend cpp_onnx \
  --cpp-onnx-provider cuda --workers 50 --num-server-threads 2 \
  --games 300 --selfplay-processes 2
```

这是训练入口内的可选多进程自对弈；`python/experiments/multiprocess_selfplay/`
下的 benchmark 是独立性能基准，不属于训练路径。默认 `1` 即回退到原路径。

`train --debug` 使用 `very_small`、1 轮、1 epoch、2 step、batch 8、
2 局、2 simulations，并把每局棋盘上限固定为 4；在 `runs/debug/` 下生成 checkpoint
与可被 C++ 评估器直接加载的 `models/very_small_latest_fp32.onnx`。自对弈不设动作数
截断：每局由棋盘上限或其他自然终局结束。checkpoint 中保存网络几何，回放缓冲区
独立存于训练目录的 `replay_buffer.pt`，续训会校验变体与 `d_model` 等配置。

`run_train.bat` 是 Windows 下的训练入口，默认使用 `cpp_onnx` + `cuda` +
`--num-server-threads 2`（可在脚本顶部的 `NUM_SERVER_THREADS` 修改，或追加
`--num-server-threads N` 覆盖；`SELFPLAY_WORKERS` 等其余旋钮同理）。服务线程
数与并发对局数 `--workers` 相互独立，每个服务线程各持有一个 ONNX Session，
调高会增加内存（CUDA 下还有显存）占用，并不保证提速，需按实际设备实测。
`cpp_onnx` 每轮把当前网络导出到
`runs/<variant>/<arch>/selfplay/model_iter_XXXX.onnx`，只保留最近 2 份；
该后端和 Python 后端均支持 `--record-games`：保存每轮 JSON 及每局可回放的
`.gn5d` 存档，含快速搜索、强制 SUBMIT 和终局前未提交的最后一手。GN5D 会提交给
贯穿训练（含连续批次）的常驻后台进程，原生校验与落盘和后续训练/自对弈重叠；正常
退出前会排空队列，写入错误会传回主进程。
训练中按一次 Ctrl+C 会等当前轮（自对弈 + 训练 + 评估）结束后保存并退出，
再按一次立即中止。按键后控制台会立刻打印“已收到中断信号”，`cpp_onnx`
后端整批自对弈期间也一样，不会出现按键无反应的情况。

脚本顶部的 `CONTINUOUS=1` 开启连续训练：`--iterations` 变成每批轮数，
每批结束（保存并发布 `latest.pt`）后训练器自动从 `latest.pt` 续训下一批，
不再退出，直到按 Ctrl+C。也可以单次运行追加 `--continuous` 参数。连续模式
下第一次 Ctrl+C 仍会完成当前轮、保存并停止；第二次立即中止。

### Standard - Turn Zero（TurnZero / T0）训练

新增变体 `standard_turn_zero`，对应引擎的 `[Board "Standard - Turn Zero"]`：
8×8 棋盘，开局保留 T0 黑方历史棋盘和 T1 白方棋盘，并非普通 `standard` 开局。
游戏引擎、搜索算法和训练目标不变。

Windows 下在项目根目录直接运行：

```bat
run_train_turnzero.bat
REM Equivalent shared entrypoint:
run_train.bat standard_turn_zero
```

新脚本复用 `run_train.bat` 顶部的 CUDA、自对弈并发、连续训练、存档和定期对战
配置；追加参数仍可覆盖默认值。CLI 选择该变体时默认将参数预算设为 **300,000**，
保留 `d_model=128`，实际参数量为 **270,202**；显式 `--max-params` 仍优先。
`very_small` 的 200,000 预算不变。直接通过 Python 构造网络时使用
`NetworkConfig(max_params=300_000)`。

该变体的自对弈默认每步 **1000** 次完整模拟（`--debug` 小预算预设和显式
`--sims` 仍优先）；快速搜索仍由 `fast_simulations`（默认 50）控制。

- 正式 checkpoint：`runs/standard_turn_zero/cls_action_mlp/latest.pt`，脚本再次运行自动续训。
- ONNX：同目录的 `models/standard_turn_zero_latest_fp32.onnx`（及 `.json` sidecar）。
- 调试目录：`runs/debug/standard_turn_zero/`，不会误续训已有的 Very Small 调试模型。
- 不自动使用 Very Small 的固定旧模型；可对战同变体的历史快照。变体不同的
  checkpoint 即使同为 8×8，也不能直接 `--resume`。

**小预算测试**用下面的 CLI 命令（已安装 Python 包；或先在项目根目录设置
`PYTHONPATH=python`）。只跑 **2 局、2 次模拟、固定棋盘上限 4、1 轮训练**，完成后退出，
不会继承批处理脚本的连续训练开关。TurnZero 初始有 2 张棋盘，因此上限 4 可快速触发
正常的 `material` 终局：

```bash
python -m suzero_5dchess train --variant standard_turn_zero --debug \
  --device cpu --games 2 --sims 2 \
  --min-board-limit 4 --max-board-limit 4 \
  --batch-size 1 --min-replay-size 1 --steps-per-epoch 1 --record-games
```

自对弈没有步数截断配置，每局由棋盘上限或吃王、无合法动作等自然终局结束。
正式 CLI 入口是 `train --variant standard_turn_zero`；续训再加 `--resume latest`。
`export`、`evaluate`、`match` 也都支持 `--variant standard_turn_zero`。

### 训练中定期模型对战

`run_train.bat` 按顶部 `EVALUATION_INTERVAL` 指定的轮次间隔，分别对战恰好 `MATCH_LOOKBACK`
轮前的模型与指定的 `very_small_old_version_v1.pt` 固定旧模型（后者只用于 `very_small`）。
每个对手 20 局换色对战，默认使用 **20 个 C++ 游戏线程并发**，双方各自独立批量推理。
并发数由脚本顶部 `MATCH_WORKERS` 或 `--model-match-workers N` 设置，独立于自对弈线程数，
实际不超过对局数；设为 1 可恢复串行。simulations 沿用脚本的 `MATCH_SIMS`；独立的
对战棋盘上限由 `MATCH_MAX_BOARDS` 控制，默认值为 `25`。达到或超过该上限时立即按物质分结束。
默认使用 CUDA。
间隔和历史跨度独立设置，例如：

```bat
run_train.bat very_small cls_action_mlp --model-match-interval 10 --model-match-lookback 30 --model-match-games 8
REM Disable periodic matches:
run_train.bat very_small cls_action_mlp --model-match-interval 0
```

脚本顶部的 `MATCH_REFERENCE` 可更换固定对手；`MATCH_PROVIDER=cpu` 可改为 CPU。
该路径独立保存每轮轻量快照，不依赖训练 checkpoint 保存间隔；历史缺失时只跳过历史对战。
完整棋局 JSON 写入训练日志目录的 `matches/`，每局同时生成可重放的 `.gn5d`，保存到
`matches/games/<terminal_reason>/`；设置 `--gn5d-data-dir` 时，这两类定期对战文件位于该
数据目录下的 `<variant>/cls_action_mlp/matches/`。汇总中的 `gn5d_records` 列出对应棋谱路径。
对战汇总进入 `training_log.jsonl` 的 `model_matches` 和 TensorBoard 的
`ModelMatch/previous/*`、`ModelMatch/fixed/*`。
控制台会打印总胜率，以及当前模型和对手分别持白、持黑的胜率；这些颜色胜率只统计对应颜色的正常终局。
截断与和棋分开统计；没有自动晋级或权重回滚。直接使用 Python CLI 时需显式开启。
参数、历史选择和旧模型兼容范围见 [定期对战说明](docs/heterogeneous-model-match.md#训练期定期对战)。

### `.gn5d` 棋谱存档

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
  文件使用紧凑 JSON。旧 `5dc.capture-king` v1 和 `.5dck` 仍可读取。通过项目内
  `5dchess_engine-main/ui/` 的原生 UI 回放 GN5D 和旧 `.5dck`，不再使用独立编写的
  棋谱页面。完整规范见 [GN5D 记谱法](docs/gn5d.md)。
- 支持 Python、C++ 单进程和 C++ 多进程自对弈。存档与训练样本分离；开启
  playout-cap randomization 也不会漏掉快速着法。
- 更新代码后需重新构建 Python 扩展：
  `cmake --build build --config Release --target suzero_5dchess_py`。
  旧扩展请求存档会明确报错，不会静默写空棋谱。
- 历史上 `moves` 为空的 JSON 无法还原完整棋谱；此次功能只保证新生成的对局。
  存档在每轮结束后写出，强制中止尚未完成的自对弈批次不会保存未返回的棋局。

Python 读取和原生校验（需已安装本项目 Python 包和编译扩展）：

```python
from suzero_5dchess.game_record import load_record, replay_record

record = load_record("path/to/game.gn5d")  # 从动作重放并校验终局
final_state = replay_record(record)          # 恢复后的原生状态，含分支棋盘历史
```

### 引擎自带 UI 读取棋谱

Windows 运行 `run_ui.bat`，浏览器打开 `http://127.0.0.1:8080/`。页面直接使用项目内
`5dchess_engine-main/ui/` 的原版 HTML、Canvas、SVG 棋子、主题和导航控件；旧的根目录
`ui/` 独立查看器已删除。点击 **Import → Open File…** 选择 `.gn5d` 或旧 `.5dck`，
再点击 **Load**（也支持粘贴 JSON 或拖入文件）。

- **← / →（Undo / Redo）**：按单个小步或 SUBMIT 后退、前进；
- **↑ / ↓（Previous / Next）**：按 SUBMIT 边界跳回合，保留末尾未提交的吃王着法；
- 复用原生 UI 的无限画布、缩放拖拽、棋盘翻转、时间线和移动箭头；
- **Export** 下载完整原始棋谱，保留 `.gn5d` / `.5dck` 格式，不覆盖训练存档。

该入口是**只读原生回放模式**：服务端用本项目 C++ 适配层的同一套规则校验并逐步恢复棋谱，
`gn5d-worker.js` 仅把快照适配到引擎 UI 的消息协议，不另写棋盘或棋规。
不需要预编译 WASM，也不需要其他项目目录；启动时显示原生 Standard - Turn Zero 开局。
局面编辑、分支删除、Hint 和 SUBMIT 已禁用，导入失败保留原局面并在原生导入框显示错误。
请求体上限 16 MiB（原生 UI 文件选择器为 10 MiB），重放帧累计 JSON 上限 128 MiB。
普通 PGN 分析仍使用引擎独立构建的 WASM 入口，构建方法见
[引擎构建文档](5dchess_engine-main/docs/usage/building.md#webassembly)。

也可从项目根目录直接启动（需已安装本项目 Python 包，或将 `python/` 加入 `PYTHONPATH`）：

```bash
python -m suzero_5dchess.ui_server --port 8080
```

`run_ui.bat` 支持 `PYTHON`、`UI_PORT` 环境变量；有日志目录且已安装 TensorBoard 时，
仍会独立启动 TensorBoard。缺少日志或 TensorBoard 不影响棋谱 UI。服务仅监听本机
`127.0.0.1`；关闭服务窗口或按 Ctrl+C 停止。

---

## 4. 构建与测试

Windows（Visual Studio 2022）：

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
不依赖任何真实棋类引擎或神经网络。`suzero_match_tests` 另行覆盖异构适配、
动作映射、终局视角、零候选动作终局和配对统计。

### 构建 5D 象棋适配层

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
| `SUZERO_ONNXRUNTIME_RUNTIME_DIR` | 自动 | 运行时 DLL 来源目录；默认取 `onnxruntime/capi`（若存在 CUDA provider）或 SDK 的 `lib`。Python 扩展导入时从此目录加载，C++ 测试输出目录以硬链接引用 |
| `SUZERO_ORT_STAGE_CUDA` | `OFF` | 是否把约 299 MB 的 `onnxruntime_providers_cuda.dll` 也硬链接到 C++ 测试输出目录（测试默认走 CPU，通常不需要） |
| `SUZERO_5DCHESS_MODEL` | `models/5dchess/very_small_latest_fp32.onnx` | 模型路径，sidecar 为 `<model>.json`。模型文件不入库，需先用 `suzero-5dc export` 生成或自行放置 |

适配层目标为 `suzero_5dchess`，测试目标为
`suzero_5dchess_tests`（覆盖状态/动作枚举、克隆、SUBMIT 换方、
物质分终局、编码、ONNX 协议、确定性、搜索与整局循环，以及批量评估、
多对局自对弈与 Python 训练器使用的 `runSelfPlay` 桥接）。
`suzero_5dchess_match_tests` 覆盖两种具体适配类型、MoveKey / SUBMIT 同步、
不同编码和动作顺序、单条 / 批量 ONNX 推理及双模型对战。

### 多对局批量自对弈

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

核心库只要求游戏实现 `IGameState` 与一个模型侧 `IBatchBackend`；5D 适配层
的 `OnnxBatchBackend` 是可选的参考实现。`BatchedEvaluator::evaluate()` 会
阻塞到请求被批量服务，因此 `state` 在调用期间必须保持有效。

`--probe [plies]` 会先走若干小步，再打印当前局面的候选动作键、策略概率、
价值和 urgency，便于与 5dc_ml 参考编码器逐项对比：

```bash
./build/Release/suzero_5dchess_tests.exe --probe 4
```

---

## 5. 目录结构

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
src/5dchess/   5D 适配层实现
models/5dchess/ 导出的 very_small ONNX 模型与 sidecar（不入库）
tests/              玩具游戏、核心单测与 5D 验收测试
python/suzero_train/  训练抽象层（不依赖 PyTorch）
python/suzero_train/pytorch/  游戏无关的 PyTorch 训练器与数据源
python/suzero_5dchess/  5D 象棋 AlphaZero：绑定、网络、MCTS、
                    自对弈、回放、ONNX 导出、评估与 CLI
python/tests/       Python 训练测试（微型模型与 very_small 调试对局）
docs/               算法映射与设计说明
THIRD_PARTY_NOTICES.md
```

---

## 6. 许可证与来源

本项目的搜索算法派生自
[KataGo](https://github.com/lightvector/KataGo)（作者 David J Wu 等，MIT）。
派生文件均带有 SPDX 头与来源说明；Python 训练器的调度与裁剪公式源自 KataGo 的
`python/train.py`，但代码是重新实现的（未复制上游文件），范围见
[python/README.md](python/README.md)。完整 MIT 原文与第三方声明见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

新增代码以 MIT 发布，见 [LICENSE](LICENSE)。
