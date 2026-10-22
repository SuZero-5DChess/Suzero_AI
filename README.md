# SuZero-02 · 游戏无关 PUCT/MCTS 核心

从 **KataGo** 的搜索模块中提取、重写出的游戏无关 MCTS 库。当前阶段只提供
**抽象接口 + 与游戏无关的搜索算法**，**没有接入任何棋类引擎**，也不包含任何棋规、
神经网络推理后端或训练代码。

- 语言：C++17
- 依赖：仅标准库
- 许可证：MIT（KataGo 派生部分保留其 MIT 版权声明，见
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)）
- 构建：CMake ≥ 3.16

---

## 1. 当前范围

### 已实现

- KataGo 风格的 PUCT 选择（含 child weight / edge visits 语义）
- First Play Urgency（FPU）：`fpuReductionMax`、`rootFpuReductionMax`、
  `fpuLossProp`、基于已访问策略质量的父值混合
- Utility 标准差对探索系数的缩放（`cpuctUtilityStdev*`）
- 根节点策略温度与 KataGo 式策略形状 Dirichlet 噪声
- 带权值聚合的备份，包括 KataGo 的 `valueWeightExponent` 坏子节点降权
- 终局价值、非交替行棋（同一方连续行动）的价值视角处理
- 访问数 / playout 数 / 时间 / 深度限制，以及外部停止条件接口
- 根节点子树复用（`advanceRoot`）
- 最终着法采样：温度、subtract/prune（`chosenMove*`）

### 明确未实现

- 多线程、原子统计、virtual loss
- Graph search / 置换表 / 评估缓存
- Go 专用逻辑：目数效用、领地、ownership、贴目、劫、对称性、停一手
- 人类 SL 模型、mirror 对抗、visit cap、时间控制策略
- 批量神经网络推理

这些是后续可独立扩展的方向，接口中已保留必要位置（例如
`IGameState::stateKey()` 预留给将来的图搜索）。详见
[docs/katago-mapping.md](docs/katago-mapping.md)。

---

## 2. 设计

```
                 ┌──────────────────────────┐
                 │      PuctSearch          │  纯算法，不含棋规
                 │  (single-threaded)       │
                 └───────┬──────────┬───────┘
                         │          │
             ┌───────────▼──┐   ┌───▼────────────┐
             │  IGameState  │   │  IEvaluator    │
             │  (抽象)      │   │  (抽象)        │
             └──────────────┘   └────────────────┘
                         │
             ┌───────────▼──────────┐
             │ 具体游戏 / 引擎适配层 │  ← 当前未实现
             └──────────────────────┘
```

### 核心契约

1. **动作空间**：`IGameState::actionCount()` 对同一游戏恒定；动作 id 为
   `[0, actionCount())` 的稠密整数。
2. **价值视角**：`Evaluation::utility` 与 `terminalUtility()` 都是
   **当前行棋方**（`currentPlayer()`）的效用。搜索在备份时只在
   **行棋方真正改变**时取负，因此支持同一方连续行动的游戏。
3. **策略**：`Evaluation::policy` 长度必须等于 `actionCount()`；不可用动作填
   负值或 `0`，引擎还会再用 `isLegal()` 过滤。正值会被重新归一化。
4. **状态克隆**：每次 playout 从根状态 `clone()` 一次，沿路径原地
   `applyAction()`。因此 `applyAction()` 只在合法动作上调用。
5. **效用尺度**：整棵树使用统一尺度。若效用落在 `[-1, 1]`，一般将
   `utilityRadius` 设为 `1`。

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

如果游戏包含循环，请设置 `params.maxDepth`，否则单次 playout 可能沿环
无限下降。

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

测试使用 `tests/` 下的玩具游戏（减法取数、单步胜负、同方连续行动），
**不是**任何真实棋类引擎，只用于验证抽象接口与搜索语义。

---

## 5. 目录结构

```
include/suzero/mcts/
  types.h           动作/玩家/常量
  game_state.h      IGameState（抽象）
  evaluator.h       IEvaluator / Evaluation（抽象）
  search_params.h   搜索参数
  search_node.h     节点、边、统计量
  puct_search.h     PuctSearch / SearchResult / IStopCriterion
  random.h          确定性 RNG（PCG32、Gamma）
src/                实现
tests/              玩具游戏与单元测试（不属于库目标）
docs/               算法映射与设计说明
THIRD_PARTY_NOTICES.md
```

---

## 6. 许可证与来源

本项目的搜索算法派生自
[KataGo](https://github.com/lightvector/KataGo)（作者 David J Wu 等，MIT）。
派生文件均带有 SPDX 头与来源说明；完整 MIT 原文、派生范围及“未复制其第三方
依赖”的声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

新增代码以 MIT 发布，见 [LICENSE](LICENSE)。
