# TECHNICAL — SuZero-02 MCTS 核心

本文件记录当前实现状态、关键不变量与变更历史。算法与上游对应关系详见
[docs/katago-mapping.md](docs/katago-mapping.md)。

## 当前状态（2026-10-22）

- 版本：0.1.0
- 语言/标准：C++17，仅标准库
- 搜索：单线程、树搜索（非 DAG）、同步评估
- 接入状态：**未接入任何游戏引擎**
- 抽象接口：`IGameState`、`IEvaluator`、`IStopCriterion`
- 构建：CMake + Visual Studio 2022 通过（MSVC 19.44，Release 与 Debug）
- 测试：`suzero_mcts_tests`，158 项检查全部通过

## 不可破坏的不变量

1. 节点价值、评估器效用、终局效用都从**该状态当前行棋方**视角表示。
2. 备份时只在 `player` 确实变化时取负。
3. 每条边只有在真正下探后才增加 edge visits；父节点每次 playout 增加一次 visits。
4. 终局节点与“无可行动作”的节点每次访问都以自身评估值作为叶子值累加权重。
5. `Evaluation::policy.size() == IGameState::actionCount()`，否则抛异常。
6. 先验列表只包含合法且策略值大于 0 的动作，按动作 id 升序，保证确定性。
7. 根节点噪声与策略温度只作用于 `root_priors_`，不修改树中节点的原始先验。
8. 同一个 `PuctSearch` 实例不支持并发调用。

## 实现要点

- `PuctSearch::playout`：展开叶子 → 选择 → 下探 → 更新 edge visit → 重算节点统计。
- `PuctSearch::selectPrior`：PUCT + FPU + utility stdev 缩放；平局保留较前先验。
- `PuctSearch::recomputeNodeStats`：child weight 聚合、根节点 subtract/prune、
  `valueWeightExponent` 坏子降权、加入自身评估权重 1。
- `PuctSearch::addRootDirichletNoise`：KataGo 策略形状 alpha 的 Dirichlet 噪声。
- `PuctSearch::sampleWithTemperature`：保留 `onlyBelowProb` 阈值语义。
- `advanceRoot`：复用子树；找不到已展开边时返回 false 且不改变状态。

## 已知限制与后续方向

1. 多线程 + virtual loss + 原子统计。
2. Graph search：以 `IGameState::stateKey()` 索引节点，合并置换。
3. 批量评估与异步评估器接口。
4. 更多最终着法选择策略（LCB 等）。
5. 评估缓存与根节点复用时的缓存失效规则。
6. 可选的 debug/统计输出接口。

## 变更记录

### 2026-10-22 · 初始提取

- 新建项目与静态库 `suzero_mcts`。
- 从 KataGo `cpp/search/` 提取游戏无关的 PUCT 语义并重写为抽象接口。
- 增加确定性 RNG（PCG32 + Gamma）、根 Dirichlet 噪声、温度采样。
- 增加玩具游戏单元测试与 CMake/CTest。
- 增加 MIT 许可、KataGo 第三方声明与算法映射文档。
