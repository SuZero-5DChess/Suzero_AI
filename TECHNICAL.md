# TECHNICAL — SuZero-02 PUCT 图搜索核心

本文件记录当前实现状态、关键不变量与变更历史。算法与上游对应关系详见
[docs/katago-mapping.md](docs/katago-mapping.md)。

## 当前状态（2026-10-22）

- 版本：0.2.0
- 语言/标准：C++17，仅标准库
- 搜索：单线程、**图搜索（DAG，置换合并）**、同步评估
- 接入状态：**未接入任何游戏引擎**
- 抽象接口：`IGameState`、`IEvaluator`、`IStopCriterion`
- 构建：CMake + Visual Studio 2022 通过（MSVC 19.44，Release 与 Debug）
- 测试：`suzero_mcts_tests`，188 项检查全部通过

## 不可破坏的不变量

1. 节点价值、评估器效用、终局效用都从**该状态当前行棋方**视角表示。
2. 备份时只在 `player` 确实变化时取负。
3. 父节点每完成一次 playout（含追赶、成环、叶子评估）恰好增加 1 次访问。
4. `edgeVisits` 是该父边上的访问数，`childVisits` 是该子节点的全局访问数；
   图搜索下两者不保证相等，追赶后可能 `edgeVisits > childVisits`。
5. 根节点不在 `node_table_` 中，任何置换边都不能指回根。
6. `stateKey()` 相等必须意味着两个状态对搜索完全等价（含行棋方、合法性、
   评估、终局与终局效用，以及重复规则所需历史）。
7. 终局节点与“无可行动作”的节点每次访问都以自身评估值作为叶子值累加权重。
8. `Evaluation::policy.size() == IGameState::actionCount()`，否则抛异常。
9. 先验列表只包含合法且策略值大于 0 的动作，按动作 id 升序，保证确定性。
10. 根节点噪声与策略温度只作用于 `root_priors_`，不修改图中节点的原始先验。
11. 同一个 `PuctSearch` 实例不支持并发调用。
12. 关闭 `graphSearchCycleCheckEnabled` 后不再维护 playout 路径集合；这只对严格
    无环的图安全，有环时只能依赖 `maxDepth` 或调用栈上限终止。

## 实现要点

- `allocateOrFindNode`：按 `stateKey()` 查表，命中复用、未命中新建；节点由表拥有。
- `playout`：展开叶子 → 选择 → 建边/查找置换 → 边访问追赶 → 路径成环检测 →
  下探 → 更新 edge visit → 重算节点统计。
- `maybeCatchUpEdgeVisits`：`edgeVisits < childVisits` 时只补边访问而不下探；
  `graphSearchCatchUpLeakProb` 控制偶尔仍然下探。
- `PathSet`：每次 playout 的节点集合；再次遇到即判定成环，只计边访问。
  `graphSearchCycleCheckEnabled = false` 时不创建、不清空该集合，playout 中也
  不做插入/查找。
- `selectPrior`：PUCT + FPU + utility stdev 缩放；平局保留较前先验。
- `recomputeNodeStats`：child weight 聚合、根节点 subtract/prune、
  `valueWeightExponent` 坏子降权、加入自身评估权重 1。
- `addRootDirichletNoise`：KataGo 策略形状 alpha 的 Dirichlet 噪声。
- `sampleWithTemperature`：保留 `onlyBelowProb` 阈值语义。
- `advanceRoot`：复制子图、重建节点表、丢弃旧表；被提升节点在表内保留副本，
  新根是独立的、不是任何边目标的副本，保证根不在环上。找不到已展开边时返回
  false 且不改变状态。复制后旧节点指针全部失效。

## 已知限制与后续方向

1. 多线程 + virtual loss + 原子统计。
2. 评估缓存及其与置换节点的交互。
3. 批量评估与异步评估器接口。
4. 更多最终着法选择策略（LCB 等）。
5. 可选的 debug/统计输出接口。
6. 更紧凑的状态键（当前为 `std::string`，键构造成本由游戏承担）。

## 变更记录

### 2026-10-22 · 成环检查可关闭

- 新增 `graphSearchCycleCheckEnabled`；关闭时不维护 playout 路径集合。
- 新增无环图开关等价性、有环图加 `maxDepth` 的终止测试；测试增至 188 项检查。

### 2026-10-22 · 改为纯图搜索

- `IGameState::stateKey()` 改为必选纯虚接口，明确等价性契约。
- 引入节点表、非拥有边指针、根节点不入表。
- 移植 KataGo 的边访问追赶、成环检测、子图复制换根。
- 新增 `graphSearchCatchUpLeakProb` 参数。
- 新增置换合并、成环终止与跨环换根测试；测试增至 176 项检查。

### 2026-10-22 · 初始提取

- 新建项目与静态库 `suzero_mcts`。
- 从 KataGo `cpp/search/` 提取游戏无关的 PUCT 语义并重写为抽象接口。
- 增加确定性 RNG（PCG32 + Gamma）、根 Dirichlet 噪声、温度采样。
- 增加玩具游戏单元测试与 CMake/CTest。
- 增加 MIT 许可、KataGo 第三方声明与算法映射文档。
