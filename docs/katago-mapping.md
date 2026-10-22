# KataGo 搜索模块 → SuZero-02 映射

本文记录提取范围、公式对应关系、刻意省略的部分，以及重写时容易出错的语义。
所有上游路径均相对 KataGo 仓库 `cpp/search/`。

上游参考版本：KataGo master 快照（2025-09-28，`katago/KataGo-master`）。

---

## 1. 类型映射

| KataGo | SuZero-02 | 说明 |
|---|---|---|
| `Search` | `PuctSearch` | 去掉了 Go 专用状态、线程池、评估缓存 |
| `SearchParams` | `SearchParams` | 仅保留游戏无关子集 |
| `SearchThread` | 无 | 单线程实现不需要每线程棋盘/随机数 |
| `SearchNode` | `SearchNode` | 保留 priors、边、统计量、状态键 |
| `SearchChildPointer` | `SearchNode::Edge` | 边 = 动作 + 非拥有子节点指针 + edge visits |
| `SearchNodeTable` | `PuctSearch::node_table_` | `StateKey -> unique_ptr<SearchNode>` |
| `GraphHash` / `graphSearchRepBound` | `IGameState::stateKey()` | 重复规则所需历史由游戏自己并入键 |
| `allocateOrFindNode` | `PuctSearch::allocateOrFindNode` | 按状态键查找或新建节点 |
| `maybeCatchUpEdgeVisits` | `PuctSearch::maybeCatchUpEdgeVisits` | 边访问落后于子访问时只补边访问 |
| `thread.graphPath` | `PuctSearch::PathSet` | 单次 playout 的路径节点集合，用于断环 |
| `makeMove` 中的 root 提升 | `PuctSearch::advanceRoot` + `copySubtree` | 复制子图重建节点表，root 不入表 |
| `NodeStats` / `NodeStatsAtomic` | `NodeStats` | 单线程普通字段 |
| `NNOutput`（policy 部分） | `Evaluation::policy` | 长度 = `actionCount()` |
| `NNOutput` 的胜负/目数 | `Evaluation::utility` | 合并为一个标量，由评估器负责 |
| `Board` / `BoardHistory` | `IGameState` | 规则完全抽象出去 |
| `NNEvaluator` | `IEvaluator` | 同步单次评估 |
| `Loc` / `NNPos` | `ActionId` | 稠密动作索引 |
| `Player`（黑白） | `PlayerId` | 只依赖相等性 |
| `addDirichletNoise` | `PuctSearch::addRootDirichletNoise` | 保留策略形状 alpha |
| `chooseIndexWithTemperature` | `PuctSearch::sampleWithTemperature` | 保留 `onlyBelowProb` 阈值语义 |
| `getPlaySelectionValues`（根） | `PuctSearch::chooseAction` | 只保留根节点最终采样所需部分 |
| `playoutDescend` | `PuctSearch::playout` | 去掉图搜索、virtual loss、缓存 |
| `recomputeNodeStats` | `PuctSearch::recomputeNodeStats` | 保留 child weight 聚合与坏子降权 |
| `addLeafValue` | `addOwnEvaluationAsLeaf` + 展开逻辑 | 终局/死路节点每次访问都计权重 |

---

## 2. 核心公式

### 2.1 Child weight

与 KataGo 一致，探索项不是直接用访问数，而是用按 edge visits 折算的权重：

```
childWeight(child) = weightSum(child) * edgeVisits / max(childVisits, 1)
```

`weightSum` 是节点自身评估权重（本实现固定为 1）与各子节点折算权重之和。
终局节点每次被访问都会累加 1 的权重，因此其 `weightSum` 会随访问增长；
这一点是正确平衡“继续挖终局”与“探索新分支”的关键。

### 2.2 图搜索、边访问与成环

- **根节点不入表**：根直接分配，`allocateOrFindNode` 永远不会返回它。这样任何
  置换边都不可能指回根，换根后也不会出现经过新根的环（KataGo 在 `Search::search`
  中同样注释了这一点）。
- **边与子节点分离**：父边记录 `edgeVisits`，子节点记录全局 `childVisits`。
  同一子节点可被多个父边引用，所以两者不必相等。
- **状态键**：展开边时先落子，再取 `IGameState::stateKey()` 查表；命中则复用
  节点，未命中才新建。节点由 `node_table_` 拥有，边只保存裸指针。
- **边访问追赶**：下探已有边之前，如果 `edgeVisits < childVisits`，按
  `graphSearchCatchUpLeakProb` 决定是否仍然下探；否则只把该边 +1、父节点 +1，
  并重新聚合父节点统计。新建立的边同样检查一次（它可能一创建就指向一个已被
  其他路径访问过的置换节点）。对应 `maybeCatchUpEdgeVisits` 与
  `updateStatsAfterPlayout`。
- **成环检测**：每次 playout 维护一个路径节点集合 `PathSet`。下探前把子节点
  插入集合；若已在集合中，说明本 playout 成环，只给该边 +1、父节点 +1，然后
  返回，不再递归。环边会保留在图中，因此
  `edgeVisits >= childVisits` **不是**不变量（KataGo 在环处理处也明确说明），
  child weight 公式仍然成立。若 `graphSearchCycleCheckEnabled = false`，则完全
  跳过路径集合的插入与查找（适用于严格无环游戏）；此时有环图只能依靠
  `maxDepth` 或调用栈上限终止。
- **换根**：`advanceRoot` 复制以目标子节点为根的整个可达子图（包括可能存在的
  环边），被提升的节点在表内保留一份副本，**新根另建一份独立副本**且只指向表内
  节点，因此新根永远不会成为任何边的目标，也就不可能位于环上。旧表整体丢弃。
  这是单线程版的“复制子树 + 标记清除”，对应 KataGo 在 `makeMove` 中
  `new SearchNode(*child, ...)`（浅拷贝子指针，旧子节点仍留在表中）后
  `applyRecursivelyAnyOrder...` 删除不可达节点的做法。

### 2.3 PUCT

```
cpuct(W) = cpuctExploration
         + cpuctExplorationLog * log((W + cpuctExplorationBase) / cpuctExplorationBase)

exploreScaling = cpuct(W) * sqrt(W + 0.01) * utilityStdevFactor

selection(child) = Q(child) + exploreScaling * P(child) / (1 + childWeight(child))
```

- `W` = 所有已展开子节点的 child weight 之和。
- `P` = 根节点使用经过温度/噪声处理的先验，其他节点使用原始先验。
- `Q` 必须先转换到**父节点行棋方**视角：
  `same player ? child.utilityAvg : -child.utilityAvg`。
- 未展开动作：`Q = fpuValue`，`childWeight = 0`。
- 平局时保留先验顺序更靠前的动作（动作 id 升序），保证确定性。

### 2.4 First Play Urgency

```
parentUtility = node.utilityAvg
parentWeightPerVisit = node.weightSum / max(node.visits, 1)
stdev = f(...)   // 见 computeFpuValue，带先验与权重下限保护
utilityStdevFactor = 1 + cpuctUtilityStdevScale * (stdev / prior - 1)

parentUtilityForFPU =
    fpuParentWeightByVisitedPolicy
      ? blend(parentUtility, node.evaluatorUtility, pow(policyMassVisited, pow))
      : blend(parentUtility, node.evaluatorUtility, fpuParentWeight)

reduction = (isRoot ? rootFpuReductionMax : fpuReductionMax) * sqrt(policyMassVisited)
fpuValue = parentUtilityForFPU - reduction
fpuValue += (-utilityRadius - fpuValue) * (isRoot ? rootFpuLossProp : fpuLossProp)
```

因为本实现的节点价值已经是“该节点行棋方视角”，所以这里始终是
`parentUtility - reduction`，不需要 KataGo 中按白/黑翻转符号的判断。

### 2.5 价值聚合与坏子降权

`recomputeNodeStats` 对齐 KataGo 的 `recomputeNodeStats`：

1. 收集所有有效子节点：`weight = childWeight`，价值转换到当前节点视角。
2. 根节点且开启噪声时，按 `chosenMoveSubtract` / `chosenMovePrune`
   做根节点专用剪枝（上限均为 `maxChildWeight / 64`）。
3. 若 `valueWeightExponent != 0`：
   - 用**剪枝前**权重计算兄弟加权均值 `simpleValue`；
   - 每个子节点精度 `precision = 1.5 * sqrt(weight)`，
     标准差 `stdev = sqrt(1e-8 + 1/precision)`；
   - `p = normalCdf((Q_i - simpleValue) / stdev) + 1e-4`；
   - `weight *= p ^ valueWeightExponent`，再归一化回剪枝后的总权重。
4. 加入节点自身评估（权重 1），得到
   `weightSum / utilitySum / utilitySqSum`。

KataGo 用 `DistributionTable` 近似正态 CDF；本实现直接用 `std::erfc`。

### 2.6 根节点噪声

`addRootDirichletNoise` 保留 KataGo 的 **策略形状 alpha 分布**，而不是简单的
对称 Dirichlet：

```
logP_i  = log(min(0.01, P_i) + 1e-20)
alphaProp_i = max(0, logP_i - mean(logP))
alpha_i = 0.5 * (alphaProp_i / sum(alphaProp) + 1/legalCount)

noise_i = Gamma(alpha_i * totalConcentration)
P'_i = (1 - weight) * P_i + weight * noise_i / sum(noise)
```

噪声只在某个节点首次成为根节点时生成一次并缓存；重复调用 `run()` 不会重新加噪。
根策略温度在噪声之前应用。

### 2.7 终局与视角

- 终局节点的 `terminalUtility()` 从**终局状态当前行棋方**视角返回。
- 备份到父节点时，仅当 `child.player != parent.player` 才取负。
- 因此同时支持：
  - 正常交替行棋；
  - 5D 象棋这类同一方连续小步、仅在真正换方时翻转价值的游戏。

---

## 3. 刻意省略的部分

| 上游特性 | 原因 |
|---|---|
| 多线程 + `NodeStatsAtomic` + `dirtyCounter` | 抽象阶段先保证语义正确；接口不阻止后续改成原子统计 |
| virtual loss | 单线程无收益 |
| eval cache（`EvalCache`、`evalCacheEntry`） | 与图搜索和线程同步强耦合；当前只保留纯搜索语义 |
| `GraphHash` 的增量哈希实现 | 由 `IGameState::stateKey()` 取代；游戏自行决定键的表示与历史范围 |
| `graphSearchRepBound` 的 Go 超级劫细节 | 重复规则属于游戏；键里应自行折叠所需历史 |
| `subtreeValueBias` | 与 NN 校准强相关，游戏无关性弱 |
| 目数/胜负双通道效用、ownership、lead | 归入 `Evaluation::utility`，由评估器定义 |
| 贴目、劫、pass、领地、dame | 纯 Go 规则 |
| 对称性剪枝 / 多对称采样 | 纯 Go 棋盘对称性 |
| mirror / anti-mirror | 纯 Go 对手建模技巧 |
| visit cap / 对手建模 | 需要额外根节点状态；当前版本未纳入 |
| LCB 最终选择 | 可独立实现；当前 `chooseAction` 只做 child weight + 温度 |
| 时间控制策略、ponder、search factor | 游戏无关但依赖对局管理；当前只提供 wall-clock 上限 |
| 批量推理 | 与评估器异步化一起设计更合适 |
| uncertainty weighting | 需要评估器提供短期误差；当前权重固定为 1 |

---

## 4. 与上游行为的已知差异

1. **单线程**：所有统计量是普通字段，不可并发访问同一个 `PuctSearch`。
2. **终局节点形态**：上游终局节点通常没有 NNOutput；本实现把
   `terminalUtility()` 存为节点自身评估值，并作为叶子值处理。
3. **根节点加噪时机**：上游通过 `maybeAddPolicyNoiseAndTemp` 惰性写入
   `noisedPolicyProbs`；本实现把有效先验放在 `root_priors_`，语义相同且避免
   修改树中节点的原始先验。
4. **`bestAction`**：按根边访问数选，平局依次比较先验、动作 id；上游最终着法
   还受 LCB、pass hack 等影响。
5. **`maxVisits` 包含根节点首次评估的 1 次访问**；`maxPlayouts` 只统计本次
   `run()` 新执行的 playout。
6. **动作概率为零的合法动作**不会进入先验列表，因此不会被主动探索——与
   KataGo 中 policy prob 为 0 的行为一致。
7. **评估器策略长度**必须严格等于 `actionCount()`，否则抛异常；这比上游的
   `MAX_NN_POLICY_SIZE` 固定数组更严格，便于尽早发现适配层错误。
8. **状态键由游戏负责**：上游用 128 位 `GraphHash` 并把重复规则的历史折叠进去；
   本实现要求 `stateKey()` 本身满足等价性契约。字符串比较没有哈希碰撞问题，但
   键的构造成本、历史范围和重复规则正确性都转移给游戏适配层。若键遗漏了影响
   合法性/终局的历史，可能出现错误置换。
9. **换根会重建图**：`advanceRoot` 复制子图并丢弃旧节点表，因此之前通过
   `rootNode()`/`childAt()` 取得的节点指针全部失效；`resetRoot` 同理。
10. **无 eval cache / forceNonTerminal / conservative pass**：这些与 Go 规则、
    线程和缓存相关，未移植。
11. **成环时 `edgeVisits` 可能大于 `childVisits`**：与上游一致，child weight
    公式仍按两者比值计算。
12. **`nodeCount()`** 返回根节点加节点表大小；根始终计入，但不占用表项。
13. **`graphSearchCycleCheckEnabled` 是新增参数**：上游总是检查路径成环；这里
    允许无环游戏显式关掉该检查以省去每步的集合操作。关闭后正确性由游戏保证。

---

## 5. 许可证注意事项

- KataGo 搜索模块属于其 MIT 许可内容（Copyright 2025 David J Wu 等）。
- 本项目**没有**复制 KataGo 的 `cpp/external/*`（CUDA、OpenCL、cutlass、
  nlohmann_json 等），这些库有各自许可证。
- 如果后续直接引入 KataGo 源文件或外部库，必须同时保留对应文件/库的许可证，
  不能只依赖本项目的 `THIRD_PARTY_NOTICES.md` 中的 MIT 文本。
