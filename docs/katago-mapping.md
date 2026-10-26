# KataGo 搜索模块 → SuZero AI 映射

本文记录提取范围、公式对应关系、刻意省略的部分，以及重写时容易出错的语义。
所有上游路径均相对 KataGo 仓库 `cpp/search/`。

上游参考版本：KataGo master 快照（2025-09-28，`katago/KataGo-master`）。

---

## 1. 类型映射

| KataGo | SuZero AI | 说明 |
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
| `NNOutput`（policy 部分） | `Evaluation::policy` | 长度 = **当前状态的** `actionCount()` |
| `NNOutput` 的胜负/目数 | `Evaluation::utility` | 合并为一个标量，由评估器负责 |
| `Board` / `BoardHistory` | `IGameState` | 规则完全抽象出去 |
| `NNEvaluator` | `IEvaluator` | 同步单次评估 |
| `NNEvaluator` 的查询队列 + server 线程 | `BatchedEvaluator` + `IBatchBackend` | `waitPopUpToN` 语义；每 server 线程独占一个 backend |
| `selfplay` 命令的对局线程 | `SelfPlayRunner` | 每局独占 `PuctSearch`，共享评估器；无 manager/热切换 |
| `Loc` / `NNPos` | `ActionId` | 稠密动作索引 |
| `Player`（黑白） | `PlayerId` | 只依赖相等性 |
| `addDirichletNoise` | `PuctSearch::addRootDirichletNoise` | 保留策略形状 alpha |
| `chooseIndexWithTemperature` | `PuctSearch::sampleWithTemperature` | 保留 `onlyBelowProb` 阈值语义 |
| `getPlaySelectionValues`（根） | `PuctSearch::computeRootPlaySelectionValues` | 非 LCB 路径：最佳子节点启发式 + 回顾式降权 |
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
  环边），旧表整体丢弃。表内只为仍被某条边引用的节点保留副本；被提升节点没有
  入边（它是新子树的根），所以它在表内的旧副本会被删除。**新根另建一份独立
  副本**且只指向表内节点，因此新根永远不会成为任何边的目标，也就不可能位于
  环上。这是单线程版的“复制子树 + 标记清除”，对应 KataGo 在 `makeMove` 中
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
- 选择顺序与 KataGo 一致：先在所有**已展开**子节点中按上式取最大；再在
  **尚未展开**的动作中只取策略概率最高者，计算它的选择值并与已展开子节点
  比较。因此一次只会新展开一个动作。
- 策略概率为 `0` 的合法动作仍会进入先验列表并可被展开，只是探索项为 0；
  负值表示不可用/非法，会被过滤。
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
   - `p = studentTCdf3((Q_i - simpleValue) / stdev) + 1e-4`；
   - `weight *= p ^ valueWeightExponent`，再整体归一化回**剪枝前**的原始总权重
     （`originalTotalWeight / totalChildWeight`），使剪枝只改变相对分布，不改变
     节点的绝对权重。
4. 加入节点自身评估（权重 1），得到
   `weightSum / utilitySum / utilitySqSum`；`utilitySqSum` 使用子节点的
   `utilitySqAvg()`，用于后续价值标准差估计。

KataGo 用 `DistributionTable` 近似 **3 自由度 Student-t** 的 CDF；本实现使用
闭式公式：

```
studentTCdf3(z) = 0.5 + (atan(z / sqrt(3)) + sqrt(3) * z / (3 + z^2)) / pi
```

两者数值一致，但闭式实现不需要查表。

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

根策略温度不是硬性的 one-hot 截断，而是在对数域按温度缩放：

```
P'_i ∝ P_i ^ (1 / T)
```

温度本身按对局步数在早期值与最终值之间插值：

```
T(move) = T + (T_early - T) * 0.5 ^ (move / halflife)
```

本实现直接使用 `IGameState::moveCount()` 作为步数；KataGo 还会把半衰期按
`19 / sqrt(boardArea)` 缩放，棋盘面积由上游的 Go 规则提供，因此这里没有该因子。
即使 `T` 很小，并列动作也会保留概率份额，直到 `exp()` 下溢，不会人为制造
one-hot。

### 2.7 根节点最终着法权重（非 LCB）

`computeRootPlaySelectionValues` 对齐 KataGo `getPlaySelectionValues` 的根节点
非 LCB 分支：

1. 只考虑已展开的根边，候选权重为 child weight；若没有已展开子节点，退回
   直接使用根策略概率（同样应用 subtract/prune）。
2. 用
   `weight * max(0, edgeVisits - 1) / max(1, edgeVisits) + 2 * prior`
   选出“探索最稳定”的最佳子节点，策略项只用于打破平局。
3. 最佳子节点保留完整 child weight；其余子节点被回顾式降权到与最佳子节点的
   PUCT 选择值相等所需的权重，并向上取整：

```
wanted  = exploreScaling * P / (bestValue - value) - 1
reduced = min(weight, max(0, wanted))
final   = ceil(reduced)
```

4. 最后对所有权重应用 `chosenMoveSubtract` / `chosenMovePrune`（上限均为
   `maxValue / 64`）。

`SearchResult::bestAction` 取上述权重的最大项；无参 `chooseAction()` 用同一组
权重做温度采样，并按 `chosenMoveTemperatureHalflife` 在
`chosenMoveTemperatureEarly` 与 `chosenMoveTemperature` 之间插值。显式传入
温度的 `chooseAction(temperature)` 只使用该温度，不做插值。

### 2.8 状态相关动作空间

KataGo 的棋盘动作空间固定为 `NNPos` 的稠密大小；本库允许
`IGameState::actionCount()` 随状态变化，以支持 5D 象棋这类合法动作按局面
生成的游戏。约束如下：

- 动作 id 在**该状态内**稠密，`[0, actionCount())`；跨状态可指向不同动作。
- `actionCount()` 必须是状态的确定函数：`stateKey()` 相等的状态必须返回相同
  数量，并以相同顺序枚举相同动作。否则图搜索的置换节点会共享错误的先验。
- 每个节点保存自己的 `priors_`；`expandNode` 用该节点的状态和评估器策略
  构建先验，不再假设整个搜索只有一个固定长度的动作数组。
- `resetRoot` 只检查新根的动作数 > 0，允许动作数与旧根不同；
  `advanceRoot` 在提升子树后更新 `action_count_ = new_root.actionCount()`。
- 评估器必须为传入的每个状态返回长度恰为 `state.actionCount()` 的策略向量。

### 2.9 终局与视角

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
| LCB 最终选择 | 非 LCB 的根节点回顾式权重已实现；LCB 的 `weightSqSum` / 置信下界尚未移植 |
| 棋盘面积对早期温度半衰期的缩放 | 面积属于具体游戏；本实现只按步数插值 |
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
4. **`bestAction`**：取根节点最终 play-selection 权重的最大项（KataGo
   `getChosenMoveLoc` 的非 LCB 语义）；上游在启用 LCB 时还会结合置信下界，
   本实现未移植该分支。
5. **`maxVisits` 包含根节点首次评估的 1 次访问**；`maxPlayouts` 只统计本次
   `run()` 新执行的 playout。时间限制至少执行 2 次 playout 后才生效，避免
   根节点尚未展开就返回空结果。
6. **动作概率为零的合法动作**会进入先验列表并可被展开（探索项为 0），与
   KataGo 保留 `policyProbs >= 0` 的行为一致；只有负值（或 NaN）才表示不可用。
7. **评估器策略长度**必须严格等于**当前状态的** `actionCount()`，否则抛异常；
   这比上游的 `MAX_NN_POLICY_SIZE` 固定数组更严格，便于尽早发现适配层错误。
   动作数允许随状态变化（见 2.8）。
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
14. **早期温度插值只按步数**：`rootPolicyTemperatureEarly` /
    `chosenMoveTemperatureEarly` 与 `chosenMoveTemperatureHalflife` 都已实现，
    但上游会把半衰期乘以 `19 / sqrt(boardArea)`；本实现没有棋盘概念，因此
    `chosenMoveTemperatureHalflife` 直接以步数为单位。
15. **显式温度不插值**：`chooseAction(temperature)` 原样使用调用者给出的温度；
    只有无参 `chooseAction()` 和根策略温度会应用早期插值。这与上游配置路径
    一致，也便于调用者完全控制采样。
16. **动作空间可以随状态变化**：上游 `NNPos` 是固定大小的稠密数组；本库把
    `actionCount()` 定义为状态的函数，节点各自持有先验。相等的 `stateKey()`
    必须保证相同的动作数与动作顺序，否则会破坏置换合并。

---

## 5. 训练器（Python）与上游对应

| 上游 `python/train.py` 中的方法 | 本仓库 |
|---|---|
| epoch/步数/checkpoint 编排 | `suzero_train/loop.py`（通用 `TrainingLoop`） |
| per-sample LR、AdamW 因子、batch 缩放 | `pytorch/schedules.py`、`pytorch/trainer.py` |
| 分段 warmup | `pytorch/schedules.py` 的 `warmup_scale` |
| 参数组 LR / weight decay | `pytorch/config.py` 的 `ParamGroupSpec`、`pytorch/trainer.py` |
| 梯度范数上限与裁剪 | `pytorch/trainer.py` 的 `_grad_norm_cap` |
| 指标累计、`_sum`/`_batch` 平均、EMA | `pytorch/metrics.py` |
| checkpoint 字段与续训 | `pytorch/trainer.py` 的 `save_checkpoint` / `load_checkpoint` |
| 上游 `python/shuffle.py` 的增长窗口、训练器文件消费思路 | 5D 侧 `replay.py` 管 RAM 窗口，`file_replay.py` 发布/消费分片，`pipeline.py` 同步编排；见 5.1 |
| 模型、Board 编码、围棋 `.npz` 数据格式、导出 | **未移植** |

- 保持一致：per-sample LR `3e-5 * learning_rate_scale`（AdamW 再乘 `1.33`）、
  warmup 曲线（1/20 … 1/1.4）、LR 的 batch 指数（AdamW 0.5 / SGD 0.0）与
  独立的 WD batch 指数（AdamW 0.5 / SGD 1.0）、梯度上限的
  `sqrt(batch/256)` 与 `1/sqrt(lr_scale)` 缩放、metric 衰减 `0.995`。
- 泛化差异：KataGo 按 normalization 类型（fixup/bnorm 等）选择 weight decay
  的表是模型相关的，本仓库改为 `weight_decay` + 参数组系数显式配置。
- 未复制上游 Python 文件；实现基于其公式重写。

### 5.1 Lossy replay window 与上游 shuffle 的对应

上游路径相对 KataGo 仓库根目录，不属于前文的 `cpp/search/`。本节核对了本地
`vibe2/5dc_ml-main/katago/KataGo-master` 快照中的：

- `python/shuffle.py`：窗口增长与 shuffled 数据发布思路；
- `python/train.py`：`-no-repeat-files` 参数、`get_files_for_subepoch` 整文件配额；
- `python/katago/utils/training_data_generator.py`：文件队列与已用完整路径持久化；
- `python/katago/train/data_processing_pytorch.py`：每文件只读完整 batch；
- `python/selfplay/synchronous_loop.sh`：同步自对弈/训练循环的 no-repeat-files 用法。

本项目保留**有损 RAM 窗口**，仅将训练交接改为文件式：
`python/suzero_5dchess/replay.py` 管理逐样本 zlib FIFO 与快照，`file_replay.py` 负责
`publish_shuffle` / `FileReplayDataSource`，`pipeline.py` 负责同步编排。不是完整
raw-disk 数据集，不移植围棋 NPZ 样本协议或永久 raw 数据仓库。

- **相同思路**：随累计生成量扩大最新窗口；把训练交接改为文件式 shuffle；读完后
  交还同步循环等待窗口更新。
- **范围不同（no-repeat 身份不同）**：窗口外样本永久淘汰，没有 `add_to_data_rows`
  或 raw 历史归档，不重新加载已丢弃的旧窗口；并且只有训练计数大于 0 的行才允许
  淘汰，未训练积压会把窗口暂时顶到目标之上。更关键的是，本实现的去重身份是
  **窗口行**，不是上游的输出文件路径：新一轮不选上一轮**实际训练过**的行，上一轮
  选中却没读到的行仍可重新提供；即使写到新目录也不允许重读已训练的行。只保留当前
  shuffled 训练视图，旧轮/临时视图会删除；`keep_samples` 限制每轮子集，不复制上游
  keep-target 的整套数据管线。

`WindowConfig` 默认 `enabled=False`，保持旧固定容量 uniform 回放；开启后的默认
参数为 `expand_per_row=0.4`、`taper_exponent=0.65`、`taper_scale=None`、
`max_samples=500000`、`keep_samples=None`、`sampling="shuffle"`。
`min_rows` 直接取现有的 `replay_buffer_size`（默认 100,000），不是 `min_replay_size`。

令 `N` 为累计成功 push 的训练样本数，`m=min_rows`，
`s=taper_scale if specified else m`，目标窗口精确为：

```text
W(N) = floor(clamp(
    m + expand_per_row * (N**taper_exponent - s**taper_exponent)
        / (taper_exponent * s**(taper_exponent - 1)),
    m, max_samples))
```

这里先钳制到有限上下界、再向下取整。`N` 不因淘汰而减少，也不因训练反复读取而增加。
本轮所有游戏完成 push 后、训练前才按新 `N` trim 一次；启动恢复后也 trim。
`capacity=max_samples` 另在 push/加载时限制行数，避免裁剪前无界增长；但这些限制只
淘汰训练计数大于 0 的前导行，因此未训练积压存在时是软上限。实际行数还受已有历史
限制。`evicted` 为本轮容量保护与动态 trim 合计删除的行数。

**上游 no-repeat-files 的粒度是文件完整路径，不是行。** 该快照的 tracked-used
语义由 `train_state["data_files_used"]` 和 `rev_data_files_remaining` 实现，跨 epoch
与 checkpoint 保留已用路径；生成器不重填耗尽队列。新输出目录/文件即为新路径，
同一原始位置出现在新 shuffle 中不算旧文件重读，并非 row-global 去重。
`get_files_for_subepoch` 对超过软 epoch 配额的**整文件**随机接受或延期，延期的文件
保持未消费；读取器按 `num_rows // batch_size` 只输出完整 batch，文件尾不足一批
的行被丢弃。上游没有 `partial-filefraction` 参数，不能用它解释这一行为。

本实现为固定预算明确作如下适配，而非宣称逐行复制上游行为：

- **发布**：`--window` 且 `--window-sampling shuffle` 且本轮确实要训练时发布；
  默认 4096 行/NPZ 分片（内装原有 zlib blob、offsets 等数值数组），写 manifest
  后原子切换 `current.json`。manifest 记录本轮窗口行 `rows`、窗口行数
  `pool_samples` 和排除集摘要 `exclude_digest`。同输入、同 seed 发布重试幂等复用
  同一轮。每轮无放回选择 `min(keep_samples, 可用行)` 行（未指定则全部可用行）；
  这里的"可用行"是当前窗口去掉上一轮实际训练过的行之后的行。上一轮选中但因预算
  或中断没有读到的行仍是 0 训练计数，下一轮会重新提供。
- **消费与多轮 epoch**：`FileReplayDataSource` 按已洗牌分片连续读；
  generation/file/row 游标跨 epoch 保留，不重排、不重置、不回绕。本轮读完后若
  epoch 预算未用完，就在训练线程内发布下一轮（排除刚读完那轮已训练的行）继续。
  因此一个 epoch 可以包含多轮 shuffle，而相邻两轮已训练的行集合不交。
- **跨 iteration 平移**：新阶段发布前，用
  `pool_samples + (N_now - N_then) - len(buffer)` 计算窗口头部淘汰数，把上一轮
  行索引整体前移；已淘汰的行退出排除集。窗口只从头部删除，所以平移是精确的。
- **文件后端**：默认的 `--replay-backend file` 把每个完成对局追加为一帧到单文件
  `replay_store/replay.log`，同样按行级规则抽轮；上一轮选择的 `(game_id, row)`
  原子写入 `replay_store/last_round.json`，下一阶段只排除其中已训练的行，读取按
  帧顺序且只持有当前对局。
- **预算与尾行**：每个 epoch 的上限仍是 `requested_steps`（无可选行时为 0）。
  跨片拼 batch，不以整文件随机接受方式超出预算；当前分片未读行延续到下一 epoch，
  最后不足 batch 时允许 partial，不重复补样，不像上游丢弃文件尾行。不会为了凑
  预算重读上一轮已训练的行；上一轮没读到的未训练行会被重新提供。
- **同步耗尽**：排除上一轮后没有可选行时停止本轮训练并标记
  `waiting_for_shuffle`；下一轮外层自对弈更新窗口后再发布。不在训练线程内无限
  等数据，也不循环旧轮文件。epoch/step/batch 参数值不改，不足时少训。
- **uniform**：`--replay-backend memory` 且 `sampling="uniform"` 时仍从整个当前
  buffer 在内存采样，每 batch 内无放回、批间可重复，不应用 keep 子集或文件队列。
  文件后端始终走文件顺序抽轮，因此它与 `uniform` 的组合会被配置校验拒绝。
  默认 `WINDOW_MODE=0`，配置/CLI 名称和默认值不变；文件后端在窗口关闭时仍按
  文件抽轮。

恢复时累计生成计数仍取
`max(extra.total_samples, buffer.total_generated, len(buffer))`；历史计数都缺失时，
不能从现存行重建已淘汰的生成历史。文件后端把每局 NPZ 留在 `replay_store/`，
`replay_buffer.pt` 只保存计数、文件索引与窗口状态；`memory` 后端则把整个当前
窗口序列化进 `replay_buffer.pt`，不只保存 keep 子集。replay 的 `shuffle_state`
和模型 extra 的 `replay_shuffle` 保存消费 generation/file/row 状态。这只保证
checkpoint 边界恢复，不是每个 optimizer step 的 exactly-once 崩溃事务。不新增
旧窗口恢复入口；残留旧权重/快照不意味着能还原任意历史窗口，删除旧轮也不会触发
从 raw 历史补回数据。

`training.shuffle` 记录 `generation`、`files`、`file_index`、`row_offset`、`files_used`、
`pool_samples`、`round_samples`、`remaining_samples`、`rounds`、`no_repeat_rows`、
`waiting_for_shuffle`；coverage 使用实际训练呈现行数而非请求配额。

窗口大小不是复用预算：令每轮实际训练呈现量为 `T`、生成量为 `G`，稳态近似平均
复用为 `T / G`，`W` 改变存留时间、覆盖比例和时间跨度，不自动扩大训练量。默认
10 × 10 × 256 = 25,600 是上限；只有实际用满且假设 `G=11,400` 时才约为
`25,600 / 11,400 ≈ 2.25`。行级 no-repeat、keep 或剩余数据不足可使 `T < 25,600`，
必须代入实际 `T`。11,400 仅为示例，不是本轮实测，不承诺非稳态满足近似。

RAM 仍保存 zlib 窗口，磁盘增加当前 shuffled 副本；切轮时新旧两轮短暂并存有双份
视图峰值。容量估算应基于代表性样本的 **zlib 实测字节数**，另计 NPZ/manifest、
对象/索引、发布/保存临时副本、分片加载与 batch 解码/padding；不使用每行 60 KB
假设，不提供未经测量的压缩比。该适配不改正式训练、自对弈、对战预算或 gate 行为。

---

## 6. 批量推理与自对弈

上游参考：`cpp/neuralnet/nneval.cpp`、`cpp/core/threadsafequeue.h`、
`cpp/command/selfplay.cpp`、`cpp/program/selfplaymanager.cpp`，以及
`cpp/configs/training/selfplay*.cfg`。

| KataGo | SuZero AI | 说明 |
|---|---|---|
| `NNEvaluator::evaluate` 入队 + 条件变量等待 | `BatchedEvaluator::evaluate` | 每个请求一个 `Request`，阻塞到被服务 |
| `m_numNNServerThreadsPerModel` | `BatchEvaluatorConfig::numServerThreads` | 每线程一个 backend（一份模型） |
| `nnMaxBatchSize` | `BatchEvaluatorConfig::maxBatchSize` | 再被 `IBatchBackend::maxBatchSize()` 钳制 |
| `ThreadSafeQueue::waitPopUpToN` | `serve()` 取 `min(queue.size(), cap)` | 阻塞到非空，之后不等待凑满 |
| 数据队列背压 | `maxQueuedRequests`（默认 `cap*4*threads`） | 超过容量时 `evaluate()` 阻塞 |
| `numGameThreads` + `numSearchThreadsPerGameThread=1` | `SelfPlayConfig::numGameThreads` | 每局一个单线程 `PuctSearch` |
| `SelfplayManager` / 模型热切换 / 数据写出线程 | **未移植** | 本库只提供进程内对局与批量推理原语 |
| per-game seed（`getRandomSeed` 一类） | `SelfPlayRunner::seedForGame` | SplitMix64 混合 `(seed, gameIndex)`，与调度无关 |

与上游的刻意差异：

- KataGo 的搜索是异步的（每个 `SearchThread` 自己发请求、做 virtual loss、
  继续下探），本库 `evaluate()` 是阻塞调用，一条对局线程在评估期间停住；
  吞吐依靠多对局并发填满 batch。
- 上游按“模型”组织 server 线程并支持多模型（不同权重/后缀）同时服务；
  本库一次构造对应一组 backend，不支持运行期换模型。
- 上游把 batch 请求分发到多个 device/stream 并聚合；本库的 backend 在一次
  `evaluateBatch` 内完成整批推理，没有流/设备调度层。
- 上游某行评估失败会单独标记；本库 backend 抛异常时同 batch 所有请求一起
  收到该异常（模型推理本身是整批原子的，失败通常不是单行可恢复的）。

### 6.1 树内位置训练行（recordTreePositions）

本节路径相对 KataGo 仓库根目录，参考 `cpp/program/play.cpp` 的
`recordTreePositionsRec` / `recordTreePositions`，以及
`cpp/dataio/trainingwrite.{h,cpp}` 的 `SidePosition` 写出逻辑。

| KataGo | SuZero 5D 训练后端 | 适配 |
|---|---|---|
| `recordTreePositions` | `record_tree_positions` / `recordTreePositions` | 默认开启，只复用 full-search 树，不增加搜索；可用 `--no-record-tree-positions` 关闭 |
| `recordTreeThreshold` | `record_tree_min_visits` | 同时限制到达边及节点出边总访问数，默认 64 |
| `recordTreeTargetWeight` | `record_tree_sample_probability` | 默认 0.25 概率保留，保留后按普通行训练；不改 loss |
| wrapper 的 `maxDepth=5` | `record_tree_max_depth` | 默认 5 个小步，含 SUBMIT，可配置 1–64 |
| `plaAlwaysBest` / `oppAlwaysBest` | 按真实 player 更新的 always-best 标记 | 5D 同方可连续走多个小步，不按深度交换 |
| `SidePosition.whiteValueTargets` | `Node.q` / `NodeStats.utilityAvg()` | 搜索价值已是候选当前行棋方视角，不能用实战结果替代 |
| `extractPolicyTarget` | 节点自身出边访问次数归一化 | 不移植 play-selection 调整或 int16 策略压缩 |

额外限制：不记录根，排除本次实际着法的整个根分支；跳过终局、未展开及仅 SUBMIT
位置；局面键去重、防环，单次最多检查 4096 个节点、默认最多保留 4 行。上限按搜索
决策小步计算，不是按整回合。C++ 图节点不持状态，沿动作克隆/重放并在换根前编码；
策略使用节点局部 edge visits，不能使用共享子节点的全局 visits 替代。该功能默认
开启，可用 `--no-record-tree-positions`（启动脚本 `TREE_POSITIONS=0`）关闭；
默认开启时旧扩展缺少 `TREE_POSITION_FORMAT_VERSION=1` 会明确报错，需重建或关闭。

KataGo 将 fractional targetWeight 随机整数化为训练行数，本实现借鉴这一低侵入
策略，但加入访问/深度/数量限制，不声称与其完整数据分布严格等价。提取使用独立
随机流，既不修改主搜索统计，也不改变实战随机序列。复用的是已有含根噪声影响的
搜索，不额外进行“去噪搜索”。克隆、去重与编码仍有成本。

数值来源数组经 C++ binding / 多进程 NPZ 转为现有 PositionSample metadata，
随后沿原压缩 replay 保存，无 ONNX 或 replay 格式升级。真实对局归档保持不变。

**本次未实现**：`sidePositionProb` 选择替代动作后的赛后独立搜索，以及
`reanalyzeCheapSearchPositions` 对 fast 实战位置的重搜索；两者都不是树内位置复用，
不可因同样写入上游 `sidePositions` 就混为一个功能。

## 7. 同步 gatekeeper 与模型准入

此节上游参考路径相对 KataGo 仓库根目录：`cpp/command/gatekeeper.cpp`，
不属于搜索模块。借鉴的是模型角色和数据生产权限，不是复制完整分布式训练架构。

| KataGo 角色/行为 | SuZero AI 对应 |
|---|---|
| 训练持续产生候选模型 | trainer.model，latest.pt / agent_final.pt 仍保存训练最新状态 |
| accepted models 供 selfplay 读取 | accepted/state.json 唯一指针及 accepted/agent_iter_XXXX.pt 不可变快照 |
| candidate 与 accepted 比赛准入 | gating.ModelGate，同步调用现有 C++ play_model_match |
| accepted 更新后切换数据生产模型 | 下一轮开始 resolve；Python/C++ 后端均只使用 accepted |
| 被拒模型不能成为数据生产模型 | 保留完整训练候选，不提升 accepted，不回滚训练，不重测 |

配置默认 `gate_interval=1`，即每轮训练后都做一次准入；`--gate-interval 0`（bat 的
`GATING=0`，或 `debug_train_config()` 预设）关闭时完全沿用旧的训练网络自对弈路径，
不创建 accepted 或 gate 日志，也不增加迭代记录字段。开启后首次在恢复 trainer 后
bootstrap accepted；
每轮先用 accepted 自对弈，再训练和导出候选，到全局间隔则评测，之后照常周期对战、
evaluation 与保存。marked.pt 不参与准入，也不替代 accepted。

计分使用候选 participant 0 的每局 utility：胜 +1、负 −1、和 0，截断不计分也不计有效局数。
`finished>0`、达到 gate_min_finished 且 mean(score)>=gate_min_average_score 才提升；
等于门槛通过。
棋盘数量上限触发的 material 是正常终局，保留连续 tanh 得分而不是压成 1/0。
gate 默认 100 局、200 simulations、20 workers、50 棋盘上限、CUDA、温度 0.1、
平均分门槛 0.05、至少 50 有效局。max_moves=0 关闭动作截断。固定样本门槛不提供
统计显著性保证。

持久化先保存每个实际参赛候选的完整训练 checkpoint，再写 accepted 快照并原子更新
state；成功或失败都保留候选。若在提升后正常训练保存前中断，可以恢复对应完整候选，
而不是把只含权重的 accepted 快照当作完整 optimizer 恢复点。state iteration 领先恢复
checkpoint、变体/网络配置不匹配、快照损坏或发布失败均中止，不自动改用训练候选。
state 缺失才 bootstrap；同名历史 accepted 快照内容不一致时拒绝覆盖，应使用新 run。

每次实际评测追加 `<log_dir>/gates/iter_XXXX_<uuid>.json`，含完整对局及判定摘要。
迭代 JSONL 仅开启时增加 selfplay_model_iteration 和 gate，checkpoint 的 accepted 摘要
只作诊断。对局执行/导出异常记录 match_error 后继续训练；有效局不足 fail-closed；
未训练或未到间隔只记录 skip。所有 gate 加载/导出隔离 Torch CPU RNG。

与上游的刻意差异：

- 本实现阻塞训练循环执行 gate，不是异步五进程架构，也没有独立守护进程或在线热切换。
- 不扫描模型目录排队或按文件时间挑选 accepted；state.json 是唯一权威来源。
- 不移动候选到独立 rejected 目录，不删除失败候选，也不自动重测被拒模型。
- 不复刻上游在胜负已确定时提前停止剩余对局的机制；按配置运行固定预算。
- 5D 棋 material 终局/截断语义不同于围棋强制终局计分，不能机械搬用其统计口径。

## 8. 许可证注意事项

- KataGo 搜索模块属于其 MIT 许可内容（Copyright 2025 David J Wu 等）。
- 本项目**没有**复制 KataGo 的 `cpp/external/*`（CUDA、OpenCL、cutlass、
  nlohmann_json 等），这些库有各自许可证。
- 如果后续直接引入 KataGo 源文件或外部库，必须同时保留对应文件/库的许可证，
  不能只依赖本项目的 `THIRD_PARTY_NOTICES.md` 中的 MIT 文本。
