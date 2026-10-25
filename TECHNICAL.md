# TECHNICAL — SuZero AI 5D 国际象棋 AlphaZero

本文件记录当前实现状态、关键不变量与变更历史。算法与上游对应关系详见
[docs/katago-mapping.md](docs/katago-mapping.md)。

## 当前状态（2026-10-25）

- 版本：0.6.0
- 语言/标准：核心 C++17，仅标准库；5D 适配层 C++23（上游引擎要求）
- 搜索：单线程、**图搜索（DAG，置换合并）**、同步评估
- 自对弈：**进程内多对局线程 + 共享批量推理**（对齐 KataGo
  `numGameThreads` / `NNEvaluator` 模型）：`BatchedEvaluator`（通用请求队列 +
  server 线程 + `IBatchBackend`）与 `SelfPlayRunner`（对局线程池）
- 接入状态：项目主线为 **5D 国际象棋**；C++ 核心保持游戏无关的接口设计，
  5D 适配层（`suzero_5dchess`）连接 5dchess_engine、ONNX 网络与 PUCT
- 抽象接口：`IGameState`、`IEvaluator`、`IBatchBackend`、`IStopCriterion`
- 训练（Python）：游戏无关抽象层 `Trainer` / `TrainingDataSource` /
  `TrainingLoop` / `MetricsSink`，以及 PyTorch 后端 `TorchTrainer` /
  `InMemoryDataSource` / `IterableDataSource`；与 C++ 目标完全解耦
- 5D 训练：`suzero_5dchess` 在通用训练核心之上实现自对弈 AlphaZero
  （pybind11 绑定 + PyTorch 网络 + PUCT + 回放 + ONNX 导出 + 评估）；
  `d_model=128` 时参数 192,058，CLI 默认变体 `very_small`；8×8 变体
  `standard_turn_zero` / `standard`（270,202 参数，Turn Zero 训练预算 300,000）；
  `run_train.bat` 默认变体为 `standard_turn_zero`
- 异构模型对战使用独立的棋盘上限 `model_match_max_boards`，默认 `25`；达到上限立即按物质分结束。
- 构建：CMake + Visual Studio 2022 通过（MSVC 19.44，Release）
- 测试：`suzero_mcts_tests` 1213 项 + `suzero_5dchess_tests` 664 项，
  全部通过；Python 共 57 项 pytest（抽象层、PyTorch 训练器与 5D 训练闭环）

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
8. `Evaluation::policy.size() == IGameState::actionCount()`（当前状态的动作数），
   否则抛异常。
9. `actionCount()` 可以随状态变化，但必须是状态的确定函数：`stateKey()` 相等
   的状态必须返回相同动作数，并以相同顺序枚举相同动作。动作 id 是状态局部的。
   节点的先验由该节点自己的状态构建，而不是搜索实例共享的固定长度数组。
10. 先验列表包含所有合法且策略值非负的动作（含 0），按动作 id 升序，保证
    确定性；负值或 NaN 表示不可用。零概率动作可被搜索展开，探索项为 0。
11. 根节点噪声与策略温度只作用于 `root_priors_`，不修改图中节点的原始先验。
12. 同一个 `PuctSearch` 实例不支持并发调用。
13. 关闭 `graphSearchCycleCheckEnabled` 后不再维护 playout 路径集合；这只对严格
    无环的图安全，有环时只能依赖 `maxDepth` 或调用栈上限终止。
14. 5D 适配层中 `currentPlayer()` 在小步之间不变，只有 `SUBMIT` 换方；终局效用
    仍是终局状态当前行棋方视角。
15. `BatchedEvaluator::evaluate(state)` 阻塞到该请求被服务完才返回，调用期间
    `state` 必须保持有效；它可被任意多线程并发调用，内部状态由队列锁保护。
16. 每个 server 线程独占一个 `IBatchBackend`，因此 backend 自身不需要加锁。
    batch 只取“当前队列里已有的”请求，从不等待凑满（KataGo `waitPopUpToN`
    语义）；并发度来自对局线程数，而不是人为延迟。
17. backend 抛异常时，同一 batch 的所有请求都收到该异常；单个请求出错不会
    杀死 server 线程。
18. `SelfPlayRunner` 的每局随机种子只由 `(seed, gameIndex)` 经 SplitMix64
    推导，与线程调度顺序无关；样本 `value` 以 `sample.player` 为视角，
    按终局 `currentPlayer()` 的效用取正负号。

## 实现要点

- `allocateOrFindNode`：按 `stateKey()` 查表，命中复用、未命中新建；节点由表拥有。
- `playout`：展开叶子 → 选择 → 建边/查找置换 → 边访问追赶 → 路径成环检测 →
  下探 → 更新 edge visit → 重算节点统计。
- `maybeCatchUpEdgeVisits`：`edgeVisits < childVisits` 时只补边访问而不下探；
  `graphSearchCatchUpLeakProb` 控制偶尔仍然下探。
- `PathSet`：每次 playout 的节点集合；再次遇到即判定成环，只计边访问。
  `graphSearchCycleCheckEnabled = false` 时不创建、不清空该集合，playout 中也
  不做插入/查找。
- `selectPrior`：PUCT + FPU + utility stdev 缩放。先比较已展开子节点，再在
  未展开动作中只取策略概率最高者参与比较；平局保留较前先验。
- `recomputeNodeStats`：child weight 聚合、根节点 subtract/prune、
  `valueWeightExponent` 坏子降权（3 自由度 Student-t CDF），归一化回剪枝前
  总权重，最后加入自身评估权重 1。
- `computeRootPlaySelectionValues`：KataGo 非 LCB 的根节点最终权重：最佳子
  节点启发式、其余子节点回顾式降权并 `ceil`、subtract/prune。
- `addRootDirichletNoise`：KataGo 策略形状 alpha（`min(0.01, p)` 上限）的
  Dirichlet 噪声。
- `applyRootPolicyTemperature` / `chooseAction()`：对数域温度缩放，温度按
  `chosenMoveTemperatureHalflife` 在 early 与 final 值之间插值。
- `sampleWithTemperature`：保留 `onlyBelowProb` 阈值语义。
- `advanceRoot`：复制子图、重建节点表、丢弃旧表；只为仍被边引用的节点保留
  副本（被提升节点的旧副本被删除），新根是独立的、不是任何边目标的副本，
  保证根不在环上。找不到已展开边时返回 false 且不改变状态。复制后旧节点
  指针全部失效。

## 批量推理与自对弈

对应 KataGo 的 `NNEvaluator` + `selfplay` 命令的进程内部分（多对局线程共享
一个批量评估器），不涉及跨进程训练循环。

- **分层**（`suzero::mcts`，游戏无关）：
  - `IBatchBackend`：`maxBatchSize()` + `evaluateBatch(states, results)`，
    由具体模型实现，负责把一批**异构**状态打包成一次推理。
  - `BatchedEvaluator`：实现 `IEvaluator`。内部是请求队列 + `numServerThreads`
    个 server 线程；每个线程持有一个由 `BackendFactory` 创建的 backend。
    `evaluate()` 入队后阻塞在自己的条件变量上，server 线程批量服务后唤醒。
  - `SelfPlayRunner`：`numGameThreads` 个对局线程，每局一个独占的
    `PuctSearch`（单线程搜索，`numSearchThreads = 1` 的等价物），共享同一个
    `IEvaluator`。支持一次性收集（`run()`）与流式回调（`run(callback)`），
    以及 `requestStop()` 协作停止。
- **队列语义**：
  - 入队受 `maxQueuedRequests` 背压限制（默认 `maxBatchSize * 4 *
    numServerThreads`），防止对局线程无限堆积请求。
  - server 线程被唤醒后取 `min(queue.size(), maxBatchSize)` 行立即推理，
    不额外等待；批次大小由并发对局数自然决定。
  - 析构时置 `closed`，已入队请求仍会被服务；析构等待 `active_clients == 0`
    后再 join，避免调用方持有的栈上请求失效。
  - `maxBatchSize` 会被最小的 backend 上限钳制；backend 工厂返回空或上限
    非正时在构造期同步抛错。
- **对局循环**（`SelfPlayRunner::playGame`）：
  1. 终局 → `terminal`；自对弈不设置动作数上限，正常对局只由游戏自身的棋盘
     上限或其他自然终局结束；收到外部停止请求才停止未完成对局。
  2. `actionCount() == 1`（5D 象棋的强制 `SUBMIT`）→ 直接应用，不搜索、
     不产生样本。
  3. 否则 `search.run()`，用 `denseVisitDistribution` 记录密集访问分布样本
     （仅在存在正访问时），再按温度选步。
  4. 选步后优先 `advanceRoot` 复用子树；失败则 clone + apply + `resetRoot`。
  5. 终局后把 `terminalUtility()` 按“样本行棋方 vs 终局行棋方”取符号填回
     每个样本的 `value`（两人零和约定）；截断局的价值为 0。
- **确定性**：每局种子 `seedForGame()` = `seed + golden * (gameIndex+1)` 再经
  SplitMix64 混合，因此同一 `seed` 下的结果与 `numGameThreads`、线程调度
  无关（测试 `testSelfPlayRunner` 用两次运行比对状态键与策略）。
- **错误处理**：任一局抛异常会记录首个异常、置停止标志并等待其余线程退出，
  最后在调用线程重抛；不会产生半收集的结果。

## 5D 象棋适配层

- **分层**：`Chess5DGameState`、`OnnxEvaluator`、`Chess5DAgent` 的公开
  头文件只依赖 C++17 与搜索核心；上游 `state`/`board` 类型藏在 pimpl 中。
  适配层目标本身按 C++23 编译。
- **动作生成**：先处理 `get_timeline_status()` 的 mandatory 时间线，再处理
  optional；对每条线取时间线末端的当前行棋方棋子，调用
  `gen_piece_move`，用 8 元 MoveKey 去重。mandatory 为空时把 `SUBMIT`
  追加到末尾，因此候选顺序与 5dc_ml 模型协议一致。
- **终局**：
  - 吃王：落子前用 `get_piece(to, mover_color)` 检查目标格的王系棋子；
    终局立即生效，白方结果 = 黑方吃王 ? -1 : +1。
  - 物质：初始化或任一动作后只要 `board_count >= board_limit` 就立即终局，结果为
    `tanh((white - black) / (total + 1e-8) * material_scale)`；若同一个小步既吃王
    又达到棋盘上限，`capture_king` 优先于 `material`。
  - 无合法动作：候选为空（mandatory 非空且没有可走棋子）时当前行棋方判负；
    该判定在惰性生成候选时完成。
- **状态键**：配置 + done/outcome/terminal reason + 精确引擎身份（棋盘尺寸、
  present、promotion、所有时间线/回合的 12 组位棋盘）。不含 PGN 注释，
  但覆盖所有影响合法性、终局与评估的内容。
- **编码**：12 个棋子通道（白 K/Q/R/B/N/P → 0–5，黑 → 6–11），
  平面下标 `i*12*16 + channel*16 + x + y*4`，`l + line_shift` 与 `t` 坐标。
- **ONNX**：校验 sidecar 协议字段与会话 IO 签名；`q == 0` 时不调用网络，
  直接返回单动作策略；否则策略为
  `softmax(clamp(other_logits,-20,20) ++ submit_logit)`，价值为当前行棋方视角。
  协议解析、张量打包与 softmax 集中在私有 `OnnxModel`；`OnnxEvaluator`
  （单条）与 `OnnxBatchBackend`（批量）都只是它的薄包装，避免两套实现漂移。
- **批量推理**：`OnnxBatchBackend` 实现 `IBatchBackend`，每个 server 线程
  一个独立 ONNX 会话。批内局面可以有不同棋盘数与候选数：棋盘维 padding 到
  `max_boards` 并用 `used_board_counts` 屏蔽填充板，动作维 padding 到
  `max_q`（零 MoveKey，组装结果时丢弃多余 logit）。强制 `SUBMIT`（`q == 0`）
  的行不进入设备 batch，直接填结果。`makeBatchedOnnxEvaluator()` 把
  `OnnxEvaluatorConfig` + `BatchEvaluatorConfig` 组装成 `BatchedEvaluator`。
- **Agent**：参考 C++ runner 默认 `maxPlayouts=200`、`cpuct=2.0`、温度
  `1.0 → 0.1`（步数阈值 30）；只有 SUBMIT 时强制返回且不搜索；选步后优先
  `advanceRoot` 复用子树，失败或终局则按需 `resetRoot`。
- **参考对齐验证**：`suzero_5dchess_tests --probe <plies>` 输出指定小步数
  后的动作键、策略概率、价值与 urgency。用 5dc_ml 的
  `alphazero.smallstep_state.SmallStepState` 编码同一局面并调用同一 ONNX
  模型，ply 0–20（含强制 SUBMIT 的奇数 ply）的候选顺序、动作键、概率和
  价值全部一致，最大绝对误差 < 5e-8（fp32 舍入）。

## 训练子系统（Python）

- **位置**：`python/`，独立包 `suzero_train`，不参与 CMake 构建，也不改变
  C++ 核心或 5D 适配层的依赖。
- **分层**：
  - 抽象层（`types.py`、`data_source.py`、`trainer.py`、`loop.py`、
    `metrics.py`）只定义契约与调度；不 `import torch`，不依赖具体游戏。
  - `suzero_train.pytorch` 是具体实现：`TorchTrainer` 接收任意
    `torch.nn.Module` 与 `objective(model, data) -> (loss, metrics)`；
    `InMemoryDataSource` / `IterableDataSource` 负责数据接入。
- **抽象契约**：
  1. `TrainingBatch.data` 对循环不透明，循环只读取 `sample_count`。
  2. `TrainingDataSource` 以 `begin_epoch()` / `next_batch()` 表示一次完整
     epoch；`validation_batches()` 是独立的一次性迭代器。
  3. `Trainer.train_step()` 必须完成一次 optimizer step；`samples_seen` 与
     `global_step` 必须能跨 checkpoint 恢复。
  4. `LoopConfig.max_steps` 只统计本次 `run()` 执行的步数，不包含续训前
     已完成的步数；`epochs <= 0` 时必须同时给出 `max_steps`。
- **从 KataGo 提取的训练方法**：
  - **per-sample LR**：`base_learning_rate * learning_rate_scale`，AdamW
    再乘 `1.33`，最后乘 warmup、`(batch / batch_reference) ** lr_exponent`
    （AdamW 为 0.5，SGD 为 0.0，即 SGD 的 LR 不按 batch 缩放）与参数组
    `lr_scale`。weight decay 的 batch 指数单独配置（SGD 为 1.0）。
  - **warmup**：KataGo 前 2,000,000 样本的 8 段曲线（1/20 … 1/1.4）按
    `warmup_samples` 等比缩放；`0` 关闭。
  - **参数组**：`ParamGroupSpec` 提供 `lr_scale` 与 `weight_decay_scale`，
    并要求覆盖全部可训练参数，否则报错，避免静默漏训。
  - **梯度裁剪**：`grad_norm_cap * grad_norm_clip_scale *
    sqrt(batch / 256) / sqrt(lr_scale)`；cap 为 0 时只测量并记录
    `gnorm_batch`，不裁剪（不会以 `max_norm=inf` 调用裁剪，避免 NaN）。
  - **AMP**：`none` / `bf16` / `fp16`，fp16 使用 `GradScaler`；可传
    `forward=` 让 autocast 只覆盖前向，loss 在 fp32 计算。
  - **loss 约定**：反向 loss 默认按 batch 总和（`loss_reduction="sum"`）；
    objective 未返回 `loss_sum` 时，训练器据此补出逐样本 loss 指标。
  - **指标**：`_sum` 按样本平均、`_batch` 按 batch 平均，其余按 running sum
    累计（与上游 `accumulate_metrics` 一致）；运行指标按 `metrics_decay`
    （默认 0.995）衰减，`_batch` 窗口按 `batch_metric_window_steps`
    （默认 100 step）收缩，与日志频率无关。
  - **checkpoint**：原子写入 model / optimizer / scaler / running_metrics /
    train_state / last_val_metrics / trainer_config，可续训。
- **未移植**：KataGo 的围棋模型、Board 编码、`.npz` 自对弈数据格式与导出
  流程；DDP/多 GPU、Muon/NorMuon/Aurora、SWA、lookahead、`torch.compile`、
  LR cycle。KataGo 按 normalization 类型选择 weight decay 的表是模型相关的，
  改为 `weight_decay` + 参数组系数显式配置。

### 5D 象棋 AlphaZero 流水线

- **变体**：`protocol.VARIANTS` 注册 `very_small`（4×4）、`standard`（8×8）和
  `standard_turn_zero`（8×8，PGN `Standard - Turn Zero`）。TurnZero 开局由原生
  引擎生成 T0 黑方与 T1 白方两张棋盘，编码包含 `t_coords=[0,1]`，不由 Python
  人工补历史。变体的 PGN 与几何统一传给环境、两类自对弈后端、ONNX sidecar
  和模型对战；不改引擎或搜索核心。
- **TurnZero 入口与隔离**：`run_train_turnzero.bat` 薄包装共享 `run_train.bat`，
  固定位置参数 `standard_turn_zero`，其余参数原样转发。训练 CLI 为该变体选择
  300,000 参数预算，显式网络参数优先；Python API 仍通过
  `NetworkConfig(max_params=300_000)` 显式放宽预算。正式产物在
  `runs/standard_turn_zero/cls_action_mlp/`；非 `very_small` 的默认调试路径为
  `runs/debug/<variant>/`，共享启动器的自动续训查找同步使用这一规则。
  原有跨变体 checkpoint 校验不变，也不自动添加 Very Small 固定对手。

- **分层**：`python_binding.cpp` 只做 pybind11 对象转换；`env.py` 缓存
  候选动作并区分强制 SUBMIT；`mcts.py` 与 `self_play.py` 只依赖
  `GameEnv` 与网络；`pipeline.py` 负责训练调度与持久化；`cli.py` 只解析
  参数。C++ 搜索核心不感知 Python。
- **动作 ABI**：`actions` 只含非 SUBMIT 的 8 元 MoveKey，顺序与
  `other_logits` 一致；SUBMIT 单独使用 `submit_logit`。`collate_samples`
  把 SUBMIT 对齐到 padding 后的最后一列（`qmax`），并用 `policy_mask`
  屏蔽 padding。不同动作数的局面因此可共享同一 batch。
- **网络**：`d_model=128`、`n_heads=4`、`n_layers=1`、`d_ff=256`、
  `action_mlp_hidden=64`。token = 棋盘投影 + urgency 特征 + (l, t)
  正弦编码；CLS 内容为 `side_to_move`。动作特征为源/目标位置编码与
  源/目标格编码，共 416 维。参数量 192,058，构造时校验不超过
  `max_params=200_000`。
- **自对弈**：每个根节点加 Dirichlet 噪声（全搜索步）；温度在
  `temperature_threshold` 后降到终值；playout-cap randomization 的 fast
  步既不产生样本也不加噪声；只有 SUBMIT 的强制局面直接推进，不调用网络。
  不提供 `max_game_length` 配置或 CLI 参数；每局只由棋盘上限或吃王、无合法动作
  等自然终局结束。任一动作触及棋盘上限都会立即产生物质分终局，吃王保持优先。
  终局价值按当前行棋方视角分配，支持吃王 ±1、物质分 tanh、无合法动作和中止。
- **C++ 自对弈后端（`cpp_onnx`）**：每轮先把当前 PyTorch 网络导出为
  `runs/<variant>/<arch>/selfplay/model_iter_XXXX.onnx`（+ sidecar），再调用
  `suzero::chess5d::runSelfPlay`：`numGameThreads` 个 C++ 对局线程共享一个
  批量 ONNX 评估器（`numServerThreads` 个 server 线程），规则、搜索与推理
  全部在 C++ 内完成，只有最终样本以 numpy 数组回到 Python。`runSelfPlay`
  可返回 `SelfPlayStats`（batch 数、行数、最大 batch、总耗时与建会话耗时），
  Python 侧据此在自对弈日志中打印 samples/s、evals/s、avg_batch 与
  max_batch，用于判断瓶颈在批量不足还是 host 侧编码/解码。因为 ONNX
  Runtime 可能仍持有已加载的文件句柄，每轮使用独立文件名，并只保留最近
  2 份导出。`provider` 为 `cpu` / `cuda`；选择 CUDA 时通过
  `Ort::GetAvailableProviders()` 检查，不可用即报错，绝不静默回退 CPU。
  逐局语义与 `self_play.py::play_game` 对齐（按局采样棋盘上限、playout-cap
  randomization、仅全搜索步产生样本并加噪声、按 semimove 切换温度、强制
  SUBMIT 不计样本、终局原因与 semimove/turn 计数一致）。
- **协作式 Ctrl+C**：`interrupts.py` 在 `run_training` 期间接管 `SIGINT`。
  第一次按下只置位停止标志；当前这一轮（自对弈、训练、评估）照常完成后，
  循环顶部检测到标志即跳出，走正常的最终 checkpoint 保存并返回
  `stopped_early=True`，退出码 0。第二次按下恢复原处理器并抛出
  `KeyboardInterrupt`，CLI 捕获后返回 130，允许用户放弃进行中的一轮。
  C++ 自对弈调用期间 GIL 已释放，Python 信号处理器会等到该调用返回后才
  执行，因此“打完这一轮”的语义在两种后端下一致。为了让这次等待可见，
  `run_selfplay` 的阻塞区间由 `ConsoleInterruptGuard` 覆盖：第一次按下在
  毫秒级直接把提示写到 stderr（控制台走 `WriteConsoleW`，重定向走 UTF-8
  `WriteFile`），然后返回 FALSE 让 CPython 照常记录 SIGINT；第二次按下
  直接以 130 终止进程，与 Python 侧的第二次按下语义一致（见
  `src/5dchess/console_interrupt.h`）。
- **连续训练**：`--continuous` 让训练器按 `num_iterations` 划分为批次并
  无限运行：每批结束时 `latest.pt` 已发布，下一批以它续训；协作式 Ctrl+C
  使当前批的 summary 带 `stopped_early=True`，循环随即结束而不是再开一批。
  `run_train.bat` 的 `CONTINUOUS` 开关映射到该参数。
- **MCTS**：PUCT，`q_init=-1`，根节点按访问次数输出原始策略；不做置换
  合并（图搜索在 C++ 侧，Python 训练侧保持简单树）。`workers > 1` 时用
  线程池并发多盘，每盘独立环境、RNG 与搜索树；网络处于 `eval()`，推理
  线程安全，结果按游戏序号稳定返回。
- **训练**：均匀回放、AdamW、余弦学习率、梯度范数裁剪；
  `AlphaZeroObjective` 把整个 batch 移到模型设备，策略 loss 在
  `clamp` 后的 logits 上做掩码交叉熵，价值为 MSE。batch 默认按总和
  （`loss_sum`）报告，与 `TorchTrainer` 的约定一致。
- **评估**：第一次到达评估间隔时把当前网络写成 `marked.pt` 基线（不进行
  对局），之后每个间隔与基线交替先后手对弈，并把新网络发布为下一轮基线；
  显式 `reference_checkpoint` 缺失时只告警并跳过，不会覆盖用户指定的基线。
- **checkpoint**：`format=suzero_5dchess.train.v1`，包含模型、
  优化器、计数器、网络几何、训练配置、评估状态与 torch RNG 状态；回放缓冲区
  独立保存在同目录的 `replay_buffer.pt`，不进入权重文件。checkpoint 原子写入
  `agent_iter_XXXX.pt`，并原子发布 `latest.pt`、`agent_final.pt`。续训校验
  变体与网络几何，并恢复 CPU/CUDA RNG；缓冲区文件缺失时告警并以空缓冲区继续。
  自对弈记录可通过 `record_games` / `--record-games` 写为每轮
  `games_iter_XXXX.json`，并逐局生成 `games/<terminal_reason>/*.gn5d`。GN5D
  写入由训练启动的常驻 spawn 子进程异步执行：主进程只提交不含训练样本的轻量
  批次，子进程逐局原生重放校验和独占写入、每批统一裁剪，因而与后续训练、对战及
  下一轮自对弈重叠；队列有界并回传异常，正常结束前排空，异常退出时快速终止。
  Python 与 C++（含多进程）后端均记录实际执行的完整动作流；训练样本只保留
  full-search 位置不影响存档。`game_record.py` 默认写出 `gn5d` v2：只保留单个
  `initial` FEN、八维整数数组或 `"submit"` 行动、结果和五个元信息字段，使用紧凑
  JSON；宽高、快照、当前棋盘、时间线状态、步数与最终 FEN 均由原生引擎重放推导。
  加载时校验合法性、结果与终局，旧 `5dc.capture-king` v1 / `.5dck` 缓存棋谱仍可读。
  UUID 文件名独占创建，失败只清理本次生成的部分文件，
  不覆盖已有棋谱；每个终局原因目录最多保留按修改时间排序的最新 5000 盘。
  训练期定期模型对战把 C++ 逻辑动作重放为同一 GN5D 格式，逐局写入
  `matches/games/<terminal_reason>/`，并在系列汇总的 `gn5d_records` 中记录路径。
- **ONNX**：opset 18、legacy exporter，7 输入 / 3 输出；B、N、q 均为
  动态轴。导出后写 sidecar（协议、几何、架构、参数量），可选地用
  ONNX Runtime 对比 PyTorch 输出。调试导出的模型已由 C++ 评估器加载并
  通过全部 410 项 5D 检查。

## 本地引擎与 GN5D 原生 UI（2026-10-24）

- CMake 默认引擎根目录改为项目内 `5dchess_engine-main/`。已有构建目录需显式
  `-DSUZERO_5DCHESS_ENGINE_ROOT=./5dchess_engine-main` 覆盖旧缓存；本次已用该源码树
  重新构建静态库及 Python 扩展，不修改引擎核心、搜索或训练代码。
- 删除根目录 `ui/index.html`、`ui/app.js`、`ui/style.css` 独立查看器。
  `ui_server.py` 直接服务引擎自己的 `ui/` 资源，包括 Canvas、SVG、字体、图标和主题。
  仅对 HTTP 返回的原版 index 替换 worker 地址，切入 `gn5d-worker.js`；磁盘上的原版
  index 和 WASM worker 保留，因此独立 WASM 构建仍可用于普通 PGN 分析。
- 新增 `gn5d-replay.js` 把原生帧映射到原版 draw/main 所用的 view 消息，包含历史棋盘、
  present、focus 及按真实行棋方着色的移动箭头。没有第二套棋盘渲染器，也不在 JS 实现棋规。
  `/api/replay` 继续调用 `game_record.replay_record` 校验并恢复全部帧；`/api/initial`
  提供由原生规则生成的 Turn Zero 起始局面，读谱入口不依赖 WASM 编译产物。
- 适配原生导入的 load/result/commit/cancel 事务，取消或失败不替换当前棋谱；
  单步导航包含 SUBMIT，大步导航按真实 SUBMIT 边界，保留吃王前的未提交尾部。
  导出完整原始 JSON，并在桌面/移动布局保留 `.gn5d` / `.5dck` 后缀。
- 原版 UI 的 main/ui/startup 仅添加传输提示、只读状态、扩展名及导入时原生聚焦适配。
  只读模式禁用编辑、分支删除、Hint 与 SUBMIT；缩放、平移、翻转及主题继续使用原版实现。
  普通 PGN 不由该本地回放入口解析，需独立 WASM 分析器。
- 静态服务仅允许引擎 UI 的脚本、样式、资源、字体、图标及 WASM 目录，校验真实路径
  防止目录穿越和符号链接逃逸；保留 Host/Origin、请求体与累计帧大小限制。
- 回归：`python/tests/test_ui_server.py` 覆盖原版页面/资源路由、原生开局、GN5D/旧格式、
  非法棋谱与本机访问限制。`python/tests/engine_ui_smoke.mjs` 使用本机 Chrome/Edge CDP
  验证真实页面启动、导入、逐帧棋盘/present 对照、单步/回合导航、原文导出和失败保留。
  运行方式：`node python/tests/engine_ui_smoke.mjs URL OUTPUT_DIR RECORD.gn5d RECORD.5dck`；
  需 Node 22+，可用 `BROWSER` 指定浏览器可执行文件。
- 本次验收：C++ CTest 4/4、读谱/存档相关 Python 测试 54/54 通过；真实浏览器完成
  4 动作 GN5D、4 动作旧 `.5dck`、21 动作真实吃王棋谱的逐帧对照及桌面/移动导出。
  Python 全量回归为 284 通过、1 失败：既有训练启动器测试硬编码要求
  `SELFPLAY_PROCESSES=1`，当前使用者配置为 `4`；未为此改动训练脚本。

## 已知限制与后续方向

1. 单局内多线程搜索 + virtual loss + 原子统计（当前并行度来自多对局）。
2. 评估缓存及其与置换节点的交互。
3. 异步评估器接口（当前 `evaluate()` 仍是阻塞调用，批量只发生在 server
   线程内部）。
4. 更多最终着法选择策略（LCB 等）。
5. 可选的 debug/统计输出接口。
6. 更紧凑的状态键（当前为 `std::string`，键构造成本由游戏承担）。
7. 5D 批量推理已支持，但每个 server 线程持有一份独立 ONNX 会话；模型热切换
   与训练期数据写出（KataGo 的 `SelfplayManager`）尚未实现。
8. 参考 runner 的 Dirichlet 噪声是对称分布（`alpha=0.3`），本库的根噪声是
   KataGo 策略形状分布，两者只能近似对应；`Chess5DAgent` 默认关闭根噪声，
   自对弈训练时可自行开启并调整 `rootDirichletNoiseTotalConcentration`。
9. Python 训练器只支持单进程；DDP/多 GPU、Muon、SWA、`torch.compile`、
   LR cycle 尚未移植。`python` 后端的 `workers` 是 Python 线程，共享同一
   网络，不跨对局批量推理；`cpp_onnx` 后端用 C++ 对局线程 + 批量评估器，
   可跨对局合批，但整个批次仍只用一个执行设备，且 `record_games` 只能
   记录样本、没有逐着法历史（C++ 桥不生成 `moves`）。
10. 5D 回放缓冲区保存在内存中，并以逐样本 zlib 记录独立序列化为
    `replay_buffer.pt`（不进入 checkpoint）；尚不支持 `.npz` 等外部自对弈
    数据格式。
11. Python 侧 MCTS 不做置换合并、评估缓存或批量叶子评估；大规模训练时
    这些是最先需要补齐的性能项。

## 变更记录

### 2026-10-25 · 文档重新定位为 5D 国际象棋 AI

- README 从“游戏无关 PUCT 图搜索核心”改为 5D 国际象棋 AlphaZero 项目说明：
  规则变体与模型规模前置，训练/对战/棋谱/回放按使用流程重组，并同步
  `run_train.bat` 当前默认值（`standard_turn_zero`、100 workers、4 进程、
  1000 局、棋盘上限 55–90、连续训练开启）。
- 在 README 与 LICENSE 开头明确标注搜索核心大量借鉴并派生自 KataGo（MIT），
  训练调度重新实现其 `python/train.py` 方法；LICENSE 增补第三方组件清单。
- 同步更新 `THIRD_PARTY_NOTICES.md`、`docs/katago-mapping.md` 与
  `run_train.bat` 中过时的旧项目名和不一致的定位表述。
- 全仓 42 个 C++ 源文件的 SPDX 版权行统一为 `SuZero-5DChess contributors`，
  文件头统一描述 SuZero AI 5D 国际象棋项目并保留 KataGo 派生声明；
  CMake 与 Python 包描述同步。

### 2026-10-25 · 构建目录可重定位，配置入口脚本化

- 问题：`build/` 内的 `CMakeCache.txt`、`.vcxproj`、`.tlog` 等生成物嵌入了
  绝对源码路径（迁移前的旧目录），全树共 483 个
  生成文件含旧路径。目录改名后 CMake 会因 `CMAKE_HOME_DIRECTORY` 不一致
  拒绝复用缓存，MSBuild 的 `OutDir` 与包含目录也全部失效。
- 源码本身没有绝对路径：`CMakeLists.txt` 使用 `CMAKE_CURRENT_SOURCE_DIR`，
  运行脚本使用 `%~dp0`。因此不修改生成物，改为在最终路径重新生成构建树。
- 新增 `configure.bat`：以 `%~dp0` 推导源码目录；`SUZERO_ONNXRUNTIME_ROOT`
  来自环境变量或 `extern/onnxruntime-win-x64-1.24.3`；检测到缓存属于其他
  源码路径时自动追加 `cmake --fresh`。不再向缓存写入
  `SUZERO_5DCHESS_ENGINE_ROOT`，由全新配置自动选中项目内引擎。
- 删除迁移前的 `build/`（608 MB）。已构建的 Python 扩展
  `python/suzero_5dchess/_5dchess.cp311-win_amd64.pyd` 不受影响。

### 2026-10-25 · ONNX Runtime DLL 按需加载，构建树不再重复副本

- 问题：`build/` 达 3.5 GB，其中 2.04 GB 是 `onnxruntime_providers_cuda.dll`
  （299 MB）在 Debug、Release、`crash_asan`、`verify`、`opt13_verify`、
  备份目录和 Python 包目录中的 7 份副本；PDB/OBJ/ILK 另占约 0.8 GB。
- CMake：删除 `suzero_copy_ort_runtime`。Python 扩展不再向
  `python/suzero_5dchess/` 复制任何 DLL；C++ 测试输出目录通过
  `cmake/link_runtime_files.cmake` 以**硬链接**（同一卷时零额外空间，
  失败时回退为复制）放置 `onnxruntime.dll` 与
  `onnxruntime_providers_shared.dll`。新增 `SUZERO_ORT_STAGE_CUDA`
  （默认 `OFF`）用于需要自包含 CUDA 的测试目录。
- Python：新增 `suzero_5dchess/_ort_runtime.py`，包初始化时先调用
  `os.add_dll_directory`，优先使用环境变量 `SUZERO_ORT_RUNTIME_DIR`，
  否则使用已安装 `onnxruntime` 包的 `capi` 目录；provider DLL 与
  `onnxruntime.dll` 同目录，ONNX Runtime 可自行加载。
- `run_train.bat` 的 CUDA 检查改为通过 Python 查询已安装包的 `capi`
  目录，不再检查扩展旁的 DLL。
- 效果：更新后的构建树中，每个配置只保留约 15 MB 的 `onnxruntime.dll`
  硬链接；CUDA provider 全树仅 pip 包中的一份。清理旧构建树/备份后
  `build/` 可从 3.5 GB 降到 1 GB 以下。

### 2026-10-23 · Standard - Turn Zero 训练版本

- 在统一变体注册表加入 `STANDARD_TURN_ZERO` / `standard_turn_zero`，复用现有
  小步/SUBMIT 协议、PyTorch 网络、C++ 规则封装和训练循环，不修改外部游戏引擎。
- 保持 `d_model=128`、单层 Transformer 和 64 维动作 MLP；8×8 输入对应
  270,202 参数。训练 CLI 默认预算为 300,000，显式 `--max-params` 可覆盖，
  Very Small 的 200,000 默认预算不变。
- 新增 `run_train_turnzero.bat`，复用共享启动器的 CUDA/并发/连续训练配置、
  自动续训、ONNX 导出、棋谱记录和同变体历史对战。正式与调试目录分别独立，
  不误用 Very Small 的 checkpoint 或固定旧模型。
- 新增 7 项专项测试：原生 T0 历史及换方、所有 CLI 子命令与路径隔离、
  参数预算和显式覆盖、8×8 动态 ONNX 数值对齐，以及 Python / C++ 单进程 /
  C++ 双进程三种路径的训练、存档重放、导出、续训与跨变体拒绝，全部通过。
  集成测试每轮仅生成 2 局，使用 2 次模拟并固定 `min_board_limit = max_board_limit = 4`。
- 真实 CLI 验收：CUDA 训练 + CUDA C++ 双进程自对弈，每轮 2 局、2 simulations、
  固定棋盘上限 4；TurnZero 初始 2 张棋盘，因此会快速正常结束于 `material`。完成
  第 1 轮后从 `latest.pt` 续训到第 2 轮，两轮均实际执行优化器更新、保存 `.gn5d`
  并导出 ONNX。定期历史对战也用 2 局的小预算跑通；其显式正数小步上限只用于
  对战截断，不进入胜、负、和或胜率统计。Windows 脚本的帮助入口
  与变体参数转发已在原生 `cmd.exe` 中验证。
- 验收时本地旧扩展缺少当前对战接口的 `num_threads` 参数，已重新构建
  `suzero_5dchess_py` 并重测上述完整流程。全量 Python 回归为 281 通过、
  1 项既有静态启动器测试失败：测试写死 `SELFPLAY_PROCESSES=1`，而工作区原有
  可调配置是 4；本次未更改该配置或放宽该测试。

### 2026-10-23 · C++ 自对弈期间的 Ctrl+C 立即反馈

- 问题：`run_selfplay` 整批期间持有 GIL 的只有 C++ 线程，主线程阻塞在
  pybind 调用里，Python 的 `SIGINT` 处理器要等整批返回才能执行。进度条
  在刷新，但 Ctrl+C 看起来完全没反应，用户无法判断按键是否生效。
- 新增私有头文件 `src/5dchess/console_interrupt.h`：
  `ConsoleInterruptGuard` 只在 `run_selfplay` 的阻塞区间注册系统级处理器。
  第一次按下立即向 stderr 打印“已收到中断信号”（控制台用
  `WriteConsoleW`，重定向用 UTF-8 `WriteFile`），并把事件透传给 CPython，
  所以本批自对弈照常跑完、协作式停止语义不变；第二次按下直接
  `TerminateProcess(130)`，与已有的“再按一次立即中止”一致。POSIX 下用
  `sigaction` 并链回原处理器。
- `python_binding.cpp` 在 `py::gil_scoped_release` 之外包了一层该守卫，
  析构时在重新拿回 GIL 之后恢复原处理器。
- 验证：在真实控制台的子进程里于自对弈中途注入控制台事件（测试用
  `CTRL_BREAK`，与 Ctrl+C 走同一条控制台事件分派路径）：提示在按键的同一
  秒内打印，而 `run_selfplay` 继续跑完并返回全部 300 局；连按两次时进程以
  130 退出（约 6 秒，剩余约 20 秒的批次被放弃）。

### 2026-10-23 · 暴露自对弈批量统计

- `runSelfPlay()` 新增可选的 `SelfPlayStats*` 输出：来自 `BatchedEvaluator`
  的 requests/batches/rows/maxObservedBatch，以及总耗时与建会话耗时；
  异常时保持未写入。
- `_5dchess.run_selfplay()` 的返回值新增 `stats` 字典。
- `cpp_self_play` 在每轮自对弈后打印两条日志：games/samples/耗时（含
  setup），以及 samples/s、evals/s、batches、rows、avg_batch、
  max_batch（观测值/上限）与 server_threads，用于区分"批量太小"与
  "host 侧太慢"。
- 新增 C++ 断言：统计计数自洽（requests == rows、batches <= requests、
  最大 batch 不超过上限）且计时区间包含 setup。

### 2026-10-23 · 连续训练（`--continuous` / `run_train.bat` 的 `CONTINUOUS`）

- CLI 新增 `--continuous`；`cmd_train` 改为调用 `_run_training_batches`。
  每批恰好 `num_iterations` 轮，批与批之间复用现有的 checkpoint 恢复路径
  （`run_training(resume=latest.pt)`），因此回放缓冲区、评估基线、
  TensorBoard/jsonl、ONNX 导出和 LR 进度都按原有语义延续。
- 停止条件：某批 summary 的 `stopped_early` 为真（协作式 Ctrl+C）即结束
  循环；第二次 Ctrl+C 仍由 `cmd_train` 捕获并返回 130。若某批没有写出
  `latest.pt` 也会停止，避免无进展的忙循环。`num_iterations <= 0` 且开启
  连续模式时直接报错。
- `_merge_summaries` 把各批 summary 合并为一份：`iterations` 逐轮累加，
  累计计数（`end_iteration`、`total_games`、`total_samples`、`buffer_size`）
  取最新批次的值；合并结果每批结束后写入 `summary.json`，中途退出也不会
  丢失已完成批次的记录。
- `run_train.bat` 顶部新增 `CONTINUOUS=0` 开关（设为 1 即追加
  `--continuous`），也支持命令行直接追加 `--continuous`；启动横幅与结束
  信息会标注连续模式。连续模式下 `--iterations` 表示每批轮数。
- 测试：新增 `test_cli_accepts_continuous_flag`、
  `test_continuous_batches_resume_until_stopped`（3 批、第 3 批
  `stopped_early` 后停止，校验 resume 路径与合并后的 summary）和
  `test_continuous_requires_positive_iterations`。Python 66 项全部通过。

### 2026-10-23 · `cpp_onnx` 自对弈后端（CUDA）+ 协作式 Ctrl+C

- 实现此前只有选项、调用即抛 `NotImplementedError` 的 `cpp_onnx` 后端：
  新增 `include/suzero/5dchess/selfplay_bridge.h` /
  `src/5dchess/selfplay_bridge.cpp`，由 C++ 对局线程运行完整
  `PuctSearch` + 批量 ONNX 评估器，逐局语义对齐 `self_play.py`。
- `run_selfplay` pybind 入口：参数化模型、provider、线程数、模拟次数、
  playout-cap、温度与棋盘上限；返回 padding 后的 numpy 样本数组 + 每局
  元数据。C++ 调用期间释放 GIL，进度回调内重新获取。
- `OnnxEvaluatorConfig` 新增 `provider` / `cuda_device_id`；CUDA 通过
  `AppendExecutionProvider_CUDA` 启用，启动前用
  `Ort::GetAvailableProviders()` 校验，不可用直接报错。CMake 新增
  `SUZERO_ONNXRUNTIME_RUNTIME_DIR`，自动从 pip 的 `onnxruntime-gpu` 目录
  复制 `onnxruntime.dll` / `providers_shared.dll` / `providers_cuda.dll`
  到扩展旁（沿用 CPU SDK 的导入库，C ABI 同版本兼容）。
- 新增 `PuctSearch::setRootNoise(enabled, totalConcentration, weight)`：
  自对弈每步按全搜索/快速搜索切换根噪声，不再依赖构造期参数。
- 新增 `python/suzero_5dchess/cpp_self_play.py`：每轮导出 ONNX，调用
  `run_selfplay`，把数组还原为 `PositionSample`，并只保留最近 2 份模型。
  `pipeline.py` 按 `self_play.backend` 分派，两种后端共用后续训练/评估。
- 新增 `interrupts.py` 与流水线集成：第一次 Ctrl+C 完成当前轮并保存退出
  （`stopped_early=True`，退出码 0），第二次立即中止（CLI 返回 130）。
- `run_train.bat` 默认改为 `cpp_onnx` + `cuda` + 3 个 worker，并在缺少
  `onnxruntime_providers_cuda.dll` 时告警。
- 修复桥接在终局着法上的崩溃：`PuctSearch::resetRoot` 要求根节点至少
  有一个动作。现在若推进后的局面无合法动作，就不再提升搜索根（该轮随后
  结束，根不会再被读取）；普通情形用已推进局面的克隆重建根。
- 测试：新增 C++ `testSelfPlayBridgeSmoke`（2 局、2 线程、批量评估、
  样本形状与策略归一化），`suzero_5dchess_tests` 988 项通过；新增 Python
  `test_cpp_onnx_backend_runs_a_debug_iteration` 与 SIGINT 协作停止测试，
  Python 63 项全部通过。实测 CPU / CUDA 两种 provider 得到相同的样本，
  `run_train.bat` 端到端跑通并正确保存 checkpoint。

### 2026-10-23 · 修复训练迭代编号累加错误

- `run_training` 在每轮末尾执行 `iteration = completed`，而下一轮的
  `completed = iteration + offset + 1` 又基于它累加，导致迭代编号变成三角数
  （1, 3, 6, 10, … 210），而不是 1..N。实测一次 `num_iterations=20` 的训练
  记录为 `iteration 210/20`。
- 影响：checkpoint 文件名与内部 `iteration` 字段、TensorBoard 横轴、评估
  间隔（`completed - last_eval`）、`save_interval` 取模、以及续训时的
  `start_iteration` / `total_steps`（LR 调度）全部错位。评估本应每 20 轮一次，
  实际变成每约 3–4 轮一次。
- 修复：循环内不再改 `iteration`；循环结束后执行
  `iteration += config.num_iterations` 再写最终 checkpoint 与摘要。
- 新增 pytest `test_iteration_numbers_are_sequential`：跑 2 轮断言记录编号为
  `[1, 2]`、checkpoint 名正确，并续训 2 轮断言为 `[3, 4]`。此前的调试测试
  只跑 1 轮，按 1 轮时三角数与顺序数恰好相同，因此没暴露问题。
- Python 测试增至 57 项。

### 2026-10-23 · 修复 TensorBoard 日志路径的 NameError

- `run_training`（模块级函数）中误写成 `self._log_tensorboard(...)`，应为
  `_log_tensorboard(...)`。该分支只在启用 TensorBoard 时执行，导致第一轮
  训练结束即 `NameError`；CLI 默认开启 TensorBoard，所以 `run_train.bat`
  必现。已修正。
- 新增 pytest `test_pipeline_writes_tensorboard`：启用 `tensorboard=True`
  跑完一轮训练并检查事件文件生成，覆盖 writer 与 `_TensorBoardSink` 两条
  此前未被测试触及的分支。
- 用 AST 扫描确认 `python/` 下没有其他模块级函数误用 `self`。
- Python 测试增至 56 项，全部通过。

### 2026-10-23 · 进程内多对局线程 + 批量推理（对齐 KataGo 自对弈）

- 新增 `suzero::mcts::BatchedEvaluator` / `IBatchBackend`：请求队列 +
  `numServerThreads` 个 server 线程，每线程独占 backend；批量语义对齐
  KataGo `NNEvaluator`（取当前可用行、不等待凑满、队列背压、优雅关闭）。
- 新增 `suzero::mcts::SelfPlayRunner`：`numGameThreads` 个对局线程共享
  评估器，每局独占 `PuctSearch`；支持收集式/流式运行与协作停止；每局种子
  由 `(seed, gameIndex)` 决定，结果与线程调度无关。
- 5D 适配层：把协议解析、张量打包与结果组装提取为私有 `OnnxModel`；
  `OnnxEvaluator` 改为薄包装，行为不变。新增 `OnnxBatchBackend` 与
  `makeBatchedOnnxEvaluator()`，支持异构 batch（不同棋盘数/候选数，padding +
  `used_board_counts` 屏蔽，强制 `SUBMIT` 行不进 batch）。
- CMake：`suzero_mcts` 链接 `Threads::Threads`，加入 `batch_evaluator.cpp`、
  `selfplay.cpp`；`suzero_5dchess` 加入 `onnx_model.cpp`、
  `onnx_batch_backend.cpp`。
- 新增测试：批量评估与逐条评估结果一致（1 个与 2 个 server 线程、异构
  局面、batch=1）、backend 异常传播、配置校验、多线程自对弈确定性与流式
  回调、协作停止、5D 棋盘上限终局、5D 批量评估对齐单条评估、5D 批量自对弈
  冒烟。`suzero_mcts_tests` 1213 项、
  `suzero_5dchess_tests` 664 项全部通过。
- 修复测试宏 `CHECK_NEAR` 的变量遮蔽（`expected_value` 自初始化问题），
  改用 `suzero_check_*` 前缀。

### 2026-10-23 · 5D 象棋 AlphaZero 训练闭环

- 新增 `python/suzero_5dchess/`：协议、环境包装、样本校验与 collate、
  动态形状网络、PUCT、自对弈、回放、目标函数、ONNX 导出、评估、流水线与
  CLI；全部建立在既有 `suzero_train` 契约之上，未复制参考项目代码。
- 新增 pybind11 薄绑定 `_5dchess`（`SUZERO_BUILD_PYTHON_BINDINGS`），
  构建后自动复制 `onnxruntime.dll`。
- `d_model=128` 下参数 192,058，低于 200,000 硬上限；默认变体
  `very_small`。
- `train --debug` 端到端跑通自对弈、训练、checkpoint、续训与 ONNX 导出；
  导出的动态模型通过 C++ 适配层全部 410 项检查。
- 新增 9 项 pytest：参数预算、SUBMIT 列对齐、自对弈样本、线程并发一致性、
  ONNX 动态 B/N/q、调试流水线与续训、评估间隔与对局，以及 MCTS 换方
  价值翻转和访问计数；Python 共 52 项通过。
- 审查修复：PUCT 选择在换方边（SUBMIT）上翻转子节点 Q；移除展开时的
  虚假 visit 计数，使策略目标等于真实模拟次数；评估基线在训练开始前
  建立并按 `evaluation_interval` 调度；显式网络参数与 checkpoint 冲突时
  报错；反向 loss 声明为 batch mean，使 `max_grad_norm` 生效；保存/恢复
  torch RNG 状态；`record_games` 现在写出每轮对局 JSON。
- 自对弈支持 `workers > 1`（线程级并发）；`pyproject.toml` 增加
  `chess5d` 可选依赖与 `suzero-5dc` 入口；`.gitignore` 忽略绑定产物
  与训练输出。

### 2026-10-23 · 对齐 KataGo 训练细节（静态审查修复）

- 修正 README 示例：反向 loss 改为 batch 总和，与默认 `loss_reduction`
  及 KataGo 的 LR/裁剪标定一致；新增 `loss_reduction` 与 `forward=`
  （前向 autocast、loss 在 fp32）说明。
- `grad_norm_cap=0` 改为只测量梯度范数，不再以 `max_norm=inf` 调用
  `clip_grad_norm_`，避免 `inf/inf` 产生 NaN 梯度；校验
  `grad_norm_clip_scale > 0` 与各浮点配置的有限性。
- `_batch` 指标窗口改为按 `batch_metric_window_steps`（默认 100 step）收缩，
  与日志频率解耦；空窗口平均返回 `nan`；非有限指标清理时记录警告。
- 调度刷新与 LR/WD 刷新按 epoch 起点计数；新增 `schedule_batch_size`，
  使 LR/WD/裁剪使用固定 batch，而非数据源的零散 batch。
- 验证 loss 统一按 `loss_reduction` 归一为逐样本量，缺少 `loss_sum` 时自动
  补齐，移除“混用即报错”的限制。
- 参数组拒绝重复参数；新增 `reference_group` 指定上报 `lr_batch`/`wd_batch`
  的参数组；`nsamp` 指标在续训时用 `global_step_samples` 补齐。
- `InMemoryDataSource` 新增 `validation_collate_fn` 与 `start_epoch`
  （续训复现 shuffle 顺序），并明确验证集保留最后一个不完整 batch。
- 修正文档中 SGD batch 指数：LR 为 0.0，WD 为 1.0。

### 2026-10-22 · 提取 KataGo 训练方法（游戏无关）

- 新增 `python/suzero_train`：游戏无关训练抽象层，定义 `Trainer`、
  `TrainingDataSource`、`TrainingLoop`/`LoopConfig`、`MetricsSink` 与
  `TrainingBatch`/`TrainStepResult`/`TrainingReport`；不依赖 PyTorch。
- 新增 `python/suzero_train/pytorch`：`TorchTrainer` 接收任意
  `torch.nn.Module` 与 objective，实现 KataGo 的 per-sample LR、warmup、
  参数组、梯度裁剪、AMP、指标累计与 checkpoint 续训；附带
  `InMemoryDataSource` / `IterableDataSource`。
- 未复制 KataGo 的任何 Python 源文件；围棋模型、Board 编码、`.npz` 数据
  格式与导出流程均不引入。
- 新增 Python 抽象层与 PyTorch 训练器的 pytest 用例，用微型回归网络覆盖
  训练、验证、学习率/warmup、参数组、梯度裁剪和 checkpoint 恢复。
- 训练子系统不参与 CMake，C++ 核心与 5D 适配层保持不变。

### 2026-10-22 · 接入 5D 象棋（Very Small - Open）

- `IGameState::actionCount()` 契约扩展为**状态相关**：动作数可随状态变化，
  但必须是状态的确定函数；搜索改为按节点自己的状态构建先验，
  `resetRoot`/`advanceRoot` 不再假设固定动作数。
- 新增 `Chess5DGameState`：封装 5dchess_engine，生成小步/SUBMIT 候选，
  实现吃王、物质分、无合法动作终局，精确状态键与网络输入编码。
- 新增 `OnnxEvaluator`：`smallstep_movekey_v1` 协议、sidecar 与 IO 签名校验、
  fp32/fp16 输入输出、q=0 强制路径。
- 新增 `Chess5DAgent`：连接游戏、网络与 PUCT，参考 runner 的默认搜索参数、
  温度调度、强制 SUBMIT 与子树复用。
- 新增 `suzero_5dchess`/`suzero_5dchess_tests` 构建目标与 410 项
  验收检查；核心测试仍为 516 项。
- 加入导出的 very_small 模型（`models/5dchess/`）。

### 2026-10-22 · 对齐 KataGo 搜索细节（第二轮静态对比）

- 坏子降权改用 3 自由度 Student-t CDF（闭式实现，`src/internal_math.h`），
  并在剪枝后归一化回剪枝前总权重。
- 修正 Dirichlet 噪声 alpha 上限方向为 `min(0.01, p)`；策略概率合法性改为
  `>= 0`，零概率合法动作保留在搜索中。
- 根策略温度改为对数域软缩放，新增 `rootPolicyTemperatureEarly`；
  新增 `chosenMoveTemperatureEarly` 与 `chosenMoveTemperatureHalflife`，
  无参 `chooseAction()` 使用插值温度。
- 根节点最终权重改为 KataGo 的非 LCB play-selection 逻辑：
  `bestAction` 不再直接取访问数最多者，而是取回顾式降权后的最大权重。
- 选择顺序改为“已展开子节点优先，未展开只比较策略概率最高者”。
- 换根时删除不可达的被提升节点表副本；时间限制至少执行 2 次 playout。
- 新增 8 组回归测试；测试增至 516 项检查。

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
