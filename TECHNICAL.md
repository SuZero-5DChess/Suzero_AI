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
  实战根样本的终局价值按行棋方视角分配，支持吃王 ±1、物质分 tanh、无合法动作和中止；
  树内侧位置（默认开启）另用自身搜索价值，不套用实战结果。
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
  第一次按下只置位停止标志；若当时正在自对弈，后端不再发新对局，已开始的对局
  照常跑完，样本（文件后端逐局落盘）全部保存后跳过本轮训练，走正常的最终
  checkpoint 保存并返回 `stopped_early=True`，退出码 0；若正在训练/评估，则当前
  这一轮（含评估）照常完成后保存退出。下次启动从该 checkpoint 与回放文件续训：
  未完成轮按 `selfplay_progress` 只补跑剩余对局，进度显示从已保存局数起算。
  第二次按下恢复原处理器并抛出 `KeyboardInterrupt`，CLI 捕获后返回 130，允许
  用户放弃进行中的一轮。C++ 自对弈调用期间 GIL 已释放，Python 信号处理器会等到
  该调用返回后才执行；为此原生侧直接观察同一停止意图：`ConsoleInterruptGuard`
  第一次按下在毫秒级把提示写到 stderr（控制台走 `WriteConsoleW`，重定向走 UTF-8
  `WriteFile`），并置位 `g_stopRequested`，调度器随即停止派发新对局。C++ 的
  `CTRL_C` 事件返回 FALSE 以便 CPython 仍记录 SIGINT；父进程发给子进程组的
  `CTRL_BREAK` 返回 TRUE，避免子进程在局中被默认动作终止。第二次按下直接以 130
  终止进程（见 `src/5dchess/console_interrupt.h`）。
- **连续训练**：`--continuous` 让训练器按 `num_iterations` 划分为批次并
  无限运行：每批结束时 `latest.pt` 已发布，下一批以它续训；协作式 Ctrl+C
  使当前批的 summary 带 `stopped_early=True`，循环随即结束而不是再开一批。
  `run_train.bat` 的 `CONTINUOUS` 开关映射到该参数。
- **MCTS**：PUCT，`q_init=-1`，根节点按访问次数输出原始策略；不做置换
  合并（图搜索在 C++ 侧，Python 训练侧保持简单树）。`workers > 1` 时用
  线程池并发多盘，每盘独立环境、RNG 与搜索树；网络处于 `eval()`，推理
  线程安全，结果按游戏序号稳定返回。
- **训练**：默认使用文件式回放（`replay_backend="file"`，每局一个 NPZ，内存中只留
  计数与文件索引）并做行级 no-repeat shuffle；`--replay-backend memory` 可退回旧的
  内存 uniform 回放。可选 lossy replay window（具体语义见下节）；AdamW、余弦学习率、
  梯度范数裁剪；`AlphaZeroObjective` 把整个 batch 移到模型设备，策略 loss 在
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
  Python 与 C++（含多进程）后端均记录实际执行的完整动作流；训练样本默认保留
  full-search 根位置和树内侧位置，可用 `--no-record-tree-positions` 关闭后者；
  两者都不改变存档。`game_record.py` 默认写出 `gn5d` v2：只保留单个
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

### 树内位置采样（默认开启，2026-10-25）

仅实现 KataGo `cpp/program/play.cpp::recordTreePositions` 思路，不实现
`sidePositionProb` 的赛后独立搜索，也不实现 `reanalyzeCheapSearchPositions`。
`SelfPlayConfig.record_tree_positions=True`；`--no-record-tree-positions`
（或启动脚本 `TREE_POSITIONS=0`）可关闭。默认阈值 64、深度 5、每次完整搜索
最多 4 行、保留概率 0.25。

- **触发时机**：full-search 结束并选定真实着法后，推进状态/换根前提取。Python
  节点拥有 `GameEnv` 快照；C++ 节点只持统计量，沿动作 clone/apply 重建局面，
  必须在 `advanceRoot` 使旧节点指针失效之前完成。
- **质量与筛选**：排除根和根的实际着法分支；到达候选的边访问数与候选自身出边
  访问总数都至少达到 `record_tree_min_visits`。非终局、已展开且至少有一个非
  SUBMIT 动作才可记录。跟踪各 player 的 always-best 标记，非最佳动作只清除
  实际走该动作一方的标记；候选须满足当前行棋方标记。最佳动作用局部边访问数
  最大值选取，平局取最小 action id。5D 不按深度奇偶交换玩家。
- **有界遍历**：局面键去重、防环；深度范围 1–64，默认 5 个小步（含 SUBMIT），
  单次最多检查 4096 个节点并在达到行数上限后结束。`max_per_turn` 的 turn 是
  搜索决策，不是 SUBMIT 分隔的整回合。去重只在单次提取内，不抑制后续搜索的新目标。
- **监督来源**：策略为该节点自身出边原始访问次数归一化，不使用零访问时的先验
  回退；C++ 不把共享子节点全局访问数当作边分布。value 为 Python `Node.q` / C++
  `NodeStats::utilityAvg()`，均为当前节点行棋方视角。实战根样本最后分配终局标签，
  树样本保留搜索标签，不能统一覆盖。
- **权重**：候选按独立 RNG 流做 Bernoulli 保留，保留后当普通行训练，无 per-row
  loss 权重。借鉴 KataGo fractional targetWeight 的随机整数化，但增加了本项目的
  质量/数量限制，不声称与完整上游采样分布或 batch loss 严格等价。
- **可复现与成本**：不改主对局 RNG、搜索节点统计或模拟预算，不额外调用评估器；
  原有噪声影响的树统计仍被复用，并非重新做一次无噪搜索。clone/重放、键生成和
  编码仍有 CPU/内存成本，新增训练行不自动扩大训练预算。
- **传输与兼容**：C++ 样本新增数值数组 `sample_source`（0=实战，1=树内）和
  `search_visits`，多进程 NPZ 白名单显式转发。Python 转为现有数值 metadata
  `tree_position=1` / `search_visits`，压缩 replay 格式不升级；ONNX 7 输入/3 输出
  不变。关闭功能时不传新 kwargs，可用旧扩展；默认开启则要求
  `TREE_POSITION_FORMAT_VERSION=1`，缺失时在导出/自对弈前明确报错要求重建，
  或显式关闭后继续使用旧扩展。
- **观测**：`SelfPlayResult.tree_samples`、JSONL `self_play.played_samples/tree_samples`
  与 TensorBoard `SelfPlay/*` 统计分别计数；replay 累计生成量包含保留的树内行。
  真实动作列表及 GN5D 归档不接收侧分支。

### Lossy replay window：计数、裁剪、采样与恢复

窗口是**有损的最新样本 FIFO**，不是全量训练数据归档。`WindowConfig` 默认关闭；
关闭时仍沿用 `replay_buffer_size` 的固定容量，但采样方式取决于回放后端：文件后端
（默认）依旧按文件顺序做行级 no-repeat 抽轮，内存后端则沿用旧的 uniform 语义。
开启时 `min_rows` 取 `replay_buffer_size`，`max_samples` 默认 500,000，是有限
上限，但在未训练积压存在时是软上限：只有训练计数大于 0 的行才允许淘汰。
`min_replay_size` 仍单独决定是否开始训练，不等同于窗口下限。

目标窗口采用先 clamp、后 floor 的公式：

```text
m = replay_buffer_size                           # min_rows，默认 100000
s = taper_scale if specified else m
W(N) = floor(clamp(
    m + expand_per_row * (N**taper_exponent - s**taper_exponent)
        / (taper_exponent * s**(taper_exponent - 1)),
    m, max_samples))
```

默认 `expand_per_row=0.4`、`taper_exponent=0.65`、`taper_scale=None`，
`keep_samples=None`（不限制训练子集）、`sampling="shuffle"`。`s` 是增长公式的
尺度，不是保存的历史行数。小于 `m` 的原始目标被钳制为 `m`，超过 `max_samples`
则不再扩大；对非整数目标最终向下取整。

- **生成计数 `N`**：累计成功 `push` 的训练样本数；不是现存行数、对局数、动作数，
  也不是训练器的 `samples_seen`。FIFO 淘汰、窗口裁剪和训练读取均不减少或增加 `N`。
  强制 SUBMIT 与不产训练样本的 fast-search 位置不计入。
- **裁剪时点**：先完成本轮所有结果的 `push`，再按更新后的 `N` 计算目标窗口并在
  训练前 `trim` 一次；启动恢复后也先裁剪再进入后续流程。不能在每个游戏入库后按
  动态窗口逐次裁剪，以免窗口随后增长时较老样本已经丢失。
- **容量兜底**：开启窗口时 `ReplayBuffer.capacity = max_samples`；`push` 和加载
  都在这个上限处尝试截取最新行。截取按“只删已训练行”执行，因此未训练积压会暂时
  把窗口顶到容量之上。实际保留量还受可用历史限制；`min_rows` 是目标下限，不能
  凭空补齐样本。窗口日志的 `evicted` 包括本轮 `push` 容量保护和动态 `trim` 合计
  删除的行数。关闭窗口时不新增采样行为或窗口运行日志，只增加配置和 replay 计数
  元数据。
- **只淘汰训练过的行（核心不变量）**：两个后端都维护与存活行一一对齐的训练计数
  （`train_counts.py`），数据源在真正呈现一批行时才递增。`trim_to`、push 超容量、
  加载时缩小容量都只删除计数大于 0 的前导行；遇到计数为 0 的行立即停止。由于淘汰
  只从头部进行，一个未训练的行保护它后面的全部行，窗口/容量成为软上限。积压被
  训练后，下一轮淘汰才能越过它。旧快照没有计数时不视为已训练，加载会完整保留。
- **未训练行优先**：发布 shuffle 轮时先打乱，再做稳定分区把 0 计数的行放到前面；
  本轮训练预算先覆盖未训练积压，再重复已训练行。全部为 0 的新窗口顺序不变。
- **分工与发布**：`python/suzero_5dchess/replay.py` 仍在 RAM 管理当前 lossy FIFO，
  `file_replay.py` 的 `publish_shuffle` 把当前窗口的一次 shuffle 发布为一轮训练视图，
  `pipeline.py` 同步编排自对弈、发布与训练。默认每片 4096 行，NPZ 只承载现有 zlib
  blob 与 offsets 等数值数组，不改样本 codec，也不是上游的围棋 NPZ 数据格式。
  写完分片和 `manifest.json` 后，原子发布 `current.json`；manifest 记录本轮选取的
  窗口行索引 `rows`、窗口行数 `pool_samples` 和排除集摘要 `exclude_digest`。同输入、
  同 seed 重试幂等复用同一轮，不借重试创建新身份重读。发布后清理旧轮，失败时清理
  临时视图；不保存永久 raw 归档，不从旧轮恢复已淘汰样本。
- **每轮选取**：仅 `window.enabled` 且 `sampling="shuffle"` 时走文件路径。每轮从
  **上一轮未选中且尚未训练的行**中无放回选取 `min(keep_samples, 可用行)` 行（未设
  keep 则取全部可用行）并打乱。`keep_samples` 不缩小持久化窗口，也不改变窗口增长。
- **行级 no-repeat**：重复身份是 replay 窗口行，不是输出文件路径。新一轮不能重复
  上一轮**实际服务过**的行（训练计数大于 0）；上一轮选了但没读到的行仍是 0 计数，
  下一轮可以重新提供，避免未训练样本被连续跳过直到淘汰。写入新目录不代表允许
  重读。`FileReplayDataSource` 保存 generation、file 与 row 游标，跨 epoch 连续读，
  不归零、不回绕；本轮读完后若 epoch 预算未用完，就在训练线程内调用发布器产生
  下一轮（排除上一轮已训练的行）并继续，所以一个 epoch 可以包含多轮 shuffle。
  `keep_samples=None` 时一轮即整个可用窗口，通常每轮只发布一次。
- **跨阶段平移**：新一轮发布前读取上一轮 manifest，用 `pool_samples +
  (N_now - N_then) - len(buffer)` 计算窗口头部淘汰数，把上一轮行索引整体前移；
  已淘汰的行退出排除集。窗口只从头部删除，因此该平移是精确的。
- **文件后端（默认）**：`--replay-backend file` 时整个窗口是一个追加型单文件
  `replay_store/replay.log`，单位是**局面行**而不是整局。每个完成的对局成为一帧：
  帧头 `<4s I I H I I q i>` 记录 magic、payload 字节数、payload 的 CRC32、版本、
  总行数、已裁掉的前导行数、`game_id` 与 `iteration`，payload 是
  `rows × [u32 长度][zlib 样本]`。对局完成即追加，不必等整轮结束；C++ 桥与多进程
  子进程在同一把 `replay.lock` 锁下写同一文件，父进程只重建帧索引（偏移、行数、
  计数），样本 blob 从不驻留内存。实现集中在 `suzero_5dchess/row_log.py`。
- **按行裁剪**：`trim_to` 删除最老的行，可以切掉最老一局的前导行；请求量先按训练
  计数截断，只删已训练的前导行。帧头的 `skipped` 记下切掉多少，幸存行保留原来的
  `(game_id, local row)` 编号，所以 `last_round.json` 的排除集跨裁剪仍然有效。裁剪
  把存活帧重写进 `.compact` 临时文件再 `os.replace`，只在父进程、自对弈子进程退出
  后执行。
- **留缓冲、成批删**：窗口允许超出目标约 1/5（`TRIM_MARGIN_DIVISOR=5`），超出后一次
  删回目标，而不是每局重写日志；追加路径触发的 `_trim_above` 删到目标下方的缓冲带，
  给下一批对局留空间。因此正常情况实际窗口在目标与其 1.2 倍之间波动；未训练积压
  存在时可以更高，直到训练计数覆盖到它们。
- **旧布局导入**：首次扫描发现 `g<id>-<count>.npz` 时按 game-id 顺序导入日志，再把
  归档移到 `replay_store/legacy/`；导入在写锁内进行，已进日志的 id 不再导入，崩溃
  残留的归档因此安全。
- **训练抽轮**：`GameFileDataSource` 同样按行级 no-repeat 抽轮：一轮排除上一轮实际
  训练过的 `(game_id, row)`，一个 epoch 可推进多轮，读取按帧偏移顺序且只持有当前
  对局。上一轮选择原子写入 `replay_store/last_round.json`；同 iteration、seed、窗口
  与 `keep_samples` 的重试复用该轮，否则把其中已训练的行当作排除集。被裁掉的行
  自然退出排除集。
- **步数与尾批**：每个 epoch 的上限仍是 `requested_steps`（本轮无可选行时为 0）。
  batch 可以跨分片拼接；数据最终不足一批时允许 partial，不重复补样。预算用完便
  保存游标，下一 epoch 接着读；不会为了凑满预算重读上一轮已训练的行（未训练行
  由下一轮重新提供）。
- **耗尽与同步循环**：排除上一轮后没有可选行时结束本轮训练，置
  `waiting_for_shuffle`，由外层下一轮自对弈更新窗口后再发布。epochs/steps/batch
  配置保持原值，是上限而非必须凑满的呈现次数；数据不足时实际少训。
- **uniform 兼容路径**：`--replay-backend memory` 且（`sampling="uniform"` 或窗口
  关闭）时沿用内存源；每批从整个当前 buffer 均匀无放回，不同 batch 可重复，
  `keep_samples` 不影响此路径。文件后端始终走上面的文件顺序抽轮，因此
  `replay_backend="file"` 搭配 `window.sampling="uniform"` 会在配置校验时报错，
  而不是静默忽略该选项。
  非空 buffer 的 epoch 步数仍为
  `min(requested_steps, max(1, len(buffer) // batch_size))`，无文件消费状态。
- **恢复 `N`**：取
  `max(extra.total_samples, buffer.total_generated, len(buffer))`，避免计数倒退；
  replay 保存 `total_generated`，权重 checkpoint 的 extra 保存 `total_samples`。
  旧 replay 没有生成计数时，可用权重 checkpoint 补齐；二者都没有历史累计值时，
  只能以现存行数为下界，无法恢复此前已淘汰的生成历史。replay 文件缺失时仍按原有
  告警路径继续，累计计数不代表丢失的样本可被还原。各计数不一致时告警并取最大值；
  若把 `max_samples` 调小，加载时按新容量截断——同样只删已训练的行，并在删除时
  告警“may discard up to … oldest trained samples”——再按当前 `W(N)` 裁剪。
- **保存范围与消费恢复**：checkpoint 流程把**整个当前窗口**写入独立的
  `replay_buffer.pt`，不是仅保存 keep 子集；其 `shuffle_state` 与权重 checkpoint
  extra 的 `replay_shuffle` 保存当前 generation/file/row 消费状态。独立使用
  `FileReplayDataSource` 时，只有同轮文件仍存在且 state 匹配才能据此续读。
  完整 `run_training` 按 iteration 边界恢复：启动时丢弃 scratch 旧轮视图，先进行
  下一轮 self-play，再发布新轮；**不承诺 mid-iteration 恢复**，也不保证逐 optimizer
  step exactly-once。权重、replay 和文件视图不是一个逐 step 联合事务。不新增
  逐轮窗口归档，旧轮/临时视图可删除。磁盘上可能还有旧权重、旧 replay 副本或其他
  快照，但本功能不查找或恢复旧窗口；恢复旧权重也不保证取得其当时的 replay。

跨轮行级去重由 `TrainConfig.no_repeat_rows`（默认 `true`，CLI `--repeat-rows` 或
bat 的 `NO_REPEAT_ROWS=0` 关闭）控制。关闭后每一轮都从整个候选池抽样，相邻轮允许
重复行，池较小时也能每轮拿满 `keep_samples` 或整池；`last_round.json` 仍记录本轮
选择，只影响日志中的 `no_repeat_rows` 字段。该开关只改采样，不改 `trim_to`。

**固定预算不随窗口增长而增长。** 每轮 10 epoch × 10 step × batch 256 对应最多
25,600 次训练呈现，而非必须完成的数量。令实际训练呈现量为 `T`、每轮生成量为 `G`，
稳定生成且窗口充分时，样本存留时间约 `W / G` 轮，每轮覆盖比例约 `T / W`，相乘
得到平均复用约 `T / G`，窗口大小约去。只有实际用满预算且假设 `G=11,400` 时才有
`25,600 / 11,400 ≈ 2.25`；行级 no-repeat 耗尽、keep 子集或剩余数据不足均可能使
`T < 25,600`，必须按真实 `T` 计算。这里的 11,400 只是计算示例，不是实测。`W`
改变存留时间、每轮覆盖和自对弈时间跨度，不会自动增加平均复用。启动增长期与其他
非稳态条件会影响近似。窗口不改 epoch、step、batch、自对弈、对战预算或 gate 行为。

仅启用窗口时，每轮 JSONL/控制台增加 `replay_window`：`total_generated`、
`window_size`、`buffered`、`evicted`、`sampling`、`keep_samples`、`estimated_bytes`、
`coverage`；coverage 使用**实际**训练报告的样本呈现数除以 `max(1, len(buffer))`，
跳过训练为 0，不把请求配额记为实际训练。uniform 的该比值仍可因重复呈现大于 1，
不是通用的全历史去重率。TensorBoard 提供
`Replay/window_size`、`Replay/buffered`、`Replay/coverage`。
文件路径另记 `training.shuffle`：发布式视图包含 `generation`、`files`、
`file_index`、`row_offset`、`files_used`；文件后端包含 `generation`、`files`、
`pool_samples`、`round_samples`、`remaining_samples`、`rounds`、`no_repeat_rows`。
两者都带 `waiting_for_shuffle`，用于区分预算结束和排除上一轮后无可选行的数据耗尽。

每轮训练报告（`training.train_counts`）另外给出**当前窗口内**每个自对弈样本已被
训练的次数分布：`samples`（窗口行数）、`trained`（次数 > 0 的行数）、`untrained`
（次数 == 0）、`total_trains`（总呈现次数）、`mean`、`median`（下中位数）、`p90`
（最近秩）、`max`，以及 `histogram`（`0`…`9`、`10+`）。计数只在数据源真正把一批
行交给训练循环时累加；被裁剪/淘汰的行连同计数一起消失，因此统计对象始终是当前
可训练窗口，“生成过但已淘汰”的历史样本不在分布内。文件后端把计数按
`(game_id, local_row)` 保存在根目录的 `train_counts.npz`（与 `replay_buffer.pt`
分开；行标识稳定，部分裁剪后仍能合并）；内存后端随 `ReplayBuffer.state_dict()`
一起保存。TensorBoard 记 `Replay/train_count_trained`、
`Replay/train_count_untrained`、`Replay/train_count_mean`、
`Replay/train_count_median`、`Replay/train_count_p90`、`Replay/train_count_max`。
计数 sidecar 损坏只告警并从零开始，不阻断续训。

**内存与磁盘按实测压缩格式估算。** 窗口仍常驻为逐样本 zlib blob，不是 float32
one-hot 棋盘。先用代表性真实样本测得压缩总字节数 `B` 与行数 `n`，以
`W × (B / n)` 估算 RAM blob 负载；另计 Python bytes/list、offsets、保存/发布临时
副本、分片加载、batch 解码、padding、张量和自对弈结果的峰值。磁盘除
`replay_buffer.pt` 外还保存当前 shuffled 视图（约选取行的压缩 blob 加 NPZ/manifest
开销）；切轮期间旧轮与新轮可短暂并存，需预留双份视图峰值，不能当成完整 raw-disk
数据集或零额外磁盘成本。不使用每行 60 KB 假设，不给出未经实测的压缩比或容量。

### 终局样本过采样（`run_train.bat` 默认开启）

回放行按 `move_count / 本局最大 move_count` 计算局内进度，权重为
`1 + strength * progress**exponent`（`endgame_sampling.py`）。发布轮次时，每个可用行
仍入选一次，权重高于 1 的行再按整数部分必然、小数部分按概率追加副本；
`keep_samples` 子集则先按权重无放回抽取（Gumbel top-k）再追加副本。因此训练步数与
行覆盖面不变，只改变同一 epoch 内各行的呈现次数。

- **进度来源。** file 后端由 `GameStore.row_move_counts()` 从压缩记录读取定长头里的
  `move_count`（`sample_codec.peek_move_count`，不完整解压样本），按 `game_id` 缓存
  并按前导裁剪切片；memory 后端在 `push()` 时按一次调用一局计算并随
  `ReplayBuffer.state_dict()` 保存 `progress`。旧快照没有该字段时，恢复路径用全窗口
  最大 `move_count` 归一化兜底；可丢弃的 shuffle shard 与 legacy 迁移跳过这次扫描。
- **语义边界。** 权重按行作用，不改变 `keep_samples`、`no_repeat_rows`、
  `trim_to`、`train_counts` 的既有规则；同一行的副本各记一次训练。轮次 sidecar
  记录 `endgame`，同 iteration/seed/窗口但设置不同的重试不会复用旧轮次。
- **默认值。** 库默认关闭以免续训旧 checkpoint 时静默改变分布；`run_train.bat`
  默认 `ENDGAME_SAMPLING=1`、`ENDGAME_WEIGHT=1.0`、`ENDGAME_EXPONENT=2.0`。

### Gatekeeper：training 与 accepted 的同步分离

`TrainConfig.gate_interval=1` 默认启用；`--gate-interval 0`（bat 的 `GATING=0`，或
`debug_train_config()` 预设）关闭。关闭时 pipeline 不实例化 gate，不创建新产物、
不增加 JSONL 字段，两个自对弈后端继续使用同一个 training network。配置新增字段会
随 `train_config` 元数据保存，不改变现有权重、optimizer 或 checkpoint 格式语义。

开启时，`gating.ModelGate` 负责准入状态和对局，pipeline 只接线：

1. 先恢复 trainer（包括 optimizer），再加载 accepted state；无 state 时从恢复后的
   training network bootstrap，iteration 为 resume 轮次，新 run 为 0。
2. 每轮从 state 解析 accepted，按 `(iteration, checkpoint path, mtime_ns, size)` 缓存。
   通过现有 `load_network_checkpoint` 加载到 trainer.device，设置 eval 并关闭梯度；
   它不是 trainer.model 的别名。Python 与 C++ 自对弈使用相同的 resolved network。
3. 训练始终更新 training model，按现有配置导出训练 ONNX。实际 gate 前额外保存完整
   候选 checkpoint 及 replay，避免稀疏保存或提升后中断时只有不可训练恢复的权重快照。
4. gate 通过时写不可变 accepted 快照，再原子更新 state；下一轮使用新 accepted。
   拒绝、有效局不足或对局异常不回滚训练，不重跑自对弈，也不让被拒权重生成数据。
5. 之后继续原 periodic match/evaluation 及保存。实际 gate 轮次也发布训练 latest，
   checkpoint 的 extra_state.accepted 只保存 iteration/checkpoint 诊断摘要。

`RunPaths` 集中定义 `<run>/accepted`、`<run>/accepted/state.json` 与 `<log>/gates`。
state 为 format 1，含 iteration、checkpoint basename、variant、network_config、
UTC updated_at 和可选 last_gate；它是唯一权威来源。历史快照命名
`accepted/agent_iter_XXXX.pt`，使用 model/extra_state 布局，格式常量集中在
`checkpoint.py`。同名文件仅可在元数据及全部权重一致时复用，不覆盖不同内容。

启动和重新解析会检查 state 格式、变体、网络配置及快照可加载性，快照元数据必须与
state 对齐。accepted iteration 领先 resume 时明确报错，要求恢复同轮完整训练 checkpoint，
或明确删除 state 后重新 bootstrap；若历史同名快照冲突，应使用新 run 目录。快照或
state 写入失败属于致命一致性错误，不能被宽泛异常捕获转换成对局失败。

生产对局复用 export_network、MatchPlayer、play_model_match：候选固定为 participant 0，
双方分别导出临时 ONNX，按 gate_games 换色，使用 gate_workers、gate_simulations、
gate_temperature（0.1）、gate_provider、gate_max_boards，以及 self_play.material_scale；
max_moves 固定为 0。gate 开启时启动即检查 C++ run_match，缺失不降级。加载、导出及
临时模型操作包在 `torch.random.fork_rng(devices=[])` 中，保护 CPU Torch RNG；导出使用
CPU 模型副本，不修改 training 的权重、设备或各子模块模式。完整对局 runner 可注入，
单元测试不依赖真实对局、ONNX 或 GPU。

纯函数 decide_gate 从 totals[0] 校验局数，并以每局候选 utility 的平均分判定：
finished=wins+losses+draws，average=mean(scores)，截断不计入。必须 finished>0、
达到最小局数，且平均分 >= gate_min_average_score 才通过（等于门槛通过）；material
棋盘上限终局保留连续 tanh 得分，不压成 1/0。
异常或畸形对局结果以 match_error 拒绝，训练继续；零有效局以 insufficient_finished 拒绝。
未训练优先于 interval 跳过，不执行对局或创建 gate 报告。

每次实际 gate 追加 `<log>/gates/iter_XXXX_<uuid>.json`，含 summary、原始 match 及可选 error。
仅开启时迭代记录新增 selfplay_model_iteration、gate；控制台打印模型轮次、accepted
迁移及平均分，TensorBoard 增加 Gate/average_score 和 Gate/accepted_iteration。latest.pt/agent_final.pt
仍是训练最新模型，marked.pt 仍是旧 evaluation baseline，UI/推理入口不改。

本实现借鉴 KataGo gatekeeper 的“训练候选 vs 数据生产 accepted”职责分离，但同步串行
执行，不包含异步五进程架构、独立守护进程、热切换或自动重测，也不实现提前结束对局。
固定局数的得分门槛不是显著性检验。对应关系见 docs/katago-mapping.md 第 7 节。

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
   尚未实现。自对弈样本已按局流式落盘（见文件回放），但没有 KataGo
   `SelfplayManager` 那样跨轮的后台数据管线。
8. 参考 runner 的 Dirichlet 噪声是对称分布（`alpha=0.3`），本库的根噪声是
   KataGo 策略形状分布，两者只能近似对应；`Chess5DAgent` 默认关闭根噪声，
   自对弈训练时可自行开启并调整 `rootDirichletNoiseTotalConcentration`。
9. Python 训练器只支持单进程；DDP/多 GPU、Muon、SWA、`torch.compile`、
   LR cycle 尚未移植。`python` 后端的 `workers` 是 Python 线程，共享同一
   网络，不跨对局批量推理；`cpp_onnx` 后端用 C++ 对局线程 + 批量评估器，
   可跨对局合批，但整个批次仍只用一个执行设备，且 `record_games` 只能
   记录样本、没有逐着法历史（C++ 桥不生成 `moves`）。
10. 回放默认写入单文件 `replay_store/replay.log`（自研帧格式：CRC32、跨进程
    写锁、行级裁剪，`.compact` + `os.replace` 原子重写），内存中只保留帧索引；
    `--replay-backend memory` 时仍按逐样本 zlib 序列化为 `replay_buffer.pt`。可选
    lossy replay window 只维护当前 FIFO 窗口，不提供完整历史数据仓库或旧窗口恢复；
    可从旧版 `g<id>-<count>.npz` 目录自动导入，尚不支持从外部数据集导入。
11. Python 侧 MCTS 不做置换合并、评估缓存或批量叶子评估；大规模训练时
    这些是最先需要补齐的性能项。

## 变更记录

### 2026-10-26 · 回放样本库默认上限改为 100,000

- `TrainConfig.replay_buffer_size` 与 `ReplayBuffer` 的默认容量从 50,000 改为
  100,000；CLI `--replay-size` 未指定时继承该默认值，`run_train.bat` 无需改动。
- 开启窗口时 `min_rows` 随之变为 100,000，`--window-max-samples` 必须不小于它；
  需要更小窗口时显式调小 `--replay-size` 或调大上限。
- 同步更新 README、TECHNICAL 与 `docs/katago-mapping.md` 中的默认值和示例。

### 2026-10-26 · 只有训练过的回放行才允许淘汰

- 修正抽样造成的未训练样本丢失：此前窗口裁剪、push 容量兜底和加载缩小容量都
  按 FIFO 直接删行，一轮产出大于训练预算时，大量样本在从未被训练前就被淘汰。
- `train_counts.py` 新增 `evictable_prefix()`（只允许淘汰前导已训练行，遇到 0 计数
  即停止）与 `untrained_first()`（稳定分区，让训练预算先覆盖未训练积压）。
- `ReplayBuffer` 的 push 超容量、`trim_to`、`_load_compact` 与 `GameStore` 的
  `trim_to`/`_trim_above` 全部改为只删计数大于 0 的行；未训练的前排行保护其后的
  所有行，窗口与 `capacity` 因此成为软上限。无计数（旧快照/legacy 格式）的行一律
  保留，不视为已训练。
- 启动时先恢复 `train_counts` 再裁剪，避免文件后端把未训练行当成可淘汰行。
- 跨轮排除集只包含上一轮实际训练过的行；上一轮选中但未读到的行在下一轮重新参与
  抽样。发布 shuffle 轮时按 `untrained_first` 排序，优先训练积压样本。
- 验证：Python 863 项通过；新增/改写计数、窗口、文件后端与流水线断言，覆盖软容量、
  未训练行保护、加载截断与跨阶段重新提供未训练行。

### 2026-10-26 · 按局内进度过采样接近终局的行

- 新增 `suzero_5dchess/endgame_sampling.py`：`EndgameSamplingConfig` 与进度/权重、
  轮内重复、按权重无放回抽样等纯函数。
- `ReplayBuffer` 记录逐行 `progress` 并随快照保存；`ReplayDataSource`（uniform /
  shuffle 子集）与 `publish_shuffle` 改用加权抽样。`GameFileDataSource` 发布轮次时
  通过 `GameStore.row_move_counts()` 读取 `move_count` 并为靠后的行追加副本。
- 轮次 sidecar 与 generation 身份包含 `endgame` 设置；`_validate_round` 仅在该设置
  存在时允许重复行。
- `TrainConfig.endgame`、CLI `--endgame-*` 与 `run_train.bat` 的
  `ENDGAME_SAMPLING/WEIGHT/EXPONENT` 接线；库默认关闭，启动脚本默认开启。

### 2026-10-26 · 追踪每个回放样本被训练的次数分布

- 新增 `suzero_5dchess/train_counts.py`：`summarize()` 输出当前窗口的
  `samples/trained/untrained/total_trains/mean/median/p90/max` 与
  `0…9/10+` 直方图；`record()` 只累加数据源真正呈现给训练循环的行。
- 三种数据源接入计数：内存 uniform/shuffle 按 buffer 行号，文件 shuffle 按
  manifest 的 pool 行号，`GameStore` 按窗口行号；裁剪时同步删除对应计数。
- 持久化：内存后端随 `ReplayBuffer.state_dict()` 保存；文件后端写独立的
  `train_counts.npz`（`(game_id, local_row)` 标识，加载时按当前窗口合并，损坏只
  告警）。每轮 JSONL 的 `training.train_counts` 与控制台日志都会输出分布，
  TensorBoard 增加 `Replay/train_count_*`。
- 验证：Python 826 项通过（新增 `test_train_counts.py` 11 项，并补充窗口/文件
  后端流水线的计数断言）。

### 2026-10-25 · 回放改为单文件行日志，裁剪按行成批进行

- `replay_store` 从“每局一个 NPZ”改为单个追加型 `replay.log`：每局一帧，帧头带
  CRC32、总行数、已裁掉的前导行数与 `game_id`/`iteration`，payload 是逐行的 zlib
  记录。新增 `suzero_5dchess/row_log.py` 承载帧编解码、`replay.lock` 跨进程写锁、
  增量索引与重写式裁剪；`GameStore` 只保留窗口策略、旧布局迁移、轮次 sidecar 与
  checkpoint 状态。
- 窗口单位改为局面行：`trim_to` 删除最老的行，可以切掉最老一局的前导行（`skipped`
  保留幸存行的编号），不再整局丢弃；训练源按帧偏移读取。
- 裁剪留缓冲、成批进行：窗口可超出目标约 1/5，超出后一次删回目标；追加路径的
  `_trim_above` 删到目标下方留出下一批空间，避免每局重写日志。
- 多进程：子进程在同一把文件锁下追加，写前增量同步并修复崩溃留下的残帧（CRC
  判定）；父进程在子进程退出后才重写日志。并行 worker 可以交错写入 id，
  `next_game_id` 取日志中的最大 id 而不是最后一帧。
- 迁移：首次扫描把旧 `g<id>-<count>.npz` 导入日志后移到 `legacy/`；旧 checkpoint 的
  `suzero_game_store_v1` 状态仍可载入（忽略文件清单），旧 `last_round.json` 忽略。
- 验证：Python 801 项通过（新增 `test_row_log.py` 10 项，以及行级裁剪、留缓冲、
  迁移、id 交错等测试）；用现网 4,167 局 / 35,007 行的旧目录实测：导入 4.8 s、
  重扫 31 ms、18.1 MB 日志裁剪一次 0.05 s，两进程并发各写 40 局无丢失无重复。

### 2026-10-25 · 回放跨轮去重可关闭，启动脚本 gate 间隔改为 3

- 新增 `TrainConfig.no_repeat_rows`（库默认 `true`，`run_train.bat` 默认
  `NO_REPEAT_ROWS=0`），CLI `--repeat-rows` / `--no-repeat-rows`。关闭时
  `GameFileDataSource` 与
  `publish_shuffle`/`FileReplayDataSource` 的每一轮都从整池抽样，不再扣除上一轮
  选中的行；`round` 日志与 summary 如实报告 `no_repeat_rows=false`。
- `run_train.bat` 的 `GATE_INTERVAL` 默认值由 1 改为 3（库默认仍为 1），并新增
  `NO_REPEAT_ROWS` 透传；gate 间隔按已完成训练轮计数，被拒绝的候选要等到下一个
  间隔才复评。
- 验证：Python 784 项通过；新增文件后端跨轮重复、发布式路径不传排除集、CLI 开关
  与启动脚本透传四类测试。

### 2026-10-25 · 续训按已保存对局数补齐（进度不从 0 重数）

- checkpoint 的 `extra_state` 新增 `selfplay_progress = {"iteration", "games"}`，
  记录最近一次自对弈把多少局写进了当前（可能未完成的）轮次。
- 续训进入首个未完成轮时先读该记录，缺失时回退为统计 `GameStore` 中
  `iteration` 相同的对局；`already` 截断到 `num_games`，本轮只再自对弈
  `num_games - already` 局，`game_id_start` 顺延，进度行先 `advance(already)`
  再叠加回调增量，因此显示的是 `257/1000` 而不是 `0/1000`。若已全部保存
  （停止发生在最后一局之后），直接进入训练，不再重跑自对弈。
- 补齐时自对弈 seed 偏移 `already * 1_000_003`，避免与已保存对局复用同一条
  RNG 流而产生重复对局；`total_games` 仍只在真正新生成时累加，不会重复计数。
- 验证：Python 780 项通过；新增“部分保存后续训补齐”和“全部已保存则不重跑”
  两项流水线测试，覆盖 `num_games`、`game_id_start` 与进度回调累计值。

### 2026-10-25 · 文件回放默认：逐局落盘与自对弈期间协作式 Ctrl+C

- `replay_backend` 默认改为 `file`：每局结束立即写成
  `replay_store/g<game-id>-<count>.npz`（`.tmp` + `os.replace`），训练直接从磁盘
  读取；`replay_buffer.pt` 只保存计数、文件索引与窗口状态。`--replay-backend
  memory` 保留旧的内存路径。文件后端与 `window.sampling="uniform"` 的组合现在
  在配置校验时报错，避免选项被静默忽略。
- C++ 桥新增逐局流式回调：`runSelfPlay` 的 `BridgeGameCallback` 携带
  `(done, index, payload)`，`GAME_STREAM_FORMAT_VERSION = 1`；请求流式时不再返回
  整批样本数组。单进程 `cpp_onnx` 与多进程子进程都用同一路径直接写共享
  `GameStore`，父进程只重新扫描并载入元数据。
- Ctrl+C 改为自对弈期间协作停止：原生调度器在取号前检查停止谓词，已开始的对局
  自然跑完，返回连续的 game-index 前缀；`run_selfplay` 返回值新增 `stopped`。
  多进程父进程在首次检测到停止后，等各子进程写出第一行进度（此时 Python 处理器
  已安装）再向子进程组发送一次 `CTRL_BREAK`（POSIX 为 `SIGINT`），子进程在
  `result.json` 中记录 `stopped`；
  只有已停止时才允许返回短前缀。流水线在自对弈结束后发现停止就跳过训练，保存
  checkpoint 与 replay 状态，下次启动从该进度续训。
- 更新 `README.md`、`TECHNICAL.md`、`run_train.bat` 与 CLI 帮助文本，描述
  文件布局、流式写入、`--replay-backend` 以及新的 Ctrl+C 语义。
- 验证：Python 768 项通过；C++ `suzero_5dchess_tests` 1196 项检查通过。新增
  覆盖包括逐局落盘、子进程流式、停止信号只发一次、短前缀校验与
  “中断后下次续训仍保留已保存对局”。

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
