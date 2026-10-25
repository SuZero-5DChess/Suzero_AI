# 多进程单 GPU ONNX 自对弈基准（实验）

本目录是一个**独立实验入口**，用于比较“多个互不共享 Session 的进程在同一张
GPU 上各自运行 C++ ONNX 自对弈”的吞吐表现。

- 不修改现有训练入口、训练配置或旧 benchmark。
- 不导出模型、不构建扩展、不训练网络。
- **不是多进程训练**：只做自对弈并输出统计，不写 replay，也不接入 replay。
- 父进程只使用 Python 标准库，不导入 `suzero_5dchess` / torch / onnxruntime；
  真正的绑定导入和自对弈发生在 `worker.py` 子进程中。

## 运行前：GPU 忙时自动拒绝

`--provider cuda`（默认）启动前会执行
`nvidia-smi --query-compute-apps=pid,process_name,used_memory`：

- 若结果**非空**，立即以状态 `gpu_busy` 退出（返回码 2），**不会**终止或干扰
  任何已有用户进程。
- 若 `nvidia-smi` 不可用或查询失败，cuda 模式 **fail closed**：状态
  `gpu_check_failed`，非 0 退出，且不会启动任何 worker。无法确认空闲就不跑。
- 因此当设备上已有用户 Python 负载时，正式 GPU 对照会如实拒绝运行；此时不得
  据此推断任何 GPU 性能结论。
- 只有确认 GPU 空闲后，才可能进行真实的 GPU 性能测量。

`--provider cpu` 是**诊断用 smoke**：跳过 GPU 忙检查和 nvidia-smi 采样，并把
GPU 数据标记为“未采集”。多进程并发、barrier、真实退出码检查和原生统计逻辑
与 cuda 模式完全相同，但**不能**用 cpu 模式推断 GPU 性能。

## 命令

正式对照（总 70 个 game worker、总 2 个 Session，总 300 局，两轮交错）：

```bash
python python/experiments/multiprocess_selfplay/benchmark.py \
  --model path/to/model.onnx \
  --output-dir path/to/new_output_dir \
  --processes 1 2 \
  --total-workers 70 \
  --total-server-threads 2 \
  --games 300 \
  --repeats 2
```

- 1 进程：70 个对局线程、2 个 Session。
- 2 进程：每进程 35 个对局线程、1 个 Session。
- 轮次顺序为 `1, 2, 2, 1`（第 1 轮升序，第 2 轮降序）。

小规模示例（README 只作示例，正式测量以上面的 70 为准）：

```bash
python python/experiments/multiprocess_selfplay/benchmark.py \
  --model path/to/model.onnx --output-dir path/to/new_dir \
  --processes 1 2 --total-workers 50 --total-server-threads 2 \
  --games 300 --repeats 2
```

CPU 诊断 smoke（不采集 GPU 数据，不做 GPU 性能结论；2 局、2 simulations、
固定 board limit 4）：

```bash
python python/experiments/multiprocess_selfplay/benchmark.py \
  --model path/to/model.onnx --output-dir path/to/new_dir \
  --provider cpu --processes 1 2 --total-workers 2 \
  --total-server-threads 2 --games 2 --repeats 1 \
  --sims 2 --fast-sims 1 --max-depth 4 --board-min 4 --board-max 4
```

其他搜索参数默认与旧基准一致：`--sims 200 --fast-sims 50
--full-search-prob 0.25 --seed 20261023 --timeout 900 --cuda-device 0
--board-min 15 --board-max 25`。自对弈没有 `--max-game-length`：每局由采样到的
board limit 或吃王、无合法动作等自然终局结束。`--help` 可查看全部参数。

## 总预算与分配语义

`--total-workers` 和 `--total-server-threads` 都是**总量**，不是每进程量。
设某轮进程数为 `P`：

- `--total-server-threads` 必须能被每个 `P` 整除，每进程
  `total_server_threads // P` 个 Session；
- game worker 平均分给各进程，余数给前几个进程，总和保持不变；
- 局数按各进程 worker 数比例分配，余数保持总量；
- 要求 `--games >= --total-workers`，否则实际活跃 worker 数会少于总预算，
  比较将失去负载等价性，程序直接拒绝；
- 每个进程至少 1 个 worker。

因此“总 worker 数”和“总 Session 数”在整个实验轮次中保持不变，而不是每进程
相同。

## 指示口径

- **吞吐主分母**：每轮所有子进程真实调用区间的一个共享 span
  `max(call_end_epoch) - min(call_start_epoch)`。
  `samples/rows/inference rows` 先跨进程求和，再除以这一个 span。
  **绝不**除以各进程时间之和，也**绝不**把各进程自己的 rate 相加。
- `group_wall_seconds`：父进程观察到的整个子进程组 wall（从最早启动到最后
  退出），单独列出。
- `cpu_seconds_sum`：各子进程 `run_cpu_seconds` 之和，单独列出。
- `setup_seconds_raw / setup_seconds_sum / setup_seconds_max`：worker 原始
  setup 值，单独列出。
- `avg_batch_inference_per_ort = sum(inference_rows) / sum(ort_runs)`，即原生
  实际平均 batch，而不是配置假设。
- 每轮记录调用区间重叠：`overlap_seconds`、`overlap_fraction`。`P > 1` 时若
  真实调用区间没有交集（`max(call_start) >= min(call_end)`），该轮判定失败。
- GPU：每轮用 `nvidia-smi -lms 500` 连续采样，stdout 直接写
  `gpu_raw.csv`（不走 PIPE）；只保留 API union interval
  `[min(call_start), max(call_end)]` 内的样本，另写 `gpu_filtered.csv`。
  summary 报告的 `gpu_util_avg / gpu_util_max / gpu_mem_used_max_mib` 是
  **采样到的 GPU 利用率均值和最高值**，不是 SM occupancy；显存是**整卡**
  已用显存，不是单进程或单 Session 显存。
- `--provider cpu` 时不采样，`gpu.collected=false`，不得据此给出 GPU 结论。

## 种子与确定性

每个进程的 seed 由 `--seed` 派生为确定的 uint64 偏移，避免各进程重复同一批
对局。不同进程数与单进程运行**不是严格同轨迹**；只应把结果当作近似、负载
等价下的吞吐比较。

## 失败处理与退出码

- 父进程对每个子进程执行 `wait`，使用**真实退出码**。即使 `result.json` 的
  `status` 是 `ok`，只要解释器析构等导致真实退出码非 0，该轮仍失败。
- 所有子进程写完 `ready.json` 后，父进程才统一写该轮的 `go` 标记，使绑定导入
  完成后几乎同时开始。
- barriered 等待期间会轮询子进程状态；若某个子进程提前退出，立即失败，不会
  空等满 120 秒。
- 超时或失败时，只终止**本基准自己启动且仍活跃**的子进程；不触碰用户进程。
  所有子进程的 stdout/stderr、真实退出码和已有 result 都会保留。
- 失败也会写 `summary.json` / `summary.csv`，并以非 0 返回。
- `--output-dir` 必须不存在，禁止覆盖旧输出。
- 使用**该实验入口共享**的排他 lock（与 `--output-dir` 无关，不同输出目录也
  互斥），防止本实验自身并发；已存在的 lock 不会被删除或覆盖。
- 首个真实非 0 子进程出现时，立即终止/等待本实验其余仍活跃子进程，并保留
  所有真实退出码，不会继续空等到 `--timeout`。
- preflight 会记录 `_5dchess.pyd` 的 SHA256 以及 `src/5dchess` 相关源文件的
  mtime；若扩展缺失或比源文件旧，直接拒绝运行（避免旧 pyd 不含
  `num_server_threads`）。

## 输出结构

```
<output_dir>/
  manifest.json                 # 顶层清单：计划、冻结模型、preflight、GPU 状态
  summary.json                  # 每轮指标 + 失败原因 + GPU 采样汇总
  summary.csv                   # 每轮一行的扁平指标
  model/                        # 原 ONNX 与 sidecar 的 copy2 冻结副本
    freeze_manifest.json        # 源/副本 SHA256 校验
  runs/<pNN_repMM>/
    request.json                # 该轮请求（含每个 worker 的 binding_kwargs）
    result.json                 # 该轮每个 worker 的原始 result + 真实退出码
    stdout.log / stderr.log     # 各 worker 日志的完整拼接
    profile.jsonl               # 各 worker 原生 Session profile 合并（带 worker_index）
    gpu_raw.csv                 # nvidia-smi -lms 500 原始输出
    gpu_filtered.csv            # 仅 API union interval 内样本
    workers/wNN/
      request.json              # worker 契约请求
      ready.json / result.json  # worker 生命周期
      profile.jsonl             # 每 worker 的真实 Session profile
      stdout.log / stderr.log
```

源模型只读：冻结时执行 `copy2` 并校验“源前后 SHA256 == 副本 SHA256”，不会
重导出或覆盖源文件。

## VerySmall 4x4 实验参数

固定使用 `board_side=4`、`line_shift=5`、`board limit 15–25`、
`max batch 128`、`max depth 512`、`c_puct=2.0`、`dirichlet_alpha=0.3`、
`dirichlet_epsilon=0.25`、`temperature=1.0 / final=0.1 / threshold=30`、
`material_scale=2.0`，并开启 playout cap randomization。任一动作使棋盘数达到或
超过本局 board limit 时立即按物质分结束；若该小步同时吃王，吃王优先。

## 测试

CPU 单元测试（不加载 GPU，不启动真实 worker）：

```bash
python -m pytest python/tests/test_multiprocess_selfplay_benchmark.py -q
```

覆盖：总量分配守恒与非法进程数拒绝、轮次交错顺序、并发聚合的正确共享分母、
真实退出码非 0 即使 `result.status=ok` 也失败、源模型冻结不覆盖。
