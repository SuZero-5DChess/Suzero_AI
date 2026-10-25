# 第三方开源声明 / Third-Party Notices

本项目是一个**重新实现的、与游戏无关的 PUCT 图搜索核心**。其算法、参数语义和统计量
设计源自 KataGo 的搜索模块；可选的 Python 训练器也重新实现了 KataGo 的训练调度与
裁剪方法。本项目不包含 KataGo 源码树中的任何文件，也不包含其第三方依赖
（CUDA、OpenCL、cutlass、nlohmann_json 等均未复制）。

---

## KataGo

- 项目主页：<https://github.com/lightvector/KataGo>
- 本项目参考/派生的源文件（上游路径）：
  - `cpp/search/search.h`
  - `cpp/search/search.cpp`
  - `cpp/search/searchnode.h`
  - `cpp/search/searchparams.h`
  - `cpp/search/searchhelpers.cpp`
  - `cpp/search/searchexplorehelpers.cpp`
  - `cpp/search/searchupdatehelpers.cpp`
  - `cpp/search/searchresults.cpp`
  - `cpp/search/searchnodetable.h`
  - `cpp/search/searchnodetable.cpp`
  - `cpp/game/graphhash.h`
  - `cpp/game/graphhash.cpp`
- 派生方式：将 Go 专用类型（`Board`、`BoardHistory`、`Loc`、`NNOutput`、玩家颜色、
  贴目、领地等）替换为抽象接口，重新组织为与游戏无关的 C++17 代码。PUCT 公式、
  FPU、根节点 Dirichlet 噪声、访问权重聚合、温度采样、图搜索的节点表/边访问
  追赶/成环与换根处理等算法语义保持与上游一致。
- 许可证：MIT

### KataGo 训练方法（可选 Python 训练器）

参考的上游路径：`python/train.py`（及其调用的训练辅助逻辑）。

派生方式：没有复制任何 Python 源文件。`python/suzero_train/pytorch/` 用
PyTorch 重新实现了以下方法：

- per-sample 学习率（基准 `3e-5`，AdamW 乘 `1.33`）与 batch 缩放
  （AdamW 指数 0.5，SGD 指数 1.0）；
- 按样本数分段 warmup（上游前 2,000,000 样本的 1/20 … 1/1.4 曲线）；
- 按参数组的学习率/weight decay 系数；
- 全局梯度范数裁剪上限（随 batch 与学习率缩放）；
- `_sum` / `_batch` 指标的累计与报告、指数衰减的运行指标；
- model / optimizer / scaler / metrics / train state 的 checkpoint 结构。

未移植：KataGo 的围棋模型、Board 编码、`.npz` 自对弈数据格式、导出流程、
DDP/多 GPU、Muon/NorMuon/Aurora、SWA、lookahead、`torch.compile`、LR cycle，
以及按 normalization 类型选择 weight decay 的模型相关表。

### KataGo MIT License（原文）

```
Copyright 2025 David J Wu ("lightvector") and/or other authors of the content in this repository.
(See 'CONTRIBUTORS' file for a list of authors as well as other indirect contributors).

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

KataGo 仓库根目录的 `LICENSE` 还说明：其随附的若干第三方库/文件适用各自的许可证。
**本项目没有复制这些内容**。如果将来直接把 KataGo 的某个源文件或第三方库引入本项目，
必须同时遵循该文件/库自身的许可证，而不仅是上述 MIT 文本。

---

## 5dchess_engine（可选 5D 适配层）

- 项目主页：<https://github.com/ftxi/5dchess_engine>
- 用途：`src/5dchess/game_state.cpp` 通过公开 API 使用该引擎的棋规与
  走法生成；默认源码树为本项目内的 `5dchess_engine-main/`，也可通过
  `SUZERO_5DCHESS_ENGINE_ROOT` 指定其他源码树。
- `run_ui.bat` 直接使用 `5dchess_engine-main/ui/` 自带的页面、Canvas、主题、
  导航控件和资源。新增 `gn5d-worker.js`、`gn5d-replay.js` 为原生重放的协议适配，
  并对 UI 导入扩展名、导出格式及只读状态做小范围适配；没有修改引擎核心。
- 许可证：BSD 2-Clause，Copyright (c) 2024-2026, ftxi。
- 引擎许可证见 `5dchess_engine-main/LICENSE`。SVG 棋子、图标及字体的独立许可证
  和归属声明保留在 `ui/assets/`、`ui/icons/` 和 `ui/fonts/` 中，分发时一并保留。

## ONNX Runtime（可选 5D 适配层）

- 项目主页：<https://github.com/microsoft/onnxruntime>
- 用途：`src/5dchess/onnx_evaluator.cpp` 使用其 C++ API 做同步推理。
  本项目不复制其源码，构建时链接用户提供的预编译 SDK。
- 许可证：MIT，Copyright (c) Microsoft Corporation。
- SDK 中的 `LICENSE` 与 `ThirdPartyNotices.txt` 必须随其分发一并保留。

## very_small ONNX 模型

- `models/5dchess/very_small_latest_fp32.onnx` 由 5dc_ml 项目
  （<https://github.com/ftxi/5dc_ml>，BSD 2-Clause，Copyright (c) 2024-2026,
  ftxi）导出的 `very_small` 检查点生成，仅用于本项目的集成与验收测试。
- 模型输入/输出协议由同目录的 `*.onnx.json` sidecar 描述。

---

## 5dc_ml 旧模型推理兼容

- 参考项目：<https://github.com/ftxi/5dc_ml>，BSD 2-Clause，Copyright (c)
  2024-2026, ftxi。
- `python/suzero_5dchess/legacy_model.py` 根据 `alphazero/network_smallstep.py`、
  `network_common.py`、`model_spec.py` 的网络拓扑与 checkpoint 协议重新实现
  LSTM 推理适配；共享本项目编码器，不复制参考项目源码文件。
- 固定 `very_small_old_version_v1.pt` 由使用者从本地路径提供，不随新增代码复制、
  改写或重新训练；其模型分发应保留原项目许可证。

---

## 本项目

- 新增代码：MIT，见 [LICENSE](LICENSE)。
- 搜索核心**不依赖**任何游戏引擎或神经网络；`tests/` 下的玩具游戏仅用于
  验证抽象接口和搜索算法。5D 适配层是可选的独立目标，不改变核心的
  游戏无关性。
- Python 训练子系统与 C++ 目标完全解耦；`python/suzero_train/` 的代码
  均为新实现，未复制 KataGo 的上游文件。
