# 5dc_ml — 5D Chess AlphaZero 训练项目

## 项目架构

### 游戏引擎（不可修改）

游戏引擎是独立的部分，**不要去修改**。引擎代码位于：
- `src/` — 核心引擎源码（core/、engine/、client/、misc/、tools/）
- `5dchess_engine-main/` — 引擎的独立仓库副本（submodule）

所有对引擎的交互必须在外层封装和调用：
- `alphazero/env.py` — `SemimoveEnv` 封装引擎，提供 semimove 级别的接口
- `alphazero/semimove_api.py` — 轻量级 semimove 抽象 API
- `alphazero/halfstep_state.py` — halfstep 级别的状态封装
- `pymodule.cpp` — Python 绑定层（编译为 `engine.pyd`）
- `alphazero/__init__.py` — 自动从 `build_py_ml` / `build_py` / `build` 导入编译好的 `engine` 模块

### Python ML 代码

- `alphazero/` — AlphaZero 训练框架（PyTorch 网络、MCTS、自对弈、训练）
- `ui/` — 前端界面
- `host.py` / `serve_ui.py` — Web 服务

## 行棋术语定义

### 坐标维度

- **4D 坐标** (`Coord4`)：`Tuple[int, int, int, int]` = `(x, y, t, l)` — 棋盘上的一个格点
- **8D 坐标** (`MoveKey`)：`Tuple[int, int, int, int, int, int, int, int]` — 一个完整移动（from_pos + to_pos）

### 层级关系

```
大步 (big step) = 一行棋方的完整行动
  └── 一组走子小步 (move small steps) + 一个 submit
        └── 每个走子小步 = 一个选择棋子的半步 + 一个选择落点的半步

半步 (half step) = 一个 submit | 一个 4D 坐标（选择棋子或选择落点）
小步 (small step) = 一个 submit | 一个 8D 坐标（包含选择棋子和选择落点）
```

详细说明：

| 术语 | 英文 | 定义 | 代码类型 |
|------|------|------|----------|
| **半步** | half step | 一个 submit，**或者**选择棋子的 4D 坐标，**或者**选择落点的 4D 坐标 | `Union[Coord4, str]`（`"SUBMIT"`） |
| **小步** | small step | 一个 submit，**或者**一个同时包含选择棋子和选择落点的 8D 坐标 | `MoveKey` = `Tuple[int,int,int,int,int,int,int,int]` |
| **走子小步** | move small step | 由**选择棋子的半步** + **选择落点的半步**组成的一个完整移动 | `Semimove`（from_pos + to_pos） |
| **大步** | big step | 一行棋方的完整行动 = **一组走子小步** + **一个 submit** | `ActionMoveList` / 多次 `apply_semimove` + `submit_turn` |

- 一个走子小步 = 2 个半步（选择棋子 half step + 选择落点 half step）
- 一个大步 = N 个走子小步 + 1 个 submit = 2N+1 个半步
- 当 `pending_from is None` 时，半步选择的是**棋子来源**；当 `pending_from` 已设置时，半步选择的是**落点目标**

## 开发规范

1. 在 git commit 时，使用中文，按照约定式提交格式进行。
2. 新建项目时，只要合适，就创建 git 存储库，方便追踪项目变更。
3. 如无特殊说明，项目代码应当保持**高内聚、低耦合、模块化**。

## 技术文档维护

维护一个技术文档 `TECHNICAL.md`，每次对项目代码做了任何修改之后，在 `TECHNICAL.md` 里同步更新对应的技术描述和变更记录，确保文档始终反映代码的当前状态。

## 重要：对 AI 生成代码保持怀疑

本项目大量代码由 AI 生成。**不要默认代码是正确的或合理的**——它完全可能是瞎写的。在分析、修改或引用现有代码时，必须：

1. 追踪完整数据流，确认语义对齐，不轻信函数名和注释
2. 检查死参数、未使用的返回值、签名与实现不一致的地方
3. 对"设计选择"保持怀疑——可能是疏忽或 bug，不是有意为之
4. 测试覆盖不足的地方，AI 生成代码的 bug 率更高