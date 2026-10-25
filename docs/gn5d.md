# GN5D 记谱法规范（v2）

`.gn5d` 是 UTF-8 JSON 棋谱，记录 5D 国际象棋 `capture_king` 规则下从开局到停止位置的
完整行动轨迹。v2 是极简格式：**只保存单个开局 FEN、实际执行的行动、结果和生产元信息；
其余状态一律由原生规则引擎重放推导，不落盘。**

## 1. 容器与编码

- 扩展名 `.gn5d`，一个文件一局，媒体类型 `application/json`。
- JSON（RFC 8259），UTF-8，无 BOM；写入端使用 LF 换行并在文件末尾保留一个 LF。
- 写入端输出紧凑 JSON（`,`、`:` 后不加空格）；读取端必须接受任意空白与键序。
- 文档为封闭对象：出现未定义字段即视为非法棋谱。
- 写入端不得产生重复键；读取端按标准 JSON 语义处理。

## 2. 顶层结构

仅允许以下 8 个字段，全部必填：

```json
{
  "format": "gn5d",
  "version": 2,
  "rules": "capture_king",
  "variant": "very_small",
  "initial": "[nbrk/3p*/P*3/KRBN:0:1:w]\n",
  "actions": [[1, 0, 1, 0, 1, 3, 1, 0], "submit"],
  "result": {"outcome": 1.0, "terminal_reason": "capture_king"},
  "meta": {
    "iteration": 490,
    "game_id": 292900,
    "backend": "cpp_onnx",
    "board_limit": 16,
    "material_scale": 2.0
  }
}
```

| 字段 | 约束 |
| --- | --- |
| `format` | 固定字符串 `"gn5d"` |
| `version` | 固定整数 `2` |
| `rules` | 固定字符串 `"capture_king"`；`"strict"` 等其它规则不属于本版本 |
| `variant` | `"very_small"`、`"standard"` 或 `"standard_turn_zero"` |
| `initial` | 开局 FEN 字符串，见第 3 节 |
| `actions` | 行动数组，见第 4 节 |
| `result` | 结果对象，见第 5 节 |
| `meta` | 生产元信息，见第 6 节 |

### 派生数据禁令

以下内容**不得出现在文档中**，读端必须通过 `initial` + `actions` 重放得到：

- 棋盘宽高 `size`；
- 开局棋盘列表 `boards`；
- 当前行动方/当前棋盘 `present`；
- 时间线状态 `timeline_status`；
- 移动步数 `total_semimoves`；
- 最终局面 `final_fen`。

它们要么与 `initial` 完全重复，要么由引擎对棋盘集合确定性推导，要么是行动序列的计数。
把派生数据写入文件会让文件体积随棋局长度膨胀，并引入互相矛盾的可能。

## 3. 开局 `initial`

`initial` 是原生引擎 `show_fen()` 的单个字符串。每个棋盘一行：

```text
[<board-fen>:<l>:<t>:<c>]
```

- `<board-fen>`：标准棋盘 FEN，行用 `/` 分隔，每行是棋子字符或连续空格数字 `1`–`8`。
- 行数即棋盘高度，行宽即棋盘宽度；v2 只支持正方形棋盘，边长 `1..8`。
- 所有棋盘行必须同宽同高，且 `(l, t, c)` 唯一。
- `<l>`：时间线整数，可为负；`<t>`：时间坐标，非负整数；`<c>`：`w` 或 `b`。
- 未移动标记 `*` 只能紧跟在王、车、兵（`KkRrPpWw`）之后，表示该棋子尚未移动。
- 行之间用 LF 分隔；末尾 LF 可省略；读取端接受 CRLF。

示例：

```text
[r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*:0:0:b]
[r*nbqk*bnr*/p*p*.../R*NBQK*BNR*:0:1:w]
```

`initial` 是开局状态的唯一真源。当前行动方、当前棋盘、必需/可选时间线、棋盘列表都由
引擎从该 FEN 重建；读端不得信任任何额外的快照字段。

## 4. 行动 `actions`

元素只有两种形式：

1. 八维整数数组：`[x_from, y_from, t_from, l_from, x_to, y_to, t_to, l_to]`
2. 字符串 `"submit"`

```json
"actions": [
  [1, 0, 1, 0, 1, 3, 1, 0],
  "submit"
]
```

规则：

- 每个坐标必须是 JSON 整数（不接受浮点、布尔或字符串），绝对值不大于 `2^53-1`。
- `x`、`y` 从 0 开始且必须位于棋盘内；`t ≥ 0`；`l` 可为负。
- 坐标使用原生引擎值，不做时间线偏移编码；跨时间线、回到过去的分支由引擎生成。
- 每个数组必须与当前局面的合法动作**唯一匹配**；不保存枚举式 `action_id`。
- `"submit"` 是独立行动，表示结束当前回合；不计入移动步数，也不等价于移动。
- 行动按实际执行顺序排列，覆盖所有着法（包括快速搜索、强制提交和终局前未提交的最后一手）。
- 不自动插入或省略提交；自然终局后不得再有行动。
- 升变沿用引擎的自动升后规则，v2 不表示欠升变。

## 5. 结果 `result`

```json
"result": {"outcome": 1.0, "terminal_reason": "capture_king"}
```

- `outcome`：白方视角的有限数值，范围 `[-1, 1]`；物质分终局可以是非整数。
- `terminal_reason` 取值：
  `capture_king`、`material`、`checkmate`、`stalemate`、`no_legal_action`、
  `max_game_length`、`aborted`。
- 自然终局必须与原生重放一致；否则棋谱非法。
- 非自然终局只接受 `max_game_length` 或 `aborted`，且 `outcome` 必须为 `0.0`。

自然终局的 `result` 虽然可由重放推导，仍作为生产者的完整性声明保留：读端据此拒绝
“声称自然终局、重放却不终局”的残缺棋谱。截断局的 `terminal_reason` 无法从行动推导，
必须记录。

## 6. 元信息 `meta`

仅允许以下 5 个字段，全部必填：

| 字段 | 含义 |
| --- | --- |
| `iteration` | 训练迭代号，非负整数 |
| `game_id` | 局号，非负整数 |
| `backend` | 生产后端标签，字符串 |
| `board_limit` | 重放规则参数，正整数 |
| `material_scale` | 物质分缩放，非负有限数 |

`iteration`、`game_id`、`backend` 只用于溯源；`board_limit`、`material_scale` 是重放输入，
必须参与原生状态构造。

## 7. 读取与校验

读取分三级：

1. **L0 结构校验**：封闭字段、类型、取值范围、FEN 形状、坐标边界与棋盘尺寸。
   不依赖原生扩展。
2. **L1 原生重放**：用 `initial`、`board_limit`、`material_scale` 构造 `GameEnv`，
   依次匹配并执行行动，核对合法性与 `result`。v2 读取**必须**执行 L1。
3. **L2 完整性**：v2 不提供独立最终 FEN；最终局面由行动确定性恢复。

参考实现：

```python
from suzero_5dchess.game_record import load_record, replay_record, validate_record

record = load_record("game.gn5d")   # L0 + L1
state = replay_record(record)       # 恢复后的原生 GameEnv
final_snapshot = state.record_snapshot()
```

`load_record()` 返回原始精简文档，不注入推导字段。读取失败直接报错，不做部分信任。
`verify_cache` 只控制旧 v1 缓存校验；v2 始终重放。

## 8. 版本与旧格式

- `format == "gn5d"` 且 `version == 2` 为本规范。
- 读端必须拒绝高于 2 的版本，不得猜测字段。
- v1 旧格式仍可读：`format == "5dc.capture-king"`、`version == 1`。v1 的 `initial`
  是含 `size`/`boards`/`present`/`timeline_status` 的快照对象，`meta` 含
  `total_semimoves`，可含 `final_fen`，且可能带 `replay.steps` 棋盘缓存。
- v1 紧凑文档（无 `replay`）和旧 `.5dck` 缓存文档都由同一读取路径兼容；`replay` 字段
  存在即按缓存文档处理。
- 写入端默认只写 v2。`build_record(compact=False)` 仅供测试或旧消费者生成内存中的 v1 文档。

## 9. 写入端义务

1. 先完成 L1 校验再写文件；不得写出未经验证的行动轨迹。
2. 使用独占创建，不覆盖已有棋谱；失败时只清理本次生成的部分文件。
3. 按 `<terminal_reason>` 分类保存，文件名包含迭代号、局号和 UUID。
4. 每类按修改时间保留最新 `MAX_RECORDS_PER_CATEGORY`（默认 5000）盘。

文件布局、保留策略、异步写盘、UI 回放与 HTTP 接口属于实现约定，不属于本格式规范。
