# 第三方开源声明 / Third-Party Notices

本项目是一个**重新实现的、与游戏无关的 PUCT/MCTS 核心**。其算法、参数语义和统计量设计
源自 KataGo 的搜索模块。本项目不包含 KataGo 源码树中的任何第三方依赖（CUDA、OpenCL、
cutlass、nlohmann_json 等均未复制）。

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
- 派生方式：将 Go 专用类型（`Board`、`BoardHistory`、`Loc`、`NNOutput`、玩家颜色、
  贴目、领地等）替换为抽象接口，重新组织为与游戏无关的 C++17 代码。PUCT 公式、
  FPU、根节点 Dirichlet 噪声、访问权重聚合、温度采样等算法语义保持与上游一致。
- 许可证：MIT

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

## 本项目

- 新增代码：MIT，见 [LICENSE](LICENSE)。
- 本项目**未**接入任何游戏引擎，也不包含任何棋规实现；`tests/` 下的玩具游戏仅用于
  验证抽象接口和搜索算法。
