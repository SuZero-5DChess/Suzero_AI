# 5D Chess Engine

An open-source C++ engine and game logic library for **5D Chess With Multiverse Time Travel**. It provides Monte Carlo tree search engines, legal action generation, checkmate detection, a web analyzer, and Python and JavaScript bindings.

## Features

- Analyze positions and explore branching game histories.
- Run search engines through the [5DUCI protocol](docs/5duci.md).
- Generate legal actions and detect checkmate using the [hypercuboid algorithm](docs/hypercuboid.md).
- Read custom starting positions and [branched 5DPGN games](docs/branched-5dpgn.md).
- Access game logic from C++, Python, or JavaScript.

Boards are limited to a maximum size of **8 × 8**. Engine strength and search methods remain under active development.

## Quick start (Analyze Games)

**[Try the web analyzer](https://ftxi.github.io/5dchess_engine/)**

## Quick start

Requires **CMake 3.20 or newer** and a compiler with **C++23 support**.

From the repository root:

```sh
cmake -S . -B build -DENGINE=ON -DTOOLS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

This builds the `5dchess` engine and `5dtools` utilities. Use a Release build for search performance.

Start an MCTS engine:

```sh
./build/5dchess mcts
```

The engine accepts [5DUCI](docs/5duci.md) commands on stdin. Enter the commands below in order, waiting for `5duciok` after `5duci` and `readyok` after `isready` before continuing:

```text
5duci
isready
position startpos
go movetime 1000
```

Possible output:

```text
5duciok
readyok
bestmove (0T1)g1f3
```

The engine also prints identification, option, and search information. The move line starts with `bestmove`, followed by the selected move or action.

With a multi-configuration generator such as Visual Studio, executables are under `build/Release/`; on Windows, they have an `.exe` suffix.

See the [build guide](docs/usage/building.md) for platform details, Python bindings, and WebAssembly builds.

## Documentation

| Task | Guide |
|---|---|
| Build native programs, Python bindings, or the web interface | [Building](docs/usage/building.md) |
| Choose an engine, configure search, or run autoplay matches | [Engine usage](docs/engine/usage.md) |
| Inspect positions and enumerate legal actions | [Command-line tools](docs/usage/tools.md) |
| Manage engine ratings and schedule matches | [Elo registry and matchmaker](docs/engine/elo-matchmaker.md) |
| Understand the linear evaluation model | [Linear features](docs/engine/linear-features.md) |
| Understand game logic and client design | [Core architecture](docs/architecture/core.md) · [Client architecture](docs/architecture/client.md) |
| Understand the web interface | [Web UI architecture](docs/architecture/webui.md) |
| Integrate an engine or read game notation | [5DUCI](docs/5duci.md) · [Branched 5DPGN](docs/branched-5dpgn.md) |

See the [documentation guide](docs/README.md) for algorithm explanations, formal grammar, and experimental reports.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development conventions, testing expectations, and documentation guidelines.

## Attribution

All resources in this project are open source or created by the project author. This is an independent project and is not affiliated with Thunkspace, LLC. It contains no copied or decompiled source code from the commercial game **5D Chess With Multiverse Time Travel**.
