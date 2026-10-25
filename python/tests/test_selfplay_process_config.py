"""Tests for the optional multiprocess self-play configuration.

These tests stay CPU-only and do not launch workers: they cover the dataclass
contract, the CLI mapping plus nested validation, checkpoint compatibility and
the Windows launcher wiring.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from suzero_5dchess.config import SelfPlayConfig, TrainConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "run_train.bat"


def _valid(**overrides: object) -> SelfPlayConfig:
    values: dict[str, object] = {
        "num_games": 4,
        "workers": 4,
        "num_server_threads": 2,
        "backend": "cpp_onnx",
        "processes": 2,
    }
    values.update(overrides)
    return SelfPlayConfig(**values)  # type: ignore[arg-type]


def test_default_processes_is_one() -> None:
    assert SelfPlayConfig().processes == 1


@pytest.mark.parametrize("value", [0, -1, -8])
def test_rejects_non_positive_processes(value: int) -> None:
    with pytest.raises(ValueError, match="processes"):
        SelfPlayConfig(processes=value)


def test_processes_above_one_requires_cpp_onnx() -> None:
    with pytest.raises(ValueError, match="cpp_onnx"):
        SelfPlayConfig(processes=2)


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"workers": 1}, "workers"),
        ({"num_server_threads": 1}, "num_server_threads"),
        ({"num_games": 1}, "num_games"),
    ],
)
def test_processes_above_one_requires_total_budgets(
    overrides: dict[str, object], needle: str
) -> None:
    with pytest.raises(ValueError, match=needle):
        _valid(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        # 50 workers / 3 server threads over 2 processes split as 25+25 and
        # 2+1; no divisibility is required.
        {"workers": 50, "num_server_threads": 3, "num_games": 300},
        {"workers": 5, "num_server_threads": 3, "num_games": 5},
        {"workers": 5, "num_server_threads": 3, "num_games": 4, "processes": 3},
    ],
)
def test_uneven_total_budgets_are_allowed_and_not_rescaled(
    overrides: dict[str, object],
) -> None:
    config = _valid(**overrides)

    assert config.workers == overrides.get("workers", 4)
    assert config.num_server_threads == overrides.get(
        "num_server_threads", 2
    )
    assert config.num_games == overrides.get("num_games", 4)
    assert config.processes == overrides.get("processes", 2)


def test_cli_maps_flag_to_nested_self_play_config() -> None:
    from suzero_5dchess.cli import _build_config, build_parser

    args = build_parser().parse_args(
        [
            "train",
            "--selfplay-processes",
            "2",
            "--selfplay-backend",
            "cpp_onnx",
            "--workers",
            "50",
            "--num-server-threads",
            "3",
            "--games",
            "300",
        ]
    )
    assert args.selfplay_processes == 2
    self_play = _build_config(args).self_play
    assert self_play.processes == 2
    assert self_play.backend == "cpp_onnx"
    assert self_play.workers == 50
    assert self_play.num_server_threads == 3
    assert self_play.num_games == 300


def test_cli_default_leaves_single_process_path() -> None:
    from suzero_5dchess.cli import _build_config, build_parser

    args = build_parser().parse_args(["train"])
    assert args.selfplay_processes is None
    assert _build_config(args).self_play.processes == 1


@pytest.mark.parametrize(
    ("argv", "needle"),
    [
        # The default backend is python.
        (["train", "--selfplay-processes", "2"], "cpp_onnx"),
        # workers default to 1 while processes is 2.
        (
            [
                "train",
                "--selfplay-processes",
                "2",
                "--selfplay-backend",
                "cpp_onnx",
            ],
            "workers",
        ),
        # Server threads default to 1 while processes is 2.
        (
            [
                "train",
                "--selfplay-processes",
                "2",
                "--selfplay-backend",
                "cpp_onnx",
                "--workers",
                "2",
            ],
            "num_server_threads",
        ),
        # Games must also cover every process.
        (
            [
                "train",
                "--selfplay-processes",
                "2",
                "--selfplay-backend",
                "cpp_onnx",
                "--workers",
                "2",
                "--num-server-threads",
                "2",
                "--games",
                "1",
            ],
            "num_games",
        ),
    ],
)
def test_cli_nested_validation_rejects_invalid_multiprocess_budget(
    argv: list[str], needle: str
) -> None:
    from suzero_5dchess.cli import _build_config, build_parser

    args = build_parser().parse_args(argv)
    # TrainConfig.__post_init__ re-runs SelfPlayConfig.__post_init__, so the
    # CLI path must surface the same error as the dataclass.
    with pytest.raises(ValueError, match=needle):
        _build_config(args)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_rejects_non_positive_processes(value: str) -> None:
    from suzero_5dchess.cli import _build_config, build_parser

    args = build_parser().parse_args(
        ["train", "--selfplay-processes", value]
    )
    with pytest.raises(ValueError, match="processes"):
        _build_config(args)


def test_dict_round_trip_and_old_checkpoint_default() -> None:
    config = TrainConfig(self_play=_valid(processes=2))

    data = config.to_dict()
    assert data["self_play"]["processes"] == 2
    restored = TrainConfig.from_dict(data)
    assert restored.self_play.processes == 2

    # Checkpoints written before the field existed fall back to 1.
    del data["self_play"]["processes"]
    old = TrainConfig.from_dict(data)
    assert old.self_play.processes == 1


def test_launcher_wires_processes_and_keeps_debug_semantics() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert any(
        line.startswith('set "SELFPLAY_PROCESSES=') for line in text.splitlines()
    )
    assert "--selfplay-processes %SELFPLAY_PROCESSES%" in text
    assert "processes=%SELFPLAY_PROCESSES%" in text

    # Existing knobs remain wired through their variables rather than being
    # pinned to specific tuning values; users may freely edit the defaults.
    assert "--games %SELFPLAY_GAMES%" in text
    assert "--workers %SELFPLAY_WORKERS%" in text
    assert "--num-server-threads %NUM_SERVER_THREADS%" in text
    assert 'set "CONTINUOUS_ARG=--continuous"' in text

    # The debug preset still replaces the default budget and forwards only
    # explicit extra arguments.
    assert (
        'if defined DEBUG_MODE set "DEFAULT_ARGS=--record-games %CONTINUOUS_ARG%"'
        in text
    )
    # Threads are not documented as guaranteeing larger batches, and the
    # worker comment does not imply a cross-process shared evaluator queue.
    assert "More threads can batch more" not in text
    assert "they all share the ONNX evaluator" not in text
