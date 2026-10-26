"""Tests for the ``num_server_threads`` self-play plumbing.

The binding is faked: these tests must run without a rebuilt C++ extension.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

from suzero_5dchess.config import MctsConfig, SelfPlayConfig, TrainConfig
from suzero_5dchess.protocol import VERY_SMALL


def test_default_is_one() -> None:
    assert SelfPlayConfig().num_server_threads == 1


@pytest.mark.parametrize("value", [0, -1, -8])
def test_rejects_non_positive(value: int) -> None:
    with pytest.raises(ValueError, match="num_server_threads"):
        SelfPlayConfig(num_server_threads=value)


def test_cli_parses_and_maps_to_self_play() -> None:
    from suzero_5dchess.cli import _build_config, build_parser

    args = build_parser().parse_args(["train", "--num-server-threads", "3"])
    assert args.num_server_threads == 3
    assert _build_config(args).self_play.num_server_threads == 3

    default_args = build_parser().parse_args(["train"])
    assert default_args.num_server_threads is None
    # Unset CLI flags leave the dataclass default untouched.
    assert _build_config(default_args).self_play.num_server_threads == 1


@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_non_positive_is_rejected_by_nested_validation(value: str) -> None:
    from suzero_5dchess.cli import _build_config, build_parser

    args = build_parser().parse_args(
        ["train", "--num-server-threads", value]
    )
    # TrainConfig.__post_init__ calls SelfPlayConfig.__post_init__, so the
    # invalid value must surface through _build_config, not just the dataclass.
    with pytest.raises(ValueError, match="num_server_threads"):
        _build_config(args)


def test_train_config_dict_round_trip_preserves_field() -> None:
    config = TrainConfig(
        self_play=SelfPlayConfig(num_server_threads=3, backend="cpp_onnx")
    )

    data = config.to_dict()
    assert data["self_play"]["num_server_threads"] == 3

    restored = TrainConfig.from_dict(data)
    assert restored.self_play.num_server_threads == 3

    # Checkpoints written before the field existed still load with the default.
    del data["self_play"]["num_server_threads"]
    old = TrainConfig.from_dict(data)
    assert old.self_play.num_server_threads == 1


class FakeBinding:
    """Records ``run_selfplay`` keyword arguments.

    With ``reject_server_threads=True`` it emulates an extension built before
    the parameter existed.
    """

    TREE_POSITION_FORMAT_VERSION = 1

    def __init__(self, *, reject_server_threads: bool = False) -> None:
        self.reject_server_threads = reject_server_threads
        self.calls: list[dict[str, Any]] = []

    def run_selfplay(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.reject_server_threads and "num_server_threads" in kwargs:
            raise TypeError(
                "run_selfplay() got an unexpected keyword argument "
                "'num_server_threads'"
            )
        return _empty_raw()


def _empty_raw() -> dict[str, Any]:
    return {
        "board_planes": np.zeros((0, 1, 12, 16), dtype=np.float32),
        "l_coords": np.zeros((0, 1), dtype=np.int64),
        "t_coords": np.zeros((0, 1), dtype=np.int64),
        "used_board_counts": np.zeros((0,), dtype=np.int64),
        "urgency": np.zeros((0,), dtype=np.float32),
        "side_to_move": np.zeros((0,), dtype=np.int64),
        "actions": np.zeros((0, 1, 8), dtype=np.int64),
        "action_counts": np.zeros((0,), dtype=np.int64),
        "policy": np.zeros((0, 1), dtype=np.float32),
        "has_submit": np.zeros((0,), dtype=bool),
        "value": np.zeros((0,), dtype=np.float32),
        "player": np.zeros((0,), dtype=np.int64),
        "move_count": np.zeros((0,), dtype=np.int64),
        "game_index": np.zeros((0,), dtype=np.int64),
        "games": [
            {
                "index": 0,
                "white_outcome": 0.0,
                "terminal_reason": "material",
                "semimoves": 1,
                "turns": 0,
                "board_limit": 2,
                "board_count": 2,
                "final_player": 0,
            }
        ],
        "stats": {
            "total_seconds": 0.0,
            "setup_seconds": 0.0,
            "batches": 0,
            "rows": 0,
            "server_threads": 1,
        },
    }


def _generate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake: FakeBinding,
    config: SelfPlayConfig,
):
    from suzero_5dchess import cpp_self_play

    monkeypatch.setattr(cpp_self_play, "_5dchess", fake)
    monkeypatch.setattr(cpp_self_play, "_require_binding", lambda: None)
    monkeypatch.setattr(
        cpp_self_play,
        "_export_model",
        lambda _network, _spec, model_path: model_path,
    )
    return cpp_self_play.generate_games(
        object(),
        VERY_SMALL,
        MctsConfig(num_simulations=4),
        config,
        num_games=1,
        model_path=tmp_path / "model.onnx",
    )


def test_python_call_forwards_non_default_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeBinding()
    config = SelfPlayConfig(
        num_games=1, workers=3, num_server_threads=4, backend="cpp_onnx"
    )

    _generate(monkeypatch, tmp_path, fake, config)

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["num_server_threads"] == 4
    assert "max_game_length" not in call
    # The game-thread knob stays distinct from the server threads.
    assert call["num_threads"] == 3


def test_default_is_omitted_for_old_extension_compatibility(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeBinding(reject_server_threads=True)
    config = SelfPlayConfig(num_games=1, backend="cpp_onnx")

    # Would raise if the kwarg leaked into a pre-num_server_threads build.
    _generate(monkeypatch, tmp_path, fake, config)

    assert "num_server_threads" not in fake.calls[0]


def test_old_extension_rejects_non_default_loudly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeBinding(reject_server_threads=True)
    config = SelfPlayConfig(
        num_games=1, num_server_threads=2, backend="cpp_onnx"
    )

    with pytest.raises(RuntimeError, match="rebuild"):
        _generate(monkeypatch, tmp_path, fake, config)
