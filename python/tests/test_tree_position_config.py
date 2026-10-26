"""Tree-position configuration, backend plumbing and logging regressions."""

from types import SimpleNamespace

import numpy as np
import pytest

from suzero_5dchess.config import MctsConfig, SelfPlayConfig, TrainConfig
from suzero_5dchess.protocol import VERY_SMALL


TREE_DEFAULTS = {
    "record_tree_positions": True,
    "record_tree_min_visits": 64,
    "record_tree_max_depth": 5,
    "record_tree_max_per_turn": 4,
    "record_tree_sample_probability": 0.25,
}


def test_defaults_and_old_checkpoint_round_trip():
    config = TrainConfig()
    for name, value in TREE_DEFAULTS.items():
        assert getattr(config.self_play, name) == value
    config.self_play.record_tree_positions = True
    config.self_play.record_tree_min_visits = 12
    data = config.to_dict()
    assert TrainConfig.from_dict(data).self_play == config.self_play
    for name in TREE_DEFAULTS:
        del data["self_play"][name]
    old = TrainConfig.from_dict(data)
    for name, value in TREE_DEFAULTS.items():
        assert getattr(old.self_play, name) == value


@pytest.mark.parametrize("field", [
    "record_tree_min_visits", "record_tree_max_depth", "record_tree_max_per_turn",
])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, float("nan"), 2**31])
def test_invalid_limits(field, value):
    with pytest.raises(ValueError, match=field):
        SelfPlayConfig(**{field: value})


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_probability(value):
    with pytest.raises(ValueError, match="record_tree_sample_probability"):
        SelfPlayConfig(record_tree_sample_probability=value)


def test_depth_and_enable_validation():
    with pytest.raises(ValueError, match="record_tree_max_depth"):
        SelfPlayConfig(record_tree_max_depth=65)
    with pytest.raises(ValueError, match="record_tree_positions"):
        SelfPlayConfig(record_tree_positions=1)
    for probability in (0.0, 1.0):
        SelfPlayConfig(record_tree_sample_probability=probability)


def test_cli_mapping_and_nested_validation():
    from suzero_5dchess.cli import _build_config, build_parser

    default_config = _build_config(build_parser().parse_args(["train"])).self_play
    assert default_config.record_tree_positions is True
    args = build_parser().parse_args([
        "train", "--record-tree-positions", "--tree-min-visits", "16",
        "--tree-max-depth", "3", "--tree-max-per-turn", "2",
        "--tree-sample-prob", "0.5",
    ])
    config = _build_config(args).self_play
    assert config.record_tree_positions
    assert config.record_tree_min_visits == 16
    assert config.record_tree_max_depth == 3
    assert config.record_tree_max_per_turn == 2
    assert config.record_tree_sample_probability == 0.5
    disabled = _build_config(build_parser().parse_args([
        "train", "--no-record-tree-positions",
    ])).self_play
    assert disabled.record_tree_positions is False
    args.tree_sample_prob = float("nan")
    with pytest.raises(ValueError, match="record_tree_sample_probability"):
        _build_config(args)


def _empty_raw():
    return {
        "board_codes": np.empty((0, 16), dtype=np.uint8),
        "l_coords": np.empty(0, dtype=np.int64),
        "t_coords": np.empty(0, dtype=np.int64),
        "board_offsets": np.array([0], dtype=np.int64),
        "action_offsets": np.array([0], dtype=np.int64),
        "actions": np.empty((0, 8), dtype=np.int64),
        "policy": np.empty(0, dtype=np.float32),
        **{name: np.empty(0, dtype=np.int64) for name in (
            "used_board_counts", "action_counts", "has_submit", "urgency",
            "side_to_move", "value", "player", "move_count", "game_index",
        )},
        "games": [],
    }


def _generate(monkeypatch, tmp_path, config, *, supported=True):
    from suzero_5dchess import cpp_self_play

    calls = []
    def run(**kwargs):
        calls.append(kwargs)
        return _empty_raw()

    binding = SimpleNamespace(run_selfplay=run)
    if supported:
        binding.TREE_POSITION_FORMAT_VERSION = 1
    monkeypatch.setattr(cpp_self_play, "_5dchess", binding)
    monkeypatch.setattr(cpp_self_play, "_require_binding", lambda: None)
    monkeypatch.setattr(cpp_self_play, "_export_model", lambda n, s, p: p)
    cpp_self_play.generate_games(
        object(), VERY_SMALL, MctsConfig(num_simulations=2), config,
        model_path=tmp_path / "tree.onnx",
    )
    return calls


def test_native_default_enables_and_forwards_config(monkeypatch, tmp_path):
    config = SelfPlayConfig()
    calls = _generate(monkeypatch, tmp_path, config)
    for name in TREE_DEFAULTS:
        assert calls[0][name] == getattr(config, name)


def test_native_disabled_omits_kwargs_for_old_extension(monkeypatch, tmp_path):
    config = SelfPlayConfig(record_tree_positions=False)
    calls = _generate(monkeypatch, tmp_path, config, supported=False)
    assert not set(TREE_DEFAULTS).intersection(calls[0])


def test_native_enabled_forwards_config(monkeypatch, tmp_path):
    config = SelfPlayConfig(record_tree_positions=True, record_tree_min_visits=7)
    calls = _generate(monkeypatch, tmp_path, config)
    for name in TREE_DEFAULTS:
        assert calls[0][name] == getattr(config, name)


def test_old_extension_fails_before_export_by_default(monkeypatch, tmp_path):
    from suzero_5dchess import cpp_self_play

    monkeypatch.setattr(cpp_self_play, "_5dchess", SimpleNamespace(run_selfplay=lambda **k: None))
    monkeypatch.setattr(cpp_self_play, "_require_binding", lambda: None)
    def fail(*args):
        pytest.fail("export must not run for an unsupported extension")
    monkeypatch.setattr(cpp_self_play, "_export_model", fail)
    with pytest.raises(RuntimeError, match="tree-position.*rebuilt"):
        cpp_self_play.generate_games(
            object(), VERY_SMALL, MctsConfig(),
            SelfPlayConfig(), model_path=tmp_path / "bad.onnx",
        )


def test_tree_sample_summary_and_tensorboard():
    from suzero_5dchess.pipeline import _log_tensorboard, _self_play_summary
    from suzero_5dchess.self_play import SelfPlayResult

    result = SelfPlayResult(
        samples=[None] * 5, white_outcome=0.0, terminal_reason="material",
        semimoves=3, turns=2, board_limit=4, board_count=4, final_player=0,
        tree_samples=2,
    )
    summary = _self_play_summary([result])
    assert summary["samples"] == 5
    assert summary["played_samples"] == 3
    assert summary["tree_samples"] == 2
    assert summary["avg_board_count_squared"] == 16.0
    calls = {}
    writer = SimpleNamespace(add_scalar=lambda tag, value, step: calls.update({tag: value}))
    _log_tensorboard(writer, {"self_play": summary}, eval_state={}, done=1)
    assert calls["SelfPlay/tree_samples"] == 2
    assert calls["SelfPlay/played_samples"] == 3
    assert calls["SelfPlay/avg_board_count_squared"] == 16.0
    assert _self_play_summary([])["tree_samples"] == 0
    assert _self_play_summary([])["avg_board_count_squared"] == 0.0


def test_board_count_squared_average_is_sum_of_squares_over_games():
    from suzero_5dchess.pipeline import _self_play_summary
    from suzero_5dchess.self_play import SelfPlayResult

    def result(board_count: int) -> SelfPlayResult:
        return SelfPlayResult(
            samples=[], white_outcome=0.0, terminal_reason="material",
            semimoves=0, turns=0, board_limit=8, board_count=board_count,
            final_player=0,
        )

    # sum(board_count^2)/n, not (mean board_count)^2.
    summary = _self_play_summary([result(2), result(4), result(6)])
    assert summary["avg_board_count_squared"] == pytest.approx((4 + 16 + 36) / 3)


def test_launcher_tree_knobs_default_on_and_can_be_disabled():
    from pathlib import Path

    launcher = (Path(__file__).resolve().parents[2] / "run_train.bat").read_text(encoding="utf-8")
    assert 'if not defined TREE_POSITIONS set "TREE_POSITIONS=1"' in launcher
    for flag in ("--record-tree-positions", "--tree-min-visits", "--tree-max-depth",
                 "--tree-max-per-turn", "--tree-sample-prob"):
        line = next(line for line in launcher.splitlines() if flag in line)
        assert 'if "%TREE_POSITIONS%"=="1"' in line
    assert (
        'if "%TREE_POSITIONS%"=="0" set "DEFAULT_ARGS=%DEFAULT_ARGS% '
        '--no-record-tree-positions"'
    ) in launcher
