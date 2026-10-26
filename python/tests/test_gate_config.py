"""Gate defaults, validation, CLI mapping and launcher opt-in behavior."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import subprocess
import sys

import pytest

from suzero_5dchess.config import GateConfig, TrainConfig, debug_train_config


DEFAULTS = {
    "gate_interval": 1,
    "gate_games": 100,
    "gate_workers": 20,
    "gate_simulations": 200,
    "gate_min_average_score": 0.05,
    "gate_min_finished": 50,
    "gate_max_boards": 50,
    "gate_provider": "cuda",
    "gate_temperature": 0.1,
}


def test_defaults_enable_gate_except_tiny_debug_preset():
    assert asdict(GateConfig()) == DEFAULTS
    assert {name: getattr(TrainConfig(), name) for name in DEFAULTS} == DEFAULTS
    tiny = {**DEFAULTS, "gate_interval": 0}
    assert {name: getattr(debug_train_config(), name) for name in tiny} == tiny


@pytest.mark.parametrize("factory", [GateConfig, TrainConfig])
@pytest.mark.parametrize("value", [-1, 0.5, float("nan"), True, "1"])
def test_interval_is_always_validated(factory, value):
    with pytest.raises(ValueError, match="gate_interval"):
        factory(gate_interval=value)


@pytest.mark.parametrize("factory", [GateConfig, TrainConfig])
def test_disabled_gate_ignores_unused_settings(factory):
    factory(
        gate_interval=0,
        gate_games=-1,
        gate_workers=2**31,
        gate_simulations=0,
        gate_min_average_score=float("nan"),
        gate_min_finished=-1,
        gate_max_boards=0,
        gate_provider="unavailable",
        gate_temperature=float("inf"),
    )


@pytest.mark.parametrize("factory", [GateConfig, TrainConfig])
@pytest.mark.parametrize(("name", "value"), [
    ("gate_games", 0), ("gate_games", -2), ("gate_games", 3),
    ("gate_games", 100.0), ("gate_games", True),
    ("gate_workers", 0), ("gate_workers", -1), ("gate_workers", 2**31),
    ("gate_workers", 1.5), ("gate_workers", True),
    ("gate_simulations", 0), ("gate_simulations", -1),
    ("gate_simulations", 1.5), ("gate_simulations", True),
    ("gate_min_average_score", -1.1), ("gate_min_average_score", 1.1),
    ("gate_min_average_score", float("nan")), ("gate_min_average_score", float("inf")),
    ("gate_min_average_score", float("-inf")), ("gate_min_average_score", "0.5"),
    ("gate_min_finished", -1), ("gate_min_finished", 101),
    ("gate_min_finished", 0.5), ("gate_min_finished", True),
    ("gate_max_boards", 0), ("gate_max_boards", -1),
    ("gate_max_boards", 1.5), ("gate_max_boards", True),
    ("gate_provider", "gpu"),
    ("gate_temperature", -0.1), ("gate_temperature", float("nan")),
    ("gate_temperature", float("inf")), ("gate_temperature", float("-inf")),
    ("gate_temperature", "0.1"),
])
def test_enabled_gate_validates_all_fields(factory, name, value):
    with pytest.raises(ValueError, match=name):
        factory(**{"gate_interval": 1, name: value})


@pytest.mark.parametrize("factory", [GateConfig, TrainConfig])
@pytest.mark.parametrize("average_score", [-1.0, 1.0])
@pytest.mark.parametrize("min_finished", [0, 2])
@pytest.mark.parametrize("provider", ["cpu", "cuda"])
def test_enabled_gate_accepts_boundaries(factory, average_score, min_finished, provider):
    factory(
        gate_interval=1, gate_games=2, gate_workers=2**31 - 1,
        gate_simulations=1, gate_min_average_score=average_score,
        gate_min_finished=min_finished, gate_max_boards=1,
        gate_provider=provider, gate_temperature=0.0,
    )


def test_round_trip_and_legacy_checkpoint_defaults():
    config = TrainConfig(gate_interval=3, gate_games=60, gate_workers=7,
                         gate_simulations=42, gate_min_average_score=0.1,
                         gate_min_finished=20, gate_max_boards=8,
                         gate_provider="cpu", gate_temperature=0.25)
    data = config.to_dict()
    assert TrainConfig.from_dict(data) == config
    for name in DEFAULTS:
        del data[name]
    restored = TrainConfig.from_dict(data)
    assert {name: getattr(restored, name) for name in DEFAULTS} == DEFAULTS


@pytest.mark.parametrize("debug", [False, True])
def test_cli_flags_map_to_flat_config(debug):
    pytest.importorskip("torch")
    from suzero_5dchess.cli import _build_config, build_parser

    args = ["train"] + (["--debug"] if debug else [])
    default = _build_config(build_parser().parse_args(args))
    expected_defaults = DEFAULTS if not debug else {**DEFAULTS, "gate_interval": 0}
    assert {name: getattr(default, name) for name in DEFAULTS} == expected_defaults
    config = _build_config(build_parser().parse_args(args + [
        "--gate-interval", "3", "--gate-games", "60", "--gate-workers", "7",
        "--gate-simulations", "42", "--gate-min-average-score", "0.1",
        "--gate-min-finished", "20", "--gate-max-boards", "8",
        "--gate-provider", "cpu", "--gate-temperature", "0.25",
    ]))
    expected = dict(zip(DEFAULTS, (3, 60, 7, 42, 0.1, 20, 8, "cpu", 0.25)))
    assert {name: getattr(config, name) for name in DEFAULTS} == expected
    assert config.mcts.num_simulations == default.mcts.num_simulations
    assert config.self_play == default.self_play
    assert config.model_match_simulations == default.model_match_simulations


def test_cli_revalidates_gate_settings():
    pytest.importorskip("torch")
    from suzero_5dchess.cli import _build_config, build_parser

    args = build_parser().parse_args(["train", "--gate-interval", "1",
                                      "--gate-workers", "0"])
    with pytest.raises(ValueError, match="gate_workers"):
        _build_config(args)


def test_gate_config_module_needs_only_standard_library():
    package_dir = Path(__file__).resolve().parents[1] / "suzero_5dchess"
    # Avoid package __init__, which intentionally imports the training runtime.
    script = """
import sys
import types
package = types.ModuleType('suzero_5dchess')
package.__path__ = [sys.argv[1]]
sys.modules[package.__name__] = package
from suzero_5dchess.config import GateConfig
assert GateConfig().gate_interval == 1
assert not {'torch', 'numpy', 'onnxruntime'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-I", "-S", "-c", script, str(package_dir)],
                   check=True, capture_output=True, text=True)


def test_launcher_gate_environment_and_debug_order():
    launcher = Path(__file__).resolve().parents[2] / "run_train.bat"
    text = launcher.read_text(encoding="utf-8")
    defaults = {
        "GATING": "1", "GATE_INTERVAL": "3", "GATE_GAMES": "100",
        "GATE_WORKERS": "20", "GATE_SIMS": "200", "GATE_MIN_AVERAGE_SCORE": "0.05",
        "GATE_MIN_FINISHED": "50", "GATE_PROVIDER": "cuda",
        "GATE_MAX_BOARDS": "50",
    }
    for name, value in defaults.items():
        assert f'if not defined {name} set "{name}={value}"' in text
    assignments = [line for line in text.splitlines()
                   if 'set "DEFAULT_ARGS=' in line and "--gate-" in line]
    assert len(assignments) == 2
    enabled = [line for line in assignments if 'if "%GATING%"=="1"' in line]
    disabled = [line for line in assignments if 'if "%GATING%"=="0"' in line]
    assert len(enabled) == 1 and len(disabled) == 1
    gate_args = enabled[0]
    assert gate_args.startswith('if "%GATING%"=="1" set "DEFAULT_ARGS=%DEFAULT_ARGS% ')
    for flag, variable in (
        ("interval", "INTERVAL"), ("games", "GAMES"), ("workers", "WORKERS"),
        ("simulations", "SIMS"), ("min-average-score", "MIN_AVERAGE_SCORE"),
        ("min-finished", "MIN_FINISHED"), ("provider", "PROVIDER"),
        ("max-boards", "MAX_BOARDS"),
    ):
        assert f"--gate-{flag} %GATE_{variable}%" in gate_args
    assert disabled[0] == (
        'if "%GATING%"=="0" set "DEFAULT_ARGS=%DEFAULT_ARGS% --gate-interval 0"')
    debug_reset = 'if defined DEBUG_MODE set "DEFAULT_ARGS=--record-games %CONTINUOUS_ARG%"'
    assert text.index(debug_reset) < text.index(gate_args)
    assert text.index(debug_reset) < text.index(disabled[0])
    assert "GATE_ARGS" not in text
    assert "echo [INFO] Gating:        on;" in text
    assert "echo [INFO] Gating:        off" in text
