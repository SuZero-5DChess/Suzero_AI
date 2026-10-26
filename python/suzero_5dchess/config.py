"""Configuration objects for 5D chess AlphaZero training."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from numbers import Integral, Real

from .endgame_sampling import EndgameSamplingConfig
from .protocol import VARIANTS
from .replay_window import WindowConfig


@dataclass
class NetworkConfig:
    """``cls_action_mlp`` network geometry.

    ``d_model`` stays at 128 (the reference width).  With one transformer
    layer and a 64-wide action MLP the model has 192,058 parameters, below the
    200,000 budget enforced at construction time.
    """

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 1
    d_ff: int = 256
    dropout: float = 0.1
    action_mlp_hidden: int = 64
    max_params: int = 200_000

    def __post_init__(self) -> None:
        if self.d_model < 4 or self.d_model % 4 != 0:
            raise ValueError("d_model must be a positive multiple of 4")
        if self.n_heads <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.action_mlp_hidden <= 0:
            raise ValueError("action_mlp_hidden must be positive")
        if self.max_params <= 0:
            raise ValueError("max_params must be positive")


@dataclass
class MctsConfig:
    num_simulations: int = 800
    c_puct: float = 2.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    q_init: float = -1.0
    max_depth: int = 512

    def __post_init__(self) -> None:
        if self.num_simulations <= 0:
            raise ValueError("num_simulations must be positive")
        if self.c_puct < 0:
            raise ValueError("c_puct must be non-negative")
        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive")
        if not 0.0 <= self.dirichlet_epsilon <= 1.0:
            raise ValueError("dirichlet_epsilon must be in [0, 1]")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")


@dataclass
class SelfPlayConfig:
    num_games: int = 64
    min_board_limit: int = 15
    max_board_limit: int = 25
    material_scale: float = 2.0
    temperature: float = 1.0
    temperature_final: float = 0.1
    temperature_threshold: int = 30
    playout_cap_randomization: bool = True
    full_search_probability: float = 0.25
    fast_simulations: int = 50
    record_games: bool = False
    # Reuse well-visited positions from full-search trees without new playouts.
    # Enabled by default; ``--no-record-tree-positions`` disables it.
    record_tree_positions: bool = True
    record_tree_min_visits: int = 64
    record_tree_max_depth: int = 5
    record_tree_max_per_turn: int = 4
    record_tree_sample_probability: float = 0.25
    # Concurrent self-play games.  With ``processes > 1`` this stays a total
    # budget shared across the worker processes rather than a per-process
    # count.
    workers: int = 1

    # Execution backend for self-play.  ``python`` runs MCTS with the live
    # PyTorch network in ``workers`` Python threads.  ``cpp_onnx`` exports the
    # network to ONNX each iteration and runs PUCT + ONNX Runtime in C++ on
    # ``workers`` game threads, selecting the device with
    # ``cpp_onnx_provider`` (cpu or cuda).
    backend: str = "python"
    cpp_onnx_provider: str = "cpu"

    # Number of C++ ONNX evaluator server threads used by the ``cpp_onnx``
    # backend.  Each server thread owns its own ONNX session (and therefore
    # its own copy of the model), so this is deliberately separate from
    # ``workers`` (the number of concurrent games).  When
    # ``processes > 1`` this is a total budget shared by the self-play
    # processes.  More threads may improve concurrency, but they do not
    # guarantee larger batches (independent sessions can each see smaller
    # batches) and they increase session memory/startup cost; benchmark
    # before increasing.
    num_server_threads: int = 1

    # Number of self-play worker processes.  ``1`` keeps the original
    # single-process path.  Values above 1 require the ``cpp_onnx`` backend;
    # ``workers``, ``num_server_threads`` and ``num_games`` remain total
    # budgets and must each be at least ``processes``.  Budgets are shared
    # across processes and need not divide evenly, and they are never scaled
    # up automatically.
    processes: int = 1

    def __post_init__(self) -> None:
        if self.num_games <= 0:
            raise ValueError("num_games must be positive")
        if not 1 <= self.min_board_limit <= self.max_board_limit:
            raise ValueError("board limit range is invalid")
        if self.temperature < 0 or self.temperature_final < 0:
            raise ValueError("temperatures must be non-negative")
        if not 0.0 <= self.full_search_probability <= 1.0:
            raise ValueError("full_search_probability must be in [0, 1]")
        if self.fast_simulations <= 0:
            raise ValueError("fast_simulations must be positive")
        if not isinstance(self.record_tree_positions, bool):
            raise ValueError("record_tree_positions must be a boolean")
        for name in (
            "record_tree_min_visits", "record_tree_max_depth", "record_tree_max_per_turn",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            if value > 2**31 - 1:
                raise ValueError(f"{name} must fit a positive 32-bit integer")
        if self.record_tree_max_depth > 64:
            raise ValueError("record_tree_max_depth must be <= 64")
        probability = self.record_tree_sample_probability
        if (
            isinstance(probability, bool)
            or not isinstance(probability, Real)
            or not 0.0 <= probability <= 1.0
        ):
            raise ValueError("record_tree_sample_probability must be in [0, 1]")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if self.backend not in ("python", "cpp_onnx"):
            raise ValueError(f"unknown self-play backend {self.backend!r}")
        if self.cpp_onnx_provider not in ("cpu", "cuda"):
            raise ValueError(
                f"unknown cpp_onnx provider {self.cpp_onnx_provider!r}"
            )
        if self.num_server_threads < 1:
            raise ValueError("num_server_threads must be >= 1")
        if self.processes < 1:
            raise ValueError("processes must be >= 1")
        if self.processes > 1:
            if self.backend != "cpp_onnx":
                raise ValueError(
                    "self-play processes > 1 requires backend 'cpp_onnx'"
                )
            if self.workers < self.processes:
                raise ValueError(
                    "workers must be >= processes when processes > 1"
                )
            if self.num_server_threads < self.processes:
                raise ValueError(
                    "num_server_threads must be >= processes when processes > 1"
                )
            if self.num_games < self.processes:
                raise ValueError(
                    "num_games must be >= processes when processes > 1"
                )


@dataclass
class GateConfig:
    """Candidate promotion settings, enabled by default without runtime dependencies."""

    gate_interval: int = 1
    gate_games: int = 100
    gate_workers: int = 20
    gate_simulations: int = 200
    gate_min_average_score: float = 0.05
    gate_min_finished: int = 50
    gate_max_boards: int = 50
    gate_provider: str = "cuda"
    gate_temperature: float = 0.1

    def __post_init__(self) -> None:
        if (
            isinstance(self.gate_interval, bool)
            or not isinstance(self.gate_interval, Integral)
            or self.gate_interval < 0
        ):
            raise ValueError("gate_interval must be a non-negative integer")
        if self.gate_interval == 0:
            return
        for name in ("gate_games", "gate_workers", "gate_simulations", "gate_max_boards"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.gate_games % 2:
            raise ValueError("gate_games must be a positive even number")
        if self.gate_workers > 2**31 - 1:
            raise ValueError("gate_workers must be a positive 32-bit integer")
        if (
            isinstance(self.gate_min_finished, bool)
            or not isinstance(self.gate_min_finished, Integral)
            or not 0 <= self.gate_min_finished <= self.gate_games
        ):
            raise ValueError("gate_min_finished must be in [0, gate_games]")
        for name in ("gate_min_average_score", "gate_temperature"):
            value = getattr(self, name)
            try:
                finite = isinstance(value, Real) and math.isfinite(value)
            except OverflowError:
                finite = False
            if isinstance(value, bool) or not finite:
                raise ValueError(f"{name} must be finite")
        if not -1.0 <= self.gate_min_average_score <= 1.0:
            raise ValueError("gate_min_average_score must be in [-1, 1]")
        if self.gate_temperature < 0:
            raise ValueError("gate_temperature must be non-negative")
        if self.gate_provider not in ("cpu", "cuda"):
            raise ValueError("gate_provider must be cpu or cuda")


@dataclass
class TrainConfig:
    variant: str = "very_small"
    device: str = "auto"

    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    max_grad_norm: float = 1.0

    batch_size: int = 256
    min_replay_size: int = 512
    replay_buffer_size: int = 100_000
    # ``file`` writes each finished self-play game to its own NPZ archive and
    # trains from those files; ``memory`` keeps the whole round in a
    # ReplayBuffer encoded in RAM.  File storage is the default so samples
    # survive a crash and never accumulate in the parent process.
    replay_backend: str = "file"
    # Shuffle rounds never repeat the rows of the round before them.  Disabling
    # this samples every round from the whole replay pool instead, which allows
    # repeats across rounds but keeps each round the full pool size.
    no_repeat_rows: bool = True
    num_iterations: int = 100
    epochs_per_iteration: int = 10
    steps_per_epoch: int = 10

    save_interval: int = 5
    export_interval: int = 1
    evaluation: bool = False
    evaluation_interval: int = 20
    evaluation_games: int = 50
    evaluation_simulations: int = 50
    reference_checkpoint: str | None = None

    model_match_interval: int = 0
    model_match_lookback: int = 20
    model_match_games: int = 20
    # Concurrent C++ games per opponent; independent of self-play workers.
    model_match_workers: int = 20
    model_match_simulations: int = 50
    model_match_max_boards: int = 25
    model_match_provider: str = "cpu"
    model_match_reference: str | None = None

    gate_interval: int = 1
    gate_games: int = 100
    gate_workers: int = 20
    gate_simulations: int = 200
    gate_min_average_score: float = 0.05
    gate_min_finished: int = 50
    gate_max_boards: int = 50
    gate_provider: str = "cuda"
    gate_temperature: float = 0.1

    checkpoint_dir: str = "runs"
    log_dir: str | None = None
    gn5d_data_dir: str | None = None
    tensorboard: bool = True
    tensorboard_dir: str | None = None
    onnx_output: str | None = None
    jsonl: bool = True
    seed: int = 20261023

    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MctsConfig = field(default_factory=MctsConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    # Sample rows closer to their game's end more often (see
    # :mod:`suzero_5dchess.endgame_sampling`).  Off by default so resuming an
    # older checkpoint keeps the replay distribution it was trained with.
    endgame: EndgameSamplingConfig = field(default_factory=EndgameSamplingConfig)

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.min_replay_size <= 0:
            raise ValueError("min_replay_size must be positive")
        if self.replay_buffer_size < self.min_replay_size:
            raise ValueError("replay_buffer_size must be >= min_replay_size")
        if self.replay_backend not in ("file", "memory"):
            raise ValueError("replay_backend must be 'file' or 'memory'")
        if not isinstance(self.no_repeat_rows, bool):
            raise ValueError("no_repeat_rows must be a boolean")
        if self.num_iterations < 0:
            raise ValueError("num_iterations must be >= 0")
        if self.epochs_per_iteration <= 0 or self.steps_per_epoch <= 0:
            raise ValueError("training schedule must be positive")
        if self.save_interval <= 0:
            raise ValueError("save_interval must be positive")
        if self.export_interval < 0:
            raise ValueError("export_interval must be >= 0")
        if self.model_match_interval < 0:
            raise ValueError("model_match_interval must be >= 0")
        if self.model_match_lookback <= 0:
            raise ValueError("model_match_lookback must be positive")
        if self.model_match_games <= 0 or self.model_match_games % 2:
            raise ValueError("model_match_games must be a positive even number")
        if self.model_match_workers <= 0 or self.model_match_workers > 2**31 - 1:
            raise ValueError("model_match_workers must be a positive 32-bit integer")
        if self.model_match_simulations <= 0:
            raise ValueError("model match simulations must be positive")
        if self.model_match_max_boards < 1:
            raise ValueError("model match max_boards must be positive")
        if self.model_match_provider not in ("cpu", "cuda"):
            raise ValueError("model_match_provider must be cpu or cuda")
        if self.evaluation_games <= 0 or self.evaluation_games % 2 != 0:
            raise ValueError("evaluation_games must be a positive even number")
        if self.evaluation_simulations <= 0:
            raise ValueError("evaluation_simulations must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.max_grad_norm < 0:
            raise ValueError("max_grad_norm must be non-negative")
        GateConfig(**{item.name: getattr(self, item.name) for item in fields(GateConfig)})
        self.network.__post_init__()
        self.mcts.__post_init__()
        self.self_play.__post_init__()
        self.window.__post_init__()
        self.endgame.__post_init__()
        if self.window.enabled and self.window.max_samples < self.replay_buffer_size:
            raise ValueError("window max_samples must be >= replay_buffer_size")
        if (
            self.replay_backend == "file"
            and self.window.enabled
            and self.window.sampling != "shuffle"
        ):
            raise ValueError(
                "replay_backend='file' always draws shuffled file-sequential "
                "rounds; window sampling must be 'shuffle' (use "
                "replay_backend='memory' for uniform sampling)"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "TrainConfig":
        data = dict(value)
        network = NetworkConfig(**data.pop("network", {}))  # type: ignore[arg-type]
        mcts = MctsConfig(**data.pop("mcts", {}))  # type: ignore[arg-type]
        self_play_data = dict(data.pop("self_play", {}))  # type: ignore[arg-type]
        # Existing checkpoints may contain the retired small-step cap.
        self_play_data.pop("max_game_length", None)
        self_play = SelfPlayConfig(**self_play_data)
        window = WindowConfig(**data.pop("window", {}))  # type: ignore[arg-type]
        endgame = EndgameSamplingConfig(
            **data.pop("endgame", {})  # type: ignore[arg-type]
        )
        known = {field.name for field in cls.__dataclass_fields__.values()}
        filtered = {key: item for key, item in data.items() if key in known}
        return cls(
            network=network, mcts=mcts, self_play=self_play, window=window,
            endgame=endgame, **filtered,
        )


def debug_train_config() -> TrainConfig:
    """A tiny, CPU-friendly preset used to exercise the whole pipeline."""

    config = TrainConfig(
        num_iterations=1,
        epochs_per_iteration=1,
        steps_per_epoch=2,
        batch_size=8,
        min_replay_size=2,
        replay_buffer_size=2_000,
        save_interval=1,
        export_interval=1,
        evaluation=False,
        tensorboard=False,
        # The debug preset stays a fast CPU pipeline smoke; pass --gate-interval
        # (or GATING=1 in the launcher) to exercise promotion matches here.
        gate_interval=0,
        self_play=SelfPlayConfig(
            num_games=2,
            min_board_limit=4,
            max_board_limit=4,
            playout_cap_randomization=False,
        ),
        mcts=MctsConfig(num_simulations=2),
    )
    return config
