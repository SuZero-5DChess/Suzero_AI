"""Persistent, opt-in candidate promotion, independent of the training pipeline."""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path, PureWindowsPath
from typing import Protocol
from uuid import uuid4

import torch

from .checkpoint import CHECKPOINT_FORMAT
from .config import GateConfig, TrainConfig  # GateConfig is deliberately re-exported.
from .env import _5dchess, _require_binding
from .evaluation import load_network_checkpoint
from .export_onnx import export_network
from .match import MatchPlayer, play_model_match
from .network import SuzeroNetwork
from .protocol import ACTION_ENCODING, ACTION_INPUT_ENCODING, VariantSpec

logger = logging.getLogger(__name__)


class _GatePaths(Protocol):
    accepted_dir: Path
    accepted_state: Path
    gate_log_dir: Path


@dataclass
class GateState:
    iteration: int
    checkpoint: str | None
    variant: str
    network_config: dict[str, object]
    format: int = 1
    updated_at: str | None = None
    last_gate: dict[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class GateDecision:
    enabled: bool = True
    skipped: bool = False
    candidate_iteration: int | None = None
    accepted_before: int | None = None
    accepted_after: int | None = None
    decision: str = "rejected"
    reason: str = "score"
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    truncated: int = 0
    finished: int = 0
    average_score: float | None = None
    threshold: float = 0.05
    seconds: float = 0.0
    match_log: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _rate(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "N/A"


def _finite_number(value: object) -> bool:
    try:
        return (
            not isinstance(value, bool)
            and isinstance(value, Real)
            and math.isfinite(value)
        )
    except OverflowError:
        return False


def finished_scores(result: Mapping) -> list[float]:
    """Candidate-perspective utilities for every finished match game.

    Forced ``material`` terminals keep their continuous tanh score instead of
    collapsing to a win or loss; truncated games carry no utility and are
    skipped because the gate never enables a move cap.
    """
    games = result.get("games")
    if not isinstance(games, list):
        raise ValueError("match result must contain a games list")
    scores: list[float] = []
    for index, game in enumerate(games):
        if not isinstance(game, Mapping):
            raise ValueError(f"match game {index} must be an object")
        truncated = game.get("truncated")
        if type(truncated) is not bool:
            raise ValueError(f"match game {index} truncated flag must be boolean")
        if truncated:
            continue
        utilities = game.get("participant_utilities")
        if not isinstance(utilities, (list, tuple)) or len(utilities) != 2:
            raise ValueError(
                f"match game {index} participant utilities must be a pair")
        utility = utilities[0]
        if not _finite_number(utility):
            raise ValueError(
                f"match game {index} candidate utility must be finite")
        scores.append(float(utility))
    return scores


def decide_gate(
    totals: Mapping[str, int], scores: Sequence[float], *,
    threshold: float, min_finished: int,
) -> GateDecision:
    """Promote when the candidate's mean game score clears the threshold.

    ``scores`` holds one candidate-perspective utility per finished game, so a
    forced ``material`` terminal contributes its continuous tanh score rather
    than being rounded down to a bare win or loss.  Draws are ordinary 0.0
    entries; truncated games are excluded from both count and mean.
    """
    counts = {key: totals[key] for key in ("wins", "losses", "draws", "truncated")}
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("gate totals must contain non-negative integer counts")
    if not all(_finite_number(value) for value in scores):
        raise ValueError("gate scores must be finite numbers")
    finished = counts["wins"] + counts["losses"] + counts["draws"]
    if len(scores) != finished:
        raise ValueError(
            f"gate scores cover {len(scores)} games but totals report "
            f"{finished} finished games")
    average = sum(scores) / finished if finished else None
    sufficient = finished > 0 and finished >= min_finished
    return GateDecision(
        decision="accepted" if sufficient and average >= threshold else "rejected",
        reason="score" if sufficient else "insufficient_finished",
        games=finished + counts["truncated"], finished=finished,
        average_score=average, threshold=threshold, **counts,
    )


@contextmanager
def _preserve_modes(*networks: SuzeroNetwork):
    # Restore individual flags too: a training model may have selected frozen
    # submodules in evaluation mode before entering the gate.
    modes = {module: module.training for net in networks for module in net.modules()}
    try:
        for network in networks:
            network.eval()
        yield
    finally:
        for module, training in modes.items():
            module.training = training


def _iteration(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


class ModelGate:
    def __init__(
        self, config: TrainConfig, paths: _GatePaths, spec: VariantSpec,
        device: torch.device, *, run_match: Callable | None = None,
        exporter: Callable | None = None,
    ) -> None:
        self.config = config
        self.paths = paths
        self.spec = spec
        self.device = device
        self.enabled = config.gate_interval > 0
        self.state: GateState | None = None
        self._cache_key: tuple | None = None
        self._accepted: SuzeroNetwork | None = None
        self._exporter = exporter or export_network
        self._runner = run_match if run_match is not None else self._run_match
        if self.enabled and run_match is None:
            _require_binding()
            if not callable(getattr(_5dchess, "run_match", None)):
                raise RuntimeError("rebuild suzero_5dchess_py to enable model gating")

    def _metadata(self, iteration: int) -> dict[str, object]:
        return {
            "format": CHECKPOINT_FORMAT,
            "action_encoding": ACTION_ENCODING,
            "action_input_encoding": ACTION_INPUT_ENCODING,
            "variant": self.config.variant,
            "network_config": asdict(self.config.network),
            "iteration": iteration,
        }

    def _read_state(self, *, resume_iteration: int | None = None) -> GateState:
        path = self.paths.accepted_state
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("expected a JSON object")
            if type(data.get("format")) is not int or data["format"] != 1:
                raise ValueError("expected gate state format 1")
            iteration = _iteration(data.get("iteration"), "accepted iteration")
            if resume_iteration is not None and iteration > resume_iteration:
                raise ValueError(
                    "accepted iteration is ahead of the resumed training iteration; "
                    "resume from a checkpoint aligned with the accepted iteration, or "
                    f"delete {path} (accepted/state.json) to bootstrap again; "
                    "use a new run directory if immutable accepted snapshots conflict"
                )
            if data.get("variant") != self.config.variant or self.spec.name != self.config.variant:
                raise ValueError("accepted variant does not match the training configuration")
            if data.get("network_config") != asdict(self.config.network):
                raise ValueError("accepted network_config does not match the training configuration")
            name = data.get("checkpoint")
            if (not isinstance(name, str) or not name or name in (".", "..")
                    or Path(name).name != name or PureWindowsPath(name).name != name
                    or PureWindowsPath(name).drive):
                raise ValueError("accepted checkpoint must be a basename")
            if not (self.paths.accepted_dir / name).is_file():
                raise ValueError(f"accepted checkpoint does not exist: {name}")
            updated_at = data.get("updated_at")
            if updated_at is not None:
                if not isinstance(updated_at, str) or not updated_at.endswith("Z"):
                    raise ValueError("updated_at must be a UTC ISO timestamp ending in Z")
                stamp = datetime.fromisoformat(updated_at[:-1] + "+00:00")
                if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
                    raise ValueError("updated_at must be a UTC ISO timestamp")
            last_gate = data.get("last_gate")
            if last_gate is not None and not isinstance(last_gate, dict):
                raise ValueError("last_gate must be an object or null")
            return GateState(iteration, name, data["variant"], data["network_config"],
                             updated_at=updated_at, last_gate=last_gate)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"invalid gate state {path}: {exc}") from exc

    def _read_checkpoint(self, path: Path, iteration: int) -> dict:
        try:
            snapshot = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(snapshot, dict) or not isinstance(snapshot.get("model"), Mapping):
                raise ValueError("checkpoint has no model state")
            extra = snapshot.get("extra_state")
            if not isinstance(extra, dict):
                raise ValueError("checkpoint has no extra_state")
            _iteration(extra.get("iteration"), "checkpoint iteration")
            for key, value in self._metadata(iteration).items():
                if extra.get(key) != value:
                    raise ValueError(f"checkpoint {key} does not match gate state/configuration")
            return snapshot
        except Exception as exc:
            raise ValueError(f"invalid accepted checkpoint {path}: {exc}") from exc

    def _load_accepted(self, state: GateState, *, force: bool = False) -> SuzeroNetwork:
        path = self.paths.accepted_dir / state.checkpoint
        try:
            stat = path.stat()
            key = (state.iteration, path, stat.st_mtime_ns, stat.st_size)
            if force or key != self._cache_key or self._accepted is None:
                with torch.random.fork_rng(devices=[]):
                    self._read_checkpoint(path, state.iteration)
                    accepted = load_network_checkpoint(path, self.spec, self.device)
                    accepted.eval()
                    accepted.requires_grad_(False)
                self._accepted = accepted
                self._cache_key = key
        except Exception as exc:
            raise ValueError(f"cannot load accepted checkpoint {path}: {exc}") from exc
        self._accepted.eval()
        return self._accepted

    def _snapshot(self, network: SuzeroNetwork, iteration: int) -> str:
        if network.config != self.config.network or network.spec.name != self.spec.name:
            raise ValueError("candidate network does not match gate configuration")
        directory = self.paths.accepted_dir
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"agent_iter_{iteration:04d}.pt"
        model = network.state_dict()
        if path.exists():
            existing = self._read_checkpoint(path, iteration)["model"]
            if existing.keys() != model.keys() or any(
                not isinstance(existing[key], torch.Tensor)
                or existing[key].dtype != tensor.dtype
                or existing[key].shape != tensor.shape
                or not torch.equal(existing[key], tensor.detach().cpu())
                for key, tensor in model.items()
            ):
                raise ValueError(f"immutable accepted checkpoint already contains different weights: {path}")
            return path.name
        tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            torch.save({"format": CHECKPOINT_FORMAT, "model": model,
                        "extra_state": self._metadata(iteration)}, tmp)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        return path.name

    def _publish(self, network: SuzeroNetwork, iteration: int,
                 last_gate: dict[str, object] | None = None) -> GateState:
        # A durable immutable snapshot must precede publication of its pointer.
        with torch.random.fork_rng(devices=[]):
            name = self._snapshot(network, iteration)
        state = GateState(
            iteration, name, self.config.variant, asdict(self.config.network),
            updated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            last_gate=last_gate,
        )
        path = self.paths.accepted_state
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            tmp.write_text(json.dumps(state.as_dict(), ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        self.state = state
        self._cache_key = None
        self._accepted = None
        return state

    def bootstrap_or_load(self, network: SuzeroNetwork, resume_iteration: int) -> GateState:
        iteration = _iteration(resume_iteration, "resume iteration")
        if not self.enabled:
            self.state = GateState(iteration, None, self.config.variant,
                                   asdict(self.config.network))
            return self.state
        if not self.paths.accepted_state.exists():
            state = self._publish(network, iteration)
            logger.info("model gate: bootstrapped accepted iteration %d at %s",
                        state.iteration, self.paths.accepted_dir / state.checkpoint)
            return state
        state = self._read_state(resume_iteration=iteration)
        self._load_accepted(state, force=True)
        self.state = state
        logger.info("model gate: loaded accepted iteration %d from %s",
                    state.iteration, self.paths.accepted_dir / state.checkpoint)
        return state

    def resolve(self, network: SuzeroNetwork) -> SuzeroNetwork:
        if not self.enabled:
            return network
        state = self._read_state()
        accepted = self._load_accepted(state)
        self.state = state
        return accepted

    def summary(self) -> dict[str, object]:
        return {"iteration": self.state.iteration if self.state else None,
                "checkpoint": self.state.checkpoint if self.state else None}

    def _run_match(self, network: SuzeroNetwork, accepted_network: SuzeroNetwork,
                   iteration: int) -> dict:
        cfg = self.config
        with tempfile.TemporaryDirectory(prefix="suzero-gate-") as directory:
            # CPU copies prevent ONNX tracing from using training GPU resources
            # or mutating the live models, including their per-module modes.
            candidate = self._exporter(copy.deepcopy(network).cpu().eval(), self.spec,
                                       Path(directory) / "candidate.onnx")
            accepted = self._exporter(copy.deepcopy(accepted_network).cpu().eval(), self.spec,
                                      Path(directory) / "accepted.onnx")
            settings = dict(simulations=cfg.gate_simulations, c_puct=cfg.mcts.c_puct,
                            max_depth=cfg.mcts.max_depth, temperature=cfg.gate_temperature,
                            provider=cfg.gate_provider)
            return play_model_match(
                MatchPlayer(candidate, name=f"candidate_iter_{iteration}", **settings),
                MatchPlayer(accepted, name=f"accepted_iter_{self.state.iteration}", **settings),
                variant=self.spec, games=cfg.gate_games, workers=cfg.gate_workers,
                max_moves=0, board_limit=cfg.gate_max_boards,
                material_scale=cfg.self_play.material_scale,
                seed=cfg.seed + iteration * 1009,
            )

    @staticmethod
    def _log_decision(decision: GateDecision) -> GateDecision:
        logger.info(
            "model gate: candidate iteration %s decision=%s reason=%s "
            "accepted=%s -> %s skipped=%s",
            decision.candidate_iteration, decision.decision, decision.reason,
            decision.accepted_before, decision.accepted_after, decision.skipped,
        )
        if decision.finished:
            # Gate matches never truncate: hitting the board limit scores the
            # game as a normal material terminal.  Keep the anomaly visible if
            # a custom runner ever reports one.
            truncated_note = (
                f", {decision.truncated} truncated" if decision.truncated else ""
            )
            average_note = (
                f"{decision.average_score:+.4f}"
                if decision.average_score is not None else "N/A"
            )
            logger.info(
                "  candidate result: %d wins, %d losses, %d draws "
                "(%d finished%s); average score=%s "
                "(candidate perspective, forced material included), "
                "win rate=%s (wins/finished); threshold=%+.4f; %.1fs",
                decision.wins, decision.losses, decision.draws,
                decision.finished, truncated_note,
                average_note,
                _rate(decision.wins / decision.finished),
                decision.threshold, decision.seconds,
            )
        return decision

    def evaluate(self, network: SuzeroNetwork, candidate_iteration: int, *,
                 trained: bool) -> GateDecision:
        iteration = _iteration(candidate_iteration, "candidate iteration")
        before = self.state.iteration if self.state else None
        base = GateDecision(enabled=self.enabled, candidate_iteration=iteration,
                            accepted_before=before, accepted_after=before,
                            threshold=self.config.gate_min_average_score)
        if not self.enabled:
            return replace(base, skipped=True, reason="disabled")
        # A replay-starved round never runs the gate, even on an interval boundary.
        if not trained:
            return self._log_decision(replace(base, skipped=True, reason="no_training"))
        if iteration % self.config.gate_interval:
            return self._log_decision(replace(base, skipped=True, reason="interval"))
        accepted = self.resolve(network)
        before = self.state.iteration
        base = replace(base, accepted_before=before, accepted_after=before)
        started = time.monotonic()
        error = None
        raw = None
        try:
            with torch.random.fork_rng(devices=[]), _preserve_modes(network, accepted):
                raw = self._runner(network, accepted, iteration)
                decision = replace(
                    decide_gate(raw["totals"][0], finished_scores(raw),
                                threshold=self.config.gate_min_average_score,
                                min_finished=self.config.gate_min_finished),
                    candidate_iteration=iteration, accepted_before=before, accepted_after=before,
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            decision = replace(base, reason="match_error", games=self.config.gate_games)
        decision.seconds = time.monotonic() - started
        output = self.paths.gate_log_dir / f"iter_{iteration:04d}_{uuid4().hex}.json"
        decision.match_log = str(output)
        if decision.decision == "accepted":
            decision.accepted_after = iteration
            # Persistence failures are fatal, never downgraded to match errors.
            self._publish(network, iteration, decision.as_dict())
        self.paths.gate_log_dir.mkdir(parents=True, exist_ok=True)
        report = {"summary": decision.as_dict(), "match": raw}
        if error is not None:
            report["error"] = error
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return self._log_decision(decision)
