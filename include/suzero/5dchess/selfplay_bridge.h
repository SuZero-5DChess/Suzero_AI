// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-02 contributors
//
// Self-play driver used by the Python trainer's `cpp_onnx` backend.
//
// The Python trainer exports the live PyTorch network to ONNX, then calls
// `runSelfPlay()`: several game threads run PUCT with the shared, batching
// ONNX evaluator (optionally on CUDA) and return fully labelled positions.
// Keeping this loop in C++ avoids the Python MCTS/network overhead that
// dominates the pure-Python backend.
//
// The loop mirrors python/suzero_5dchess/self_play.py (forced SUBMIT, playout
// cap randomization, temperature schedule, samples only on full-search moves)
// but uses the game-agnostic PuctSearch with subtree reuse.

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "suzero/5dchess/game_state.h"
#include "suzero/5dchess/onnx_evaluator.h"
#include "suzero/mcts/types.h"

namespace suzero::chess5d {

/// Configuration of one self-play batch.
struct SelfPlayBridgeConfig {
  /// ONNX model + execution provider ("cpu" or "cuda").
  OnnxEvaluatorConfig evaluator;

  /// Board geometry. `game.board_limit` is replaced per game by the sampled
  /// value in [minBoardLimit, maxBoardLimit].
  GameConfig game;

  /// Number of game threads and games in this batch.
  int numGameThreads = 3;
  int numGames = 1;

  /// Full search and playout-cap randomized fast search.
  int numSimulations = 200;
  int fastSimulations = 50;
  bool playoutCapRandomization = true;
  double fullSearchProbability = 0.25;

  /// PUCT / root noise.
  double cpuctExploration = 2.0;
  double dirichletAlpha = 0.3;
  double dirichletEpsilon = 0.25;
  int maxDepth = 512;

  /// Temperature schedule, expressed in small steps.
  double temperature = 1.0;
  double temperatureFinal = 0.1;
  int temperatureThreshold = 30;

  /// Per-game board-limit sampling and material-scoring trigger.
  int minBoardLimit = 15;
  int maxBoardLimit = 25;
  double materialScale = 2.0;

  /// Evaluator batching. One server thread is enough for global batching.
  int maxBatchSize = 128;
  int numServerThreads = 1;

  /// Retain authoritative native actions and the final native FEN for archival.
  bool recordGames = false;

  /// Base seed; game i derives its seed from (seed, i).
  std::uint64_t seed = 0;
};

/// One labelled training position produced by self-play.
///
/// `policy` is the dense root visit distribution over all candidate actions
/// (SUBMIT last, like `Chess5DGameState::actions()`).
struct BridgeSample {
  EncodedPosition encoded;
  std::vector<Action> actions;
  std::vector<float> policy;
  mcts::PlayerId player = 0;
  int moveCount = 0;
  float value = 0.0f;
};

/// One authoritative action retained for .5dck archival. `player` is captured
/// before applying the concrete action.
struct BridgeMove {
  Action action;
  mcts::PlayerId player = 0;
};

/// Result of one game.
struct BridgeGame {
  int index = 0;

  /// Outcome from White's perspective.
  double whiteOutcome = 0.0;

  /// "capture_king", "material", or "no_legal_action".
  std::string terminalReason;

  int semimoves = 0;
  int turns = 0;
  int boardLimit = 0;
  int boardCount = 0;
  int finalPlayer = 0;
  int movesPlayed = 0;

  /// Empty unless SelfPlayBridgeConfig::recordGames is enabled.
  std::vector<BridgeMove> moves;
  std::string finalFen;

  std::vector<BridgeSample> samples;
};

/// Called once per finished game with the number of games completed so far.
/// It may run on any worker thread and must be thread-safe.
using BridgeGameCallback =
    std::function<void(int completed, const BridgeGame& game)>;

/// Evaluator/batching statistics for one runSelfPlay() call.
///
/// The counters come from the shared `BatchedEvaluator`: `batches` device
/// calls served `rows` requests, so `rows / batches` is the average batch
/// size actually handed to the GPU. A ratio far below `maxBatchSize` means
/// the search threads are not feeding the evaluator fast enough, while a high
/// `setupSeconds` relative to `totalSeconds` means most of the wall time was
/// session construction rather than inference.
struct SelfPlayStats {
  int numServerThreads = 0;
  int maxBatchSize = 0;

  std::uint64_t requests = 0;
  std::uint64_t batches = 0;
  std::uint64_t rows = 0;
  std::uint64_t maxObservedBatch = 0;

  /// Wall time of the whole call, including evaluator/session construction.
  double totalSeconds = 0.0;
  /// Time spent constructing the evaluator (ONNX session plus CUDA provider
  /// setup and cuDNN algorithm search), a subset of `totalSeconds`.
  double setupSeconds = 0.0;
};

/// Runs `config.numGames` games on `config.numGameThreads` threads and blocks
/// until all workers are joined. Results are indexed by game index. An
/// exception in any worker stops the remaining games and is rethrown.
///
/// When `stats` is non-null it receives the batching counters and wall-clock
/// timings of this call. It is left untouched if the call throws.
std::vector<BridgeGame> runSelfPlay(
    const SelfPlayBridgeConfig& config,
    const BridgeGameCallback& onGameFinished = {},
    SelfPlayStats* stats = nullptr);

}  // namespace suzero::chess5d
