// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-5DChess contributors
//
// This file is part of SuZero AI, a 5D chess AlphaZero project. The search
// core is derived from KataGo (https://github.com/lightvector/KataGo).
// See THIRD_PARTY_NOTICES.md.
//
// Self-play driver: several game threads each own one search
// tree and one game, and share a single (normally batching) evaluator. This is
// the C++ counterpart of KataGo's `selfplay` game-loop threads.

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "suzero/mcts/evaluator.h"
#include "suzero/mcts/game_state.h"
#include "suzero/mcts/search_params.h"
#include "suzero/mcts/types.h"

namespace suzero::mcts {

/// Configuration of a self-play run.
struct SelfPlayConfig {
  /// Number of concurrent games (one std::thread and one search tree each).
  int numGameThreads = 1;

  /// Total games to play. `<= 0` means unlimited; the streaming overload then
  /// runs until `requestStop()`. The collecting overload requires `> 0`.
  int maxGames = 1;

  /// Safety cap on applied actions per game. `<= 0` means unlimited.
  int maxMovesPerGame = 0;

  /// Search parameters for every move. Set at least `maxPlayouts` and the
  /// `chosenMoveTemperature` schedule here.
  SearchParams searchParams;

  /// Base seed. Game `i` receives a deterministic seed derived from this value
  /// and `i`, so a run is reproducible regardless of thread scheduling.
  std::uint64_t seed = kDefaultSeed;
};

/// One labelled position produced by self-play.
///
/// `value` is the final game outcome from `player`'s perspective, using the
/// same scale as `Evaluation::utility` and `IGameState::terminalUtility`. It is
/// filled in after the game ends, so a truncated (unfinished) game yields 0.
struct SelfPlaySample {
  /// Snapshot of the position. Owned because the game state advances.
  std::unique_ptr<IGameState> state;

  /// Dense root visit distribution of size `state->actionCount()`, normalized
  /// to sum to 1.
  std::vector<float> policy;

  /// Player to move in `state`.
  PlayerId player = 0;

  /// Number of actions applied before this position (0 for the initial one).
  int moveIndex = 0;

  /// Final outcome from `player`'s perspective.
  double value = 0.0;
};

/// Result of one self-play game.
struct SelfPlayGame {
  std::int64_t gameIndex = 0;
  std::vector<SelfPlaySample> samples;

  /// Player to move in the final position, and the terminal utility from that
  /// player's perspective (0 when the game was truncated).
  PlayerId finalPlayer = 0;
  double finalUtility = 0.0;

  int movesPlayed = 0;

  /// True when the final position is terminal. `truncated` is true when the
  /// run stopped early (move cap or requestStop()).
  bool terminal = false;
  bool truncated = false;
};

/// Runs independent self-play games in parallel.
///
/// Every game owns a `PuctSearch`; the evaluator is shared. Therefore the
/// evaluator must be thread-safe. `BatchedEvaluator` is, and is the intended
/// pairing: the searches block on individual queries while a server thread
/// combines queries from different games into one device call.
///
/// The runner never mutates a state while an evaluation is in flight: the
/// search thread is blocked inside `evaluate()` until the batch completes.
///
/// A runner drives one run at a time: do not call `run()` concurrently, and do
/// not destroy the runner while a run is in progress. `requestStop()` is the
/// supported way to end a run from another thread.
class SelfPlayRunner {
 public:
  /// Creates a fresh initial state for each game. Must return a non-null state.
  using StateFactory = std::function<std::shared_ptr<IGameState>()>;

  /// Called once per finished game. It may run on any worker thread and must
  /// be thread-safe; games finish out of order.
  using GameCallback =
      std::function<void(std::int64_t gameIndex, SelfPlayGame&& game)>;

  /// `stateFactory` and `evaluator` must be non-null.
  SelfPlayRunner(SelfPlayConfig config, StateFactory stateFactory,
                 std::shared_ptr<IEvaluator> evaluator);
  ~SelfPlayRunner();

  SelfPlayRunner(const SelfPlayRunner&) = delete;
  SelfPlayRunner& operator=(const SelfPlayRunner&) = delete;

  /// Plays `config().maxGames` games and returns them indexed by game index.
  /// Throws std::invalid_argument when `maxGames <= 0`, because an unlimited
  /// run cannot be returned as a finite vector.
  std::vector<SelfPlayGame> run();

  /// Plays games and streams each finished game to `onGameFinished` instead of
  /// collecting them. Returns the number of finished games. With
  /// `maxGames <= 0` this runs until `requestStop()`.
  std::size_t run(const GameCallback& onGameFinished);

  /// Asks workers to stop after the current position. Games in progress are
  /// returned as truncated. Safe to call from any thread.
  void requestStop();

  bool stopRequested() const;

  const SelfPlayConfig& config() const { return config_; }

 private:
  SelfPlayGame playGame(std::int64_t gameIndex);
  std::uint64_t seedForGame(std::int64_t gameIndex) const;

  std::size_t runImpl(
      std::vector<std::unique_ptr<SelfPlayGame>>* collected,
      const GameCallback* onGameFinished);

  SelfPlayConfig config_;
  StateFactory state_factory_;
  std::shared_ptr<IEvaluator> evaluator_;
  std::atomic<bool> stop_requested_{false};
};

}  // namespace suzero::mcts
