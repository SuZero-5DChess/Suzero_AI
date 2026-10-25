// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-02 contributors

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "suzero/mcts/evaluator.h"
#include "suzero/mcts/game_state.h"
#include "suzero/mcts/search_params.h"

namespace suzero::mcts {

/// Opaque logical action shared by adapters, never a local policy index.
using MatchAction = std::string;

/// Two-player, zero-sum game adapter for heterogeneous model matches.
/// clone() must preserve this interface and the concrete model encoding.
class IMatchGame : public IGameState {
 public:
  /// Includes rules and utility scale, but excludes model-only encoding options.
  virtual std::string rulesKey() const = 0;
  /// Exact logical identity, including rule-relevant history, independent of
  /// local action ordering and model input encoding.
  virtual StateKey canonicalStateKey() const = 0;
  virtual MatchAction toMatchAction(ActionId action) const = 0;
  /// Returns kInvalidAction if the logical action cannot be represented here.
  virtual ActionId fromMatchAction(const MatchAction& action) const = 0;
  virtual std::string terminalReason() const { return {}; }
};

struct MatchParticipant {
  std::string name;
  PlayerId player = 0;
  /// Must return a fresh independently owned state for each call. Both
  /// participants' factories must start at the same logical position.
  /// Concurrent series require thread-safe factories and evaluators.
  std::function<std::shared_ptr<IMatchGame>()> stateFactory;
  std::shared_ptr<IEvaluator> evaluator;
  SearchParams searchParams;
};

struct MatchConfig {
  /// Applied small-step actions, not turns. Zero disables the cap.
  int maxMoves = 1000;
  std::uint64_t seed = kDefaultSeed;
};

struct MatchMove {
  PlayerId player = 0;
  /// Index in the participant array, not a PlayerId.
  int participantIndex = 0;
  MatchAction action;
  std::array<ActionId, 2> localActions{{kInvalidAction, kInvalidAction}};
};

struct MatchResult {
  bool terminal = false;
  bool truncated = false;
  PlayerId finalPlayer = 0;
  /// Absent for truncation: an unfinished game is NOT a draw.
  std::optional<double> finalUtility;
  std::array<std::optional<double>, 2> participantUtilities;
  std::string terminalReason;
  std::vector<MatchMove> moves;
};

/// Single-threaded match driver. Each side has an independent search and
/// evaluator. Configuration/protocol errors throw instead of becoming losses.
class MatchRunner {
 public:
  MatchRunner(MatchConfig config,
              std::array<MatchParticipant, 2> participants);
  MatchResult run() const;

 private:
  MatchConfig config_;
  std::array<MatchParticipant, 2> participants_;
};

struct MatchScore {
  int wins = 0;
  int losses = 0;
  int draws = 0;
  int truncated = 0;
};

struct MatchSeriesConfig {
  /// Each pair plays two games, exchanging the participant PlayerIds.
  int pairs = 1;
  MatchConfig game;
  /// Concurrent games, capped at pairs * 2. One preserves sequential execution.
  int numGameThreads = 1;
};

struct MatchSeriesResult {
  /// Games stay in scheduled order, regardless of completion order. Participant
  /// indices always refer to the input participants, even after a swap.
  std::vector<MatchResult> games;
  std::array<MatchScore, 2> totals;
  /// [participant index][seat index]. Seat 0/1 means the player id assigned to
  /// input participant 0/1, not necessarily the first/second player to act.
  std::array<std::array<MatchScore, 2>, 2> bySeat;
};

/// Called on the calling thread after each game, with its zero-based scheduled
/// index. Concurrent games may finish out of order; callbacks never overlap.
/// Exceptions stop new games, join active games and propagate to the caller;
/// a partial series is never reported as a complete paired result.
using MatchGameCallback = std::function<void(int, const MatchResult&)>;

/// Paired evaluation with up to numGameThreads concurrent games. Factories and
/// evaluators must support concurrent calls when numGameThreads > 1. Distinct
/// model evaluators stay distinct; only PlayerId assignments change per pair.
MatchSeriesResult runMatchSeries(
    const MatchSeriesConfig& config,
    const std::array<MatchParticipant, 2>& participants,
    const MatchGameCallback& onGameFinished = {});

}  // namespace suzero::mcts
