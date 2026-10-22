// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-02 contributors
//
// This file is part of a game-agnostic PUCT/MCTS library derived from KataGo
// (https://github.com/lightvector/KataGo). See THIRD_PARTY_NOTICES.md.

#pragma once

#include <vector>

#include "suzero/mcts/game_state.h"
#include "suzero/mcts/types.h"

namespace suzero::mcts {

/// Output of one position evaluation (a policy prior plus a scalar value).
struct Evaluation {
  /// Dense policy vector of size `state.actionCount()`.
  ///
  /// Entries for unavailable actions must be negative (or zero); the engine
  /// additionally filters entries through `IGameState::isLegal`. Positive
  /// entries are rescaled to sum to one. The vector may contain values larger
  /// than one before normalization.
  std::vector<float> policy;

  /// Expected utility from `state.currentPlayer()`'s perspective.
  ///
  /// The search engine treats this as an opaque utility. For Go, KataGo
  /// combines a win/loss term and a score term into this scalar before
  /// search; in this library that combination belongs to the evaluator.
  double utility = 0.0;
};

/// Abstract position evaluator (normally a neural network).
class IEvaluator {
 public:
  virtual ~IEvaluator() = default;

  /// Evaluates a non-terminal state. Must not mutate `state`.
  virtual Evaluation evaluate(const IGameState& state) = 0;
};

}  // namespace suzero::mcts
