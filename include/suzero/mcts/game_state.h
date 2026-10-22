// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-02 contributors
//
// This file is part of a game-agnostic PUCT/MCTS library derived from KataGo
// (https://github.com/lightvector/KataGo). See THIRD_PARTY_NOTICES.md.

#pragma once

#include <cstdint>
#include <memory>

#include "suzero/mcts/types.h"

namespace suzero::mcts {

/// Abstract game state.
///
/// The search engine owns no game rules. It only clones states, applies legal
/// actions, asks for the current player, and reads terminal utilities.
///
/// Contract:
///  * `actionCount()` is constant for every state of a given game instance.
///    It is the policy-vector size returned by IEvaluator.
///  * Action ids are dense in [0, actionCount()).
///  * `applyAction(a)` must only be called with a legal action. It mutates
///    *this* and returns nothing; call `clone()` first if the caller needs to
///    branch.
///  * `terminalUtility()` is the utility of the terminal position from the
///    perspective of `currentPlayer()`. It is only called when
///    `isTerminal()` is true.
///  * Values need not be in [-1, 1], but all utilities returned by the
///    evaluator and by terminal states should use the same scale. The search
///    parameters (`utilityRadius`, FPU reductions) are expressed on that scale.
///
/// Non-alternating games (for example games where one side makes several
/// consecutive sub-moves) are supported: the engine flips the value sign only
/// when a child's `currentPlayer()` differs from its parent's.
class IGameState {
 public:
  virtual ~IGameState() = default;

  /// Deep copy. The search engine clones the root once per playout.
  virtual std::unique_ptr<IGameState> clone() const = 0;

  /// Size of the policy vector, constant for the game.
  virtual int actionCount() const = 0;

  /// Whether `action` is legal in this state.
  virtual bool isLegal(ActionId action) const = 0;

  /// Applies a legal action in place.
  virtual void applyAction(ActionId action) = 0;

  /// Stable identifier of the side to move.
  virtual PlayerId currentPlayer() const = 0;

  /// Whether the game has ended.
  virtual bool isTerminal() const = 0;

  /// Utility for `currentPlayer()` at a terminal state.
  virtual double terminalUtility() const = 0;

  /// Optional identity used by future graph-search extensions.
  ///
  /// The current tree-search implementation does not merge transpositions and
  /// never calls this. It is part of the interface so that a graph-search
  /// backend can be added without changing game implementations.
  virtual std::uint64_t stateKey() const { return 0; }
};

}  // namespace suzero::mcts
