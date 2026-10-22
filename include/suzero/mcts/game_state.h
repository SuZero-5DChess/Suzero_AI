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
///  * `stateKey()` identifies a state for transposition merging. Equal keys
///    mean the two states are indistinguishable to the search: same player to
///    move, same legal actions, same evaluator output, same terminal status and
///    the same terminal utility. The key must therefore include every part of
///    the state that affects those things. If the game has repetition rules
///    (superko, draw by repetition, move counters), the history context that
///    those rules depend on must be folded into the key; otherwise the search
///    may merge positions that are not actually interchangeable. This is the
///    abstract counterpart of KataGo's GraphHash plus graphSearchRepBound.
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

  /// Identity of the position for graph search.
  ///
  /// Called once for the root and once for every edge that is expanded. The
  /// search keeps at most one node per key and reuses it when the same state is
  /// reached through different move orders. See the contract above.
  virtual StateKey stateKey() const = 0;
};

}  // namespace suzero::mcts
