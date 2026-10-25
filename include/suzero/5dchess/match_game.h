// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-02 contributors

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "suzero/5dchess/game_state.h"
#include "suzero/mcts/match.h"

namespace suzero::chess5d {

/// Match adapter that gives one independently owned 5D-chess state a stable
/// cross-adapter action representation and model-specific local action order.
class Chess5DMatchGame final : public mcts::IMatchGame,
                                  public IChess5DModelInput {
 public:
  explicit Chess5DMatchGame(GameConfig config = {},
                               bool reverse_move_order = false);
  ~Chess5DMatchGame() override;

  Chess5DMatchGame(const Chess5DMatchGame&) = delete;
  Chess5DMatchGame& operator=(const Chess5DMatchGame&) = delete;

  std::unique_ptr<mcts::IGameState> clone() const override;

  int actionCount() const override;
  bool isLegal(mcts::ActionId action) const override;
  void applyAction(mcts::ActionId action) override;

  mcts::PlayerId currentPlayer() const override;
  bool isTerminal() const override;
  double terminalUtility() const override;
  mcts::StateKey stateKey() const override;
  int moveCount() const override;

  std::string rulesKey() const override;
  mcts::StateKey canonicalStateKey() const override;
  mcts::MatchAction toMatchAction(mcts::ActionId action) const override;
  mcts::ActionId fromMatchAction(
      const mcts::MatchAction& action) const override;
  std::string terminalReason() const override;

  const std::vector<Action>& actions() const override;
  EncodedPosition encodePosition() const override;

  /// Concrete state for integrations that need engine-specific read access.
  const Chess5DGameState& game() const;

 private:
  Chess5DMatchGame(std::unique_ptr<Chess5DGameState> game,
                      bool reverse_move_order);

  void refreshActions();

  std::unique_ptr<Chess5DGameState> game_;
  bool reverse_move_order_ = false;
  std::vector<Action> actions_;
  std::vector<mcts::ActionId> local_to_game_action_;
};

}  // namespace suzero::chess5d
