// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-5DChess contributors

#include "suzero/5dchess/match_game.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace suzero::chess5d {

namespace {

void validateMaterialScale(const GameConfig& config) {
  if (!(config.material_scale > 0.0) ||
      !std::isfinite(config.material_scale)) {
    throw std::invalid_argument(
        "Chess5DMatchGame: material_scale must be positive and finite");
  }
}

std::unique_ptr<Chess5DGameState> makeValidatedGame(GameConfig config) {
  validateMaterialScale(config);
  return std::make_unique<Chess5DGameState>(std::move(config));
}

void appendSized(std::ostringstream& out, const std::string& value) {
  out << value.size() << ':';
  out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

std::string serializeRules(const GameConfig& config) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << "5dc-rules-v1|variant_pgn=";
  appendSized(out, config.variant_pgn);
  out << "|board_side=" << config.board_side
      << "|board_squares=" << config.board_squares
      << "|board_limit=" << config.board_limit << "|material_scale="
      << std::setprecision(std::numeric_limits<double>::max_digits10)
      << config.material_scale;
  return out.str();
}

mcts::MatchAction serializeAction(const Action& action) {
  if (action.is_submit) {
    return "5dc-action-v1:SUBMIT";
  }

  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << "5dc-action-v1:MOVE";
  for (std::int32_t coordinate : action.move) {
    out << ':' << coordinate;
  }
  return out.str();
}

}  // namespace

Chess5DMatchGame::Chess5DMatchGame(GameConfig config,
                                         bool reverse_move_order)
    : Chess5DMatchGame(makeValidatedGame(std::move(config)),
                          reverse_move_order) {}

Chess5DMatchGame::Chess5DMatchGame(
    std::unique_ptr<Chess5DGameState> game, bool reverse_move_order)
    : game_(std::move(game)), reverse_move_order_(reverse_move_order) {
  if (game_ == nullptr) {
    throw std::invalid_argument("Chess5DMatchGame: game must not be null");
  }
  validateMaterialScale(game_->config());
  refreshActions();
}

Chess5DMatchGame::~Chess5DMatchGame() = default;

std::unique_ptr<mcts::IGameState> Chess5DMatchGame::clone() const {
  return std::unique_ptr<mcts::IGameState>(
      new Chess5DMatchGame(game_->cloneState(), reverse_move_order_));
}

int Chess5DMatchGame::actionCount() const {
  return static_cast<int>(actions_.size());
}

bool Chess5DMatchGame::isLegal(mcts::ActionId action) const {
  if (action < 0 || static_cast<std::size_t>(action) >=
                        local_to_game_action_.size()) {
    return false;
  }
  return game_->isLegal(
      local_to_game_action_[static_cast<std::size_t>(action)]);
}

void Chess5DMatchGame::applyAction(mcts::ActionId action) {
  if (!isLegal(action)) {
    throw std::out_of_range("Chess5DMatchGame::applyAction: bad action id");
  }
  const mcts::ActionId game_action =
      local_to_game_action_[static_cast<std::size_t>(action)];
  game_->applyAction(game_action);
  refreshActions();
}

mcts::PlayerId Chess5DMatchGame::currentPlayer() const {
  return game_->currentPlayer();
}

bool Chess5DMatchGame::isTerminal() const { return game_->isTerminal(); }

double Chess5DMatchGame::terminalUtility() const {
  return game_->terminalUtility();
}

mcts::StateKey Chess5DMatchGame::stateKey() const {
  const mcts::StateKey canonical = canonicalStateKey();
  const GameConfig& config = game_->config();
  const std::string rules = rulesKey();

  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << "5dc-match-state-v1|rules=";
  appendSized(out, rules);
  out << "|encoding=board_side:" << config.board_side
      << ",board_squares:" << config.board_squares
      << ",piece_channels:" << config.piece_channels
      << ",line_shift:" << config.line_shift
      << "|move_order="
      << (reverse_move_order_ ? "reverse_non_submit" : "native")
      << "|canonical=";
  appendSized(out, canonical);
  return out.str();
}

int Chess5DMatchGame::moveCount() const { return game_->moveCount(); }

std::string Chess5DMatchGame::rulesKey() const {
  return serializeRules(game_->config());
}

mcts::StateKey Chess5DMatchGame::canonicalStateKey() const {
  // The engine discovers no-legal-action termination lazily while generating
  // actions. Force that transition before taking the underlying logical key.
  (void)game_->isTerminal();
  return game_->stateKey();
}

mcts::MatchAction Chess5DMatchGame::toMatchAction(
    mcts::ActionId action) const {
  if (!isLegal(action)) {
    throw std::out_of_range(
        "Chess5DMatchGame::toMatchAction: bad action id");
  }
  return serializeAction(actions_[static_cast<std::size_t>(action)]);
}

mcts::ActionId Chess5DMatchGame::fromMatchAction(
    const mcts::MatchAction& action) const {
  for (std::size_t i = 0; i < actions_.size(); ++i) {
    if (serializeAction(actions_[i]) == action) {
      return static_cast<mcts::ActionId>(i);
    }
  }
  return mcts::kInvalidAction;
}

std::string Chess5DMatchGame::terminalReason() const {
  if (!game_->isTerminal()) {
    return {};
  }
  return game_->terminalReason();
}

const std::vector<Action>& Chess5DMatchGame::actions() const {
  return actions_;
}

EncodedPosition Chess5DMatchGame::encodePosition() const {
  return game_->encodePosition();
}

const Chess5DGameState& Chess5DMatchGame::game() const { return *game_; }

void Chess5DMatchGame::refreshActions() {
  const std::vector<Action>& game_actions = game_->actions();

  std::vector<mcts::ActionId> move_actions;
  move_actions.reserve(game_actions.size());
  mcts::ActionId submit_action = mcts::kInvalidAction;
  for (std::size_t i = 0; i < game_actions.size(); ++i) {
    if (game_actions[i].is_submit) {
      if (submit_action != mcts::kInvalidAction ||
          i + 1 != game_actions.size()) {
        throw std::logic_error(
            "Chess5DGameState must list one SUBMIT after all move actions");
      }
      submit_action = static_cast<mcts::ActionId>(i);
    } else {
      move_actions.push_back(static_cast<mcts::ActionId>(i));
    }
  }

  if (reverse_move_order_) {
    std::reverse(move_actions.begin(), move_actions.end());
  }
  if (submit_action != mcts::kInvalidAction) {
    move_actions.push_back(submit_action);
  }

  std::vector<Action> refreshed;
  refreshed.reserve(move_actions.size());
  for (mcts::ActionId game_action : move_actions) {
    refreshed.push_back(
        game_actions[static_cast<std::size_t>(game_action)]);
  }

  local_to_game_action_.swap(move_actions);
  actions_.swap(refreshed);
}

}  // namespace suzero::chess5d
