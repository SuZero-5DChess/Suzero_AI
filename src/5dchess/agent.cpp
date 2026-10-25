// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-5DChess contributors

#include "suzero/5dchess/agent.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace suzero::chess5d {

Chess5DAgent::Chess5DAgent(AgentConfig config)
    : config_(std::move(config)),
      state_(std::make_shared<Chess5DGameState>(config_.game)) {
  if (config_.model_path.empty()) {
    throw std::invalid_argument("Chess5DAgent: model_path is required");
  }
  if (config_.temperature_threshold < 0) {
    throw std::invalid_argument(
        "Chess5DAgent: temperature_threshold must be non-negative");
  }
  for (const double value :
       {config_.temperature, config_.temperature_final}) {
    if (!(value >= 0.0) || !std::isfinite(value)) {
      throw std::invalid_argument(
          "Chess5DAgent: temperatures must be finite and non-negative");
    }
  }

  OnnxEvaluatorConfig evaluator_config;
  evaluator_config.model_path = config_.model_path;
  evaluator_config.metadata_path = config_.metadata_path;
  evaluator_config.provider = config_.provider;
  evaluator_config.intra_op_threads = config_.intra_op_threads;
  evaluator_config.board_side = config_.game.board_side;
  evaluator_config.board_squares = config_.game.board_squares;
  evaluator_config.piece_channels = config_.game.piece_channels;
  evaluator_ = std::make_shared<OnnxEvaluator>(std::move(evaluator_config));

  search_ = std::make_unique<mcts::PuctSearch>(
      cloneForRoot(), evaluator_, config_.search, config_.seed);
}

Chess5DAgent::~Chess5DAgent() = default;

const AgentConfig& Chess5DAgent::config() const { return config_; }

const Chess5DGameState& Chess5DAgent::state() const { return *state_; }

const OnnxEvaluator& Chess5DAgent::evaluator() const { return *evaluator_; }

mcts::PuctSearch& Chess5DAgent::search() { return *search_; }

const mcts::PuctSearch& Chess5DAgent::search() const { return *search_; }

std::shared_ptr<mcts::IGameState> Chess5DAgent::cloneForRoot() const {
  return std::shared_ptr<mcts::IGameState>(state_->cloneState());
}

void Chess5DAgent::reset() {
  state_ = std::make_shared<Chess5DGameState>(config_.game);
  search_->resetRoot(cloneForRoot());
}

void Chess5DAgent::applyAction(mcts::ActionId action) {
  if (state_->isTerminal()) {
    throw std::logic_error("Chess5DAgent::applyAction: game is over");
  }
  if (action < 0 || action >= state_->actionCount()) {
    throw std::out_of_range(
        "Chess5DAgent::applyAction: action is not a candidate");
  }

  // Promote the subtree while the search root still describes the same
  // position as state_. A forced or unexpanded action falls back to a reset.
  const bool advanced = search_->advanceRoot(action);
  state_->applyAction(action);

  if (!state_->isTerminal() && !advanced) {
    search_->resetRoot(cloneForRoot());
  }
}

void Chess5DAgent::applyAction(const Action& action) {
  const auto& candidates = state_->actions();
  const auto found = std::find(candidates.begin(), candidates.end(), action);
  if (found == candidates.end()) {
    throw std::out_of_range(
        "Chess5DAgent::applyAction: action is not a candidate");
  }
  applyAction(
      static_cast<mcts::ActionId>(std::distance(candidates.begin(), found)));
}

Decision Chess5DAgent::selectAction() {
  const double temperature =
      state_->moveCount() < config_.temperature_threshold
          ? config_.temperature
          : config_.temperature_final;
  return selectAction(temperature);
}

Decision Chess5DAgent::selectAction(double temperature) {
  if (!(temperature >= 0.0) || !std::isfinite(temperature)) {
    throw std::invalid_argument(
        "Chess5DAgent::selectAction: invalid temperature");
  }
  if (state_->isTerminal()) {
    throw std::logic_error("Chess5DAgent::selectAction: game is over");
  }

  const auto& candidates = state_->actions();
  if (candidates.empty()) {
    throw std::runtime_error("Chess5DAgent::selectAction: no candidates");
  }

  Decision decision;
  decision.temperature = temperature;

  // The reference runner skips the network and search when SUBMIT is the only
  // legal action.
  if (candidates.size() == 1) {
    decision.actionId = 0;
    decision.action = candidates.front();
    decision.forced = true;
    decision.temperature = 0.0;
    return decision;
  }

  decision.searchResult = search_->run();
  const mcts::ActionId chosen = search_->chooseAction(temperature);
  if (chosen < 0 || chosen >= state_->actionCount()) {
    throw std::runtime_error(
        "Chess5DAgent::selectAction: search returned no action");
  }
  decision.actionId = chosen;
  decision.action = state_->action(chosen);
  return decision;
}

Decision Chess5DAgent::play() {
  const Decision decision = selectAction();
  applyAction(decision.actionId);
  return decision;
}

Decision Chess5DAgent::play(double temperature) {
  const Decision decision = selectAction(temperature);
  applyAction(decision.actionId);
  return decision;
}

}  // namespace suzero::chess5d
