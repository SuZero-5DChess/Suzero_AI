// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-5DChess contributors
//
// Connects the three pieces of the 5D chess integration:
//   Chess5DGameState (IGameState) + OnnxEvaluator (IEvaluator) + PuctSearch.
//
// The agent owns a root position and the search tree, and exposes the
// reference runner's action-selection policy: a forced SUBMIT needs no network
// call, otherwise PUCT runs with the reference defaults and the move is sampled
// with a hard temperature schedule.

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "suzero/5dchess/game_state.h"
#include "suzero/5dchess/onnx_evaluator.h"
#include "suzero/mcts/puct_search.h"
#include "suzero/mcts/search_params.h"

namespace suzero::chess5d {

/// Search parameters matching the 5dc_ml C++ runner defaults (SearchConfig):
/// 200 simulations and c_puct = 2.0. Draw/temperature settings live on
/// AgentConfig because the runner applies them as a hard move-count switch.
inline mcts::SearchParams DefaultSearchParams() {
  mcts::SearchParams params;
  params.maxPlayouts = 200;
  params.cpuctExploration = 2.0;
  params.chosenMoveTemperature = 0.1;
  params.chosenMoveTemperatureEarly = 1.0;
  params.chosenMoveTemperatureHalflife = 30.0;
  return params;
}

struct AgentConfig {
  GameConfig game;

  // ONNX evaluator.
  std::string model_path;
  std::string metadata_path;
  std::string provider = "cpu";
  int intra_op_threads = 1;

  // PUCT search.
  mcts::SearchParams search = DefaultSearchParams();
  std::uint64_t seed = 1;

  // Reference temperature schedule: use `temperature` while the number of
  // small steps played is below `temperature_threshold`, then
  // `temperature_final` (1.0 -> 0.1 after 30 steps by default).
  double temperature = 1.0;
  double temperature_final = 0.1;
  int temperature_threshold = 30;
};

/// One action-selection result.
struct Decision {
  mcts::ActionId actionId = mcts::kInvalidAction;
  Action action;

  /// Search statistics. Empty for a forced action (no search was run).
  mcts::SearchResult searchResult;

  /// True when the position had exactly one candidate action (forced SUBMIT).
  bool forced = false;

  /// Temperature used for sampling (0 for a forced action).
  double temperature = 0.0;
};

/// Stateful 5D chess player: game state + neural evaluator + PUCT search.
///
/// Actions are small steps; `applyAction` mirrors them into the search tree by
/// reusing the chosen subtree when it exists and resetting the root otherwise.
/// The agent is single-threaded and must not be used concurrently.
class Chess5DAgent {
 public:
  explicit Chess5DAgent(AgentConfig config);
  ~Chess5DAgent();

  Chess5DAgent(const Chess5DAgent&) = delete;
  Chess5DAgent& operator=(const Chess5DAgent&) = delete;

  const AgentConfig& config() const;
  const Chess5DGameState& state() const;
  const OnnxEvaluator& evaluator() const;
  mcts::PuctSearch& search();
  const mcts::PuctSearch& search() const;

  /// Restarts from the initial position and clears the search tree.
  void reset();

  /// Applies a candidate index. Advances the search root when possible.
  void applyAction(mcts::ActionId action);
  void applyAction(const Action& action);

  /// Selects an action using the reference temperature schedule.
  Decision selectAction();

  /// Selects an action with an explicit temperature (no schedule).
  Decision selectAction(double temperature);

  /// Selects an action and applies it (the reference play-game step).
  Decision play();
  Decision play(double temperature);

 private:
  std::shared_ptr<mcts::IGameState> cloneForRoot() const;

  AgentConfig config_;
  std::shared_ptr<Chess5DGameState> state_;
  std::shared_ptr<OnnxEvaluator> evaluator_;
  std::unique_ptr<mcts::PuctSearch> search_;
};

}  // namespace suzero::chess5d
