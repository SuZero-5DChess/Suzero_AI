// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-5DChess contributors

#include "suzero/mcts/match.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <typeindex>
#include <utility>

#include "suzero/mcts/puct_search.h"

namespace suzero::mcts {

namespace {

constexpr double kTerminalUtilityTolerance = 1e-9;

/// Search also clones states internally. Guard that boundary before a model
/// receives a clone that has silently lost its adapter or input encoding type.
class MatchEvaluator final : public IEvaluator {
 public:
  MatchEvaluator(std::shared_ptr<IEvaluator> evaluator, const IMatchGame& root)
      : evaluator_(std::move(evaluator)), adapter_type_(typeid(root)) {}

  Evaluation evaluate(const IGameState& state) override {
    if (dynamic_cast<const IMatchGame*>(&state) == nullptr ||
        std::type_index(typeid(state)) != adapter_type_) {
      throw std::runtime_error("MatchRunner: clone changed the game adapter type");
    }
    return evaluator_->evaluate(state);
  }

 private:
  std::shared_ptr<IEvaluator> evaluator_;
  std::type_index adapter_type_;
};

std::uint64_t deriveSeed(std::uint64_t seed, std::uint64_t index) {
  // SplitMix64 finalizer: nearby games and the two searches receive unrelated,
  // reproducible random streams.
  std::uint64_t z = seed +
                    0x9E3779B97F4A7C15ULL * (index + 1);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

void validateSearchParams(const SearchParams& params, int participant_index) {
  const std::string prefix =
      "MatchRunner: participant " + std::to_string(participant_index) + " ";
  auto require = [&prefix](bool condition, const char* message) {
    if (!condition) {
      throw std::invalid_argument(prefix + message);
    }
  };

  require(params.maxVisits >= 0, "maxVisits must be >= 0");
  require(params.maxPlayouts >= 0, "maxPlayouts must be >= 0");
  require(std::isfinite(params.maxTimeSeconds) &&
              params.maxTimeSeconds >= 0.0,
          "maxTimeSeconds must be finite and >= 0");
  require(params.maxDepth >= 0, "maxDepth must be >= 0");
  require(params.maxVisits > 0 || params.maxPlayouts > 0 ||
              params.maxTimeSeconds > 0.0,
          "search must have at least one positive limit");

  require(std::isfinite(params.cpuctExploration) &&
              params.cpuctExploration >= 0.0,
          "cpuctExploration must be finite and >= 0");
  require(std::isfinite(params.cpuctExplorationLog) &&
              params.cpuctExplorationLog >= 0.0,
          "cpuctExplorationLog must be finite and >= 0");
  require(std::isfinite(params.cpuctExplorationBase) &&
              params.cpuctExplorationBase > 0.0,
          "cpuctExplorationBase must be finite and > 0");
  require(std::isfinite(params.cpuctUtilityStdevPrior) &&
              params.cpuctUtilityStdevPrior > 0.0,
          "cpuctUtilityStdevPrior must be finite and > 0");
  require(std::isfinite(params.cpuctUtilityStdevPriorWeight) &&
              params.cpuctUtilityStdevPriorWeight > 0.0,
          "cpuctUtilityStdevPriorWeight must be finite and > 0");
  require(std::isfinite(params.cpuctUtilityStdevScale) &&
              params.cpuctUtilityStdevScale >= 0.0,
          "cpuctUtilityStdevScale must be finite and >= 0");
  require(std::isfinite(params.utilityRadius) && params.utilityRadius > 0.0,
          "utilityRadius must be finite and > 0");

  require(std::isfinite(params.fpuReductionMax) &&
              params.fpuReductionMax >= 0.0,
          "fpuReductionMax must be finite and >= 0");
  require(std::isfinite(params.fpuLossProp) && params.fpuLossProp >= 0.0 &&
              params.fpuLossProp <= 1.0,
          "fpuLossProp must be finite and in [0, 1]");
  require(std::isfinite(params.rootFpuReductionMax) &&
              params.rootFpuReductionMax >= 0.0,
          "rootFpuReductionMax must be finite and >= 0");
  require(std::isfinite(params.rootFpuLossProp) &&
              params.rootFpuLossProp >= 0.0 &&
              params.rootFpuLossProp <= 1.0,
          "rootFpuLossProp must be finite and in [0, 1]");
  require(std::isfinite(params.fpuParentWeightByVisitedPolicyPow) &&
              params.fpuParentWeightByVisitedPolicyPow > 0.0,
          "fpuParentWeightByVisitedPolicyPow must be finite and > 0");
  require(std::isfinite(params.fpuParentWeight) &&
              params.fpuParentWeight >= 0.0 &&
              params.fpuParentWeight <= 1.0,
          "fpuParentWeight must be finite and in [0, 1]");

  require(std::isfinite(params.rootDirichletNoiseWeight) &&
              params.rootDirichletNoiseWeight >= 0.0 &&
              params.rootDirichletNoiseWeight <= 1.0,
          "rootDirichletNoiseWeight must be finite and in [0, 1]");
  if (params.rootNoiseEnabled) {
    require(std::isfinite(params.rootDirichletNoiseTotalConcentration) &&
                params.rootDirichletNoiseTotalConcentration > 0.0,
            "rootDirichletNoiseTotalConcentration must be finite and > 0 when "
            "noise is enabled");
  }
  require(std::isfinite(params.rootPolicyTemperature) &&
              params.rootPolicyTemperature > 0.0,
          "rootPolicyTemperature must be finite and > 0");
  require(std::isfinite(params.rootPolicyTemperatureEarly) &&
              params.rootPolicyTemperatureEarly > 0.0,
          "rootPolicyTemperatureEarly must be finite and > 0");

  require(std::isfinite(params.chosenMoveTemperature) &&
              params.chosenMoveTemperature >= 0.0,
          "chosenMoveTemperature must be finite and >= 0");
  require(std::isfinite(params.chosenMoveTemperatureEarly) &&
              params.chosenMoveTemperatureEarly >= 0.0,
          "chosenMoveTemperatureEarly must be finite and >= 0");
  require(std::isfinite(params.chosenMoveTemperatureHalflife) &&
              params.chosenMoveTemperatureHalflife > 0.0,
          "chosenMoveTemperatureHalflife must be finite and > 0");
  require(std::isfinite(params.chosenMoveTemperatureOnlyBelowProb) &&
              params.chosenMoveTemperatureOnlyBelowProb > 0.0 &&
              params.chosenMoveTemperatureOnlyBelowProb <= 1.0,
          "chosenMoveTemperatureOnlyBelowProb must be finite and in (0, 1]");
  require(std::isfinite(params.chosenMoveSubtract) &&
              params.chosenMoveSubtract >= 0.0,
          "chosenMoveSubtract must be finite and >= 0");
  require(std::isfinite(params.chosenMovePrune) &&
              params.chosenMovePrune >= 0.0,
          "chosenMovePrune must be finite and >= 0");
  require(std::isfinite(params.valueWeightExponent) &&
              params.valueWeightExponent >= 0.0,
          "valueWeightExponent must be finite and >= 0");
  require(std::isfinite(params.graphSearchCatchUpLeakProb) &&
              params.graphSearchCatchUpLeakProb >= 0.0 &&
              params.graphSearchCatchUpLeakProb <= 1.0,
          "graphSearchCatchUpLeakProb must be finite and in [0, 1]");
}

void validateRunnerConfig(
    const MatchConfig& config,
    const std::array<MatchParticipant, 2>& participants) {
  if (config.maxMoves < 0) {
    throw std::invalid_argument("MatchRunner: maxMoves must be non-negative");
  }
  if (participants[0].player == participants[1].player) {
    throw std::invalid_argument(
        "MatchRunner: participant PlayerIds must be distinct");
  }

  for (int i = 0; i < 2; ++i) {
    const MatchParticipant& participant =
        participants[static_cast<std::size_t>(i)];
    if (!participant.stateFactory) {
      throw std::invalid_argument("MatchRunner: participant " +
                                  std::to_string(i) +
                                  " stateFactory is required");
    }
    if (!participant.evaluator) {
      throw std::invalid_argument("MatchRunner: participant " +
                                  std::to_string(i) +
                                  " evaluator is required");
    }
    validateSearchParams(participant.searchParams, i);
  }
}

bool sharesOwnership(const std::shared_ptr<IMatchGame>& lhs,
                     const std::shared_ptr<IMatchGame>& rhs) {
  return !lhs.owner_before(rhs) && !rhs.owner_before(lhs);
}

struct StatePairStatus {
  PlayerId currentPlayer = 0;
  bool terminal = false;
  double terminalUtility = 0.0;
  std::string terminalReason;
};

bool utilitiesAgree(double lhs, double rhs) {
  const double scale = std::max({1.0, std::fabs(lhs), std::fabs(rhs)});
  return std::fabs(lhs - rhs) <= kTerminalUtilityTolerance * scale;
}

StatePairStatus validateStatePair(
    const std::array<const IMatchGame*, 2>& states,
    const std::array<MatchParticipant, 2>& participants,
    const std::string* expected_rules_key) {
  if (states[0] == nullptr || states[1] == nullptr) {
    throw std::runtime_error("MatchRunner: match state is null");
  }

  const std::string rules_key_0 = states[0]->rulesKey();
  const std::string rules_key_1 = states[1]->rulesKey();
  if (rules_key_0 != rules_key_1) {
    throw std::runtime_error("MatchRunner: participant rulesKey values differ");
  }
  if (expected_rules_key != nullptr &&
      rules_key_0 != *expected_rules_key) {
    throw std::runtime_error("MatchRunner: rulesKey changed during the game");
  }

  if (states[0]->canonicalStateKey() != states[1]->canonicalStateKey()) {
    throw std::runtime_error(
        "MatchRunner: participant canonicalStateKey values differ");
  }

  const PlayerId player_0 = states[0]->currentPlayer();
  const PlayerId player_1 = states[1]->currentPlayer();
  if (player_0 != player_1) {
    throw std::runtime_error(
        "MatchRunner: participant currentPlayer values differ");
  }
  if (player_0 != participants[0].player &&
      player_0 != participants[1].player) {
    throw std::runtime_error(
        "MatchRunner: currentPlayer does not identify either participant");
  }

  const bool terminal_0 = states[0]->isTerminal();
  const bool terminal_1 = states[1]->isTerminal();
  if (terminal_0 != terminal_1) {
    throw std::runtime_error(
        "MatchRunner: participant terminal states differ");
  }

  StatePairStatus status;
  status.currentPlayer = player_0;
  status.terminal = terminal_0;
  if (!status.terminal) {
    return status;
  }

  const double utility_0 = states[0]->terminalUtility();
  const double utility_1 = states[1]->terminalUtility();
  if (!std::isfinite(utility_0) || !std::isfinite(utility_1)) {
    throw std::runtime_error(
        "MatchRunner: terminalUtility must be finite for both participants");
  }
  if (!utilitiesAgree(utility_0, utility_1)) {
    throw std::runtime_error(
        "MatchRunner: participant terminalUtility values differ");
  }
  status.terminalUtility = utility_0;

  const std::string reason_0 = states[0]->terminalReason();
  const std::string reason_1 = states[1]->terminalReason();
  if (!reason_0.empty() && !reason_1.empty() && reason_0 != reason_1) {
    throw std::runtime_error(
        "MatchRunner: participant terminalReason values differ");
  }
  status.terminalReason = reason_0.empty() ? reason_1 : reason_0;
  return status;
}

std::array<const IMatchGame*, 2> matchRootStates(
    const std::array<std::unique_ptr<PuctSearch>, 2>& searches) {
  std::array<const IMatchGame*, 2> states{{nullptr, nullptr}};
  for (int i = 0; i < 2; ++i) {
    states[static_cast<std::size_t>(i)] =
        dynamic_cast<const IMatchGame*>(
            &searches[static_cast<std::size_t>(i)]->rootState());
    if (states[static_cast<std::size_t>(i)] == nullptr) {
      throw std::runtime_error(
          "MatchRunner: IMatchGame::clone did not preserve IMatchGame");
    }
  }
  return states;
}

std::unique_ptr<IGameState> applyRootAction(PuctSearch& search, ActionId action) {
  if (search.advanceRoot(action)) {
    if (dynamic_cast<const IMatchGame*>(&search.rootState()) == nullptr) {
      throw std::runtime_error(
          "MatchRunner: IMatchGame::clone did not preserve IMatchGame");
    }
    return {};
  }

  std::unique_ptr<IGameState> next = search.rootState().clone();
  if (!next) {
    throw std::runtime_error("MatchRunner: IMatchGame::clone returned null");
  }
  if (dynamic_cast<IMatchGame*>(next.get()) == nullptr) {
    throw std::runtime_error(
        "MatchRunner: IMatchGame::clone did not preserve IMatchGame");
  }
  if (!next->isLegal(action)) {
    throw std::runtime_error(
        "MatchRunner: cloned state rejected a previously legal action");
  }
  next->applyAction(action);
  // A terminal dynamic-action game may have no candidates. PuctSearch's
  // resetRoot requires a positive actionCount, so retain that final snapshot
  // outside the search rather than constructing a graph that will never run.
  if (next->isTerminal()) {
    return next;
  }
  search.resetRoot(std::shared_ptr<IGameState>(std::move(next)));
  return {};
}

MatchResult terminalResult(
    MatchResult result, const StatePairStatus& status,
    const std::array<MatchParticipant, 2>& participants) {
  result.terminal = true;
  result.truncated = false;
  result.finalPlayer = status.currentPlayer;
  result.finalUtility = status.terminalUtility;
  result.terminalReason = status.terminalReason;
  for (int i = 0; i < 2; ++i) {
    result.participantUtilities[static_cast<std::size_t>(i)] =
        participants[static_cast<std::size_t>(i)].player == status.currentPlayer
            ? status.terminalUtility
            : -status.terminalUtility;
  }
  return result;
}

void addScore(MatchScore& score, const MatchResult& game,
              int participant_index) {
  if (game.truncated) {
    ++score.truncated;
    return;
  }
  if (!game.terminal ||
      !game.participantUtilities[static_cast<std::size_t>(participant_index)]
           .has_value()) {
    throw std::logic_error(
        "runMatchSeries: completed game has no participant utility");
  }

  const double utility =
      *game.participantUtilities[static_cast<std::size_t>(participant_index)];
  if (utility > 0.0) {
    ++score.wins;
  } else if (utility < 0.0) {
    ++score.losses;
  } else {
    ++score.draws;
  }
}

}  // namespace

MatchRunner::MatchRunner(MatchConfig config,
                         std::array<MatchParticipant, 2> participants)
    : config_(std::move(config)), participants_(std::move(participants)) {
  validateRunnerConfig(config_, participants_);
}

MatchResult MatchRunner::run() const {
  std::array<std::shared_ptr<IMatchGame>, 2> initial_states;
  for (int i = 0; i < 2; ++i) {
    initial_states[static_cast<std::size_t>(i)] =
        participants_[static_cast<std::size_t>(i)].stateFactory();
    if (!initial_states[static_cast<std::size_t>(i)]) {
      throw std::runtime_error("MatchRunner: participant " +
                               std::to_string(i) +
                               " stateFactory returned null");
    }
    if (initial_states[static_cast<std::size_t>(i)].use_count() != 1) {
      throw std::runtime_error(
          "MatchRunner: stateFactory must return a fresh independently owned "
          "state");
    }
  }
  if (initial_states[0].get() == initial_states[1].get() ||
      sharesOwnership(initial_states[0], initial_states[1])) {
    throw std::runtime_error(
        "MatchRunner: participant factories returned aliased states");
  }

  const std::array<const IMatchGame*, 2> initial_state_ptrs{
      {initial_states[0].get(), initial_states[1].get()}};
  const StatePairStatus initial_status =
      validateStatePair(initial_state_ptrs, participants_, nullptr);

  MatchResult result;
  if (initial_status.terminal) {
    return terminalResult(std::move(result), initial_status, participants_);
  }

  const std::string rules_key = initial_states[0]->rulesKey();
  std::array<std::unique_ptr<PuctSearch>, 2> searches;
  for (int i = 0; i < 2; ++i) {
    searches[static_cast<std::size_t>(i)] = std::make_unique<PuctSearch>(
        initial_states[static_cast<std::size_t>(i)],
        std::make_shared<MatchEvaluator>(
            participants_[static_cast<std::size_t>(i)].evaluator,
            *initial_states[static_cast<std::size_t>(i)]),
        participants_[static_cast<std::size_t>(i)].searchParams,
        deriveSeed(config_.seed, static_cast<std::uint64_t>(i)));
  }
  initial_states[0].reset();
  initial_states[1].reset();

  for (;;) {
    const std::array<const IMatchGame*, 2> states = matchRootStates(searches);
    const StatePairStatus status =
        validateStatePair(states, participants_, &rules_key);

    // Terminal state wins over the move cap, including immediately after the
    // action that reaches maxMoves.
    if (status.terminal) {
      return terminalResult(std::move(result), status, participants_);
    }
    if (config_.maxMoves > 0 && result.moves.size() >=
        static_cast<std::size_t>(config_.maxMoves)) {
      result.terminal = false;
      result.truncated = true;
      result.finalPlayer = status.currentPlayer;
      return result;
    }

    const int acting_index =
        status.currentPlayer == participants_[0].player ? 0 : 1;
    PuctSearch& acting_search =
        *searches[static_cast<std::size_t>(acting_index)];
    (void)acting_search.run();
    const ActionId selected_action = acting_search.chooseAction();

    const IMatchGame& acting_state =
        *states[static_cast<std::size_t>(acting_index)];
    if (selected_action == kInvalidAction || selected_action < 0 ||
        selected_action >= acting_state.actionCount() ||
        !acting_state.isLegal(selected_action)) {
      throw std::runtime_error(
          "MatchRunner: active search selected no legal action");
    }

    const MatchAction match_action =
        acting_state.toMatchAction(selected_action);
    std::array<ActionId, 2> local_actions{{kInvalidAction, kInvalidAction}};
    for (int i = 0; i < 2; ++i) {
      const IMatchGame& state = *states[static_cast<std::size_t>(i)];
      const ActionId local_action = state.fromMatchAction(match_action);
      if (local_action == kInvalidAction || local_action < 0 ||
          local_action >= state.actionCount()) {
        throw std::runtime_error(
            "MatchRunner: logical action cannot be mapped by participant " +
            std::to_string(i));
      }
      if (!state.isLegal(local_action)) {
        throw std::runtime_error(
            "MatchRunner: mapped action is illegal for participant " +
            std::to_string(i));
      }
      if (state.toMatchAction(local_action) != match_action) {
        throw std::runtime_error(
            "MatchRunner: logical action mapping does not round-trip for "
            "participant " +
            std::to_string(i));
      }
      local_actions[static_cast<std::size_t>(i)] = local_action;
    }
    if (local_actions[static_cast<std::size_t>(acting_index)] !=
        selected_action) {
      throw std::runtime_error(
          "MatchRunner: active participant action mapping does not round-trip");
    }

    std::array<std::unique_ptr<IGameState>, 2> final_states;
    for (int i = 0; i < 2; ++i) {
      final_states[static_cast<std::size_t>(i)] =
          applyRootAction(*searches[static_cast<std::size_t>(i)],
                          local_actions[static_cast<std::size_t>(i)]);
    }

    MatchMove move;
    move.player = status.currentPlayer;
    move.participantIndex = acting_index;
    move.action = match_action;
    move.localActions = local_actions;
    result.moves.push_back(std::move(move));

    if (final_states[0] || final_states[1]) {
      auto advanced = matchRootStates(searches);
      for (std::size_t i = 0; i < final_states.size(); ++i) {
        if (final_states[i]) {
          advanced[i] = dynamic_cast<const IMatchGame*>(final_states[i].get());
        }
      }
      const auto final_status = validateStatePair(advanced, participants_, &rules_key);
      return terminalResult(std::move(result), final_status, participants_);
    }
  }
}

MatchSeriesResult runMatchSeries(
    const MatchSeriesConfig& config,
    const std::array<MatchParticipant, 2>& participants,
    const MatchGameCallback& onGameFinished) {
  if (config.pairs <= 0) {
    throw std::invalid_argument("runMatchSeries: pairs must be positive");
  }
  if (config.pairs > std::numeric_limits<int>::max() / 2) {
    throw std::invalid_argument("runMatchSeries: pairs * 2 overflows int");
  }
  if (config.numGameThreads <= 0) {
    throw std::invalid_argument("runMatchSeries: numGameThreads must be positive");
  }

  // Validate every static setting before reserving storage or starting a game.
  validateRunnerConfig(config.game, participants);

  const int game_count = config.pairs * 2;
  MatchSeriesResult series;
  const std::size_t game_count_size = static_cast<std::size_t>(game_count);
  if (game_count_size > series.games.max_size()) {
    throw std::length_error("runMatchSeries: game count is too large");
  }
  series.games.resize(game_count_size);

  auto play_game = [&](int game_index) {
    std::array<MatchParticipant, 2> game_participants = participants;
    if (game_index % 2 != 0) {
      std::swap(game_participants[0].player, game_participants[1].player);
    }
    MatchConfig game_config = config.game;
    game_config.seed =
        deriveSeed(config.game.seed, static_cast<std::uint64_t>(game_index));
    return MatchRunner(game_config, std::move(game_participants)).run();
  };
  auto record_game = [&](int game_index) {
    const MatchResult& game = series.games[static_cast<std::size_t>(game_index)];
    for (int participant_index = 0; participant_index < 2;
         ++participant_index) {
      addScore(series.totals[static_cast<std::size_t>(participant_index)], game,
               participant_index);
      const int seat_index =
          game_index % 2 != 0 ? 1 - participant_index : participant_index;
      addScore(
          series.bySeat[static_cast<std::size_t>(participant_index)]
                       [static_cast<std::size_t>(seat_index)],
          game, participant_index);
    }
    if (onGameFinished) {
      onGameFinished(game_index, game);
    }
  };

  const int thread_count = std::min(config.numGameThreads, game_count);
  if (thread_count == 1) {
    for (int index = 0; index < game_count; ++index) {
      series.games[static_cast<std::size_t>(index)] = play_game(index);
      record_game(index);
    }
    return series;
  }

  std::mutex mutex;
  std::condition_variable ready;
  std::deque<int> finished;
  int next_index = 0;
  bool stopped = false;
  std::exception_ptr first_error;
  auto worker = [&]() {
    try {
      for (;;) {
        int index;
        {
          std::lock_guard<std::mutex> lock(mutex);
          if (stopped || next_index >= game_count) {
            return;
          }
          index = next_index++;
        }
        // Each worker owns a different slot. Publishing its index under the
        // mutex makes the completed result visible to the calling thread.
        series.games[static_cast<std::size_t>(index)] = play_game(index);
        {
          std::lock_guard<std::mutex> lock(mutex);
          finished.push_back(index);
        }
        ready.notify_one();
      }
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(mutex);
        if (!first_error) {
          first_error = std::current_exception();
        }
        stopped = true;
      }
      ready.notify_one();
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(thread_count));
  auto join = [&]() {
    for (auto& thread : threads) {
      thread.join();
    }
  };
  try {
    for (int i = 0; i < thread_count; ++i) {
      threads.emplace_back(worker);
    }
    for (int done = 0; done < game_count; ++done) {
      int index;
      {
        std::unique_lock<std::mutex> lock(mutex);
        ready.wait(lock, [&]() { return first_error || !finished.empty(); });
        if (first_error) {
          std::rethrow_exception(first_error);
        }
        index = finished.front();
        finished.pop_front();
      }
      // Keep aggregation and callbacks on the calling thread, including
      // Python signal checks, instead of invoking Python from game threads.
      record_game(index);
    }
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopped = true;
    }
    join();
    throw;
  }
  join();
  return series;
}

}  // namespace suzero::mcts
