// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-02 contributors
//
// Standalone tests for heterogeneous matches. These use two deliberately
// different toy adapters and do not depend on any external game engine.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "suzero/mcts/match.h"

namespace {

using suzero::mcts::ActionId;
using suzero::mcts::Evaluation;
using suzero::mcts::IEvaluator;
using suzero::mcts::IGameState;
using suzero::mcts::IMatchGame;
using suzero::mcts::MatchAction;
using suzero::mcts::MatchConfig;
using suzero::mcts::MatchMove;
using suzero::mcts::MatchParticipant;
using suzero::mcts::MatchResult;
using suzero::mcts::MatchRunner;
using suzero::mcts::MatchScore;
using suzero::mcts::MatchSeriesConfig;
using suzero::mcts::MatchSeriesResult;
using suzero::mcts::PlayerId;
using suzero::mcts::SearchParams;
using suzero::mcts::StateKey;
using suzero::mcts::kInvalidAction;
using suzero::mcts::runMatchSeries;

int g_failures = 0;
int g_checks = 0;

#define CHECK(condition)                                                     \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(condition)) {                                                      \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": "            \
                << #condition << "\n";                                       \
      ++g_failures;                                                          \
    }                                                                        \
  } while (0)

#define CHECK_NEAR(actual, expected, tolerance)                              \
  do {                                                                       \
    ++g_checks;                                                              \
    const double suzero_check_actual = (actual);                             \
    const double suzero_check_expected = (expected);                         \
    if (!(std::fabs(suzero_check_actual - suzero_check_expected) <=          \
          (tolerance))) {                                                    \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": "            \
                << #actual << " = " << suzero_check_actual                   \
                << ", expected " << suzero_check_expected << " +/- "         \
                << (tolerance) << "\n";                                      \
      ++g_failures;                                                          \
    }                                                                        \
  } while (0)

template <typename Function>
bool throwsAnything(Function&& function) {
  try {
    function();
  } catch (...) {
    return true;
  }
  return false;
}

SearchParams rootEvaluationOnlyParams() {
  SearchParams params;
  params.maxVisits = 1;
  params.maxPlayouts = 0;
  params.maxTimeSeconds = 0.0;
  params.chosenMoveTemperature = 0.0;
  return params;
}

SearchParams twoPlayoutParams() {
  SearchParams params;
  params.maxVisits = 0;
  params.maxPlayouts = 2;
  params.maxTimeSeconds = 0.0;
  params.chosenMoveTemperature = 0.0;
  return params;
}

struct ToySpec {
  PlayerId firstPlayer = 17;
  PlayerId secondPlayer = -4;
  PlayerId terminalPlayer = 17;
  int initialStep = 0;
  int terminalStep = 3;
  double terminalUtility = 0.75;
  std::string rules = "heterogeneous-toy-v1";
  std::string terminalReason = "script-complete";
};

enum class MappingFault {
  kNone,
  kUnmappable,
  kMappedToIllegalAction,
  kRoundTripMismatch,
};

struct AdapterBehavior {
  bool overrideRules = false;
  std::string rules;
  bool divergeCanonicalInitially = false;
  bool divergeCanonicalAfterMove = false;
  bool unknownPlayer = false;
  bool noLegalActions = false;
  bool cloneDropsMatchInterface = false;
  MappingFault mappingFault = MappingFault::kNone;
  bool overrideTerminalUtility = false;
  double terminalUtility = 0.0;
  bool overrideTerminalReason = false;
  std::string terminalReason;
};

const std::array<MatchAction, 3>& localOrdering(bool adapter_b) {
  static const std::array<MatchAction, 3> kAdapterA{{"alpha", "beta",
                                                     "gamma"}};
  static const std::array<MatchAction, 3> kAdapterB{{"gamma", "alpha",
                                                     "beta"}};
  return adapter_b ? kAdapterB : kAdapterA;
}

MatchAction requiredAction(int step) {
  switch (step) {
    case 0:
      return "alpha";
    case 1:
      return "beta";
    case 2:
      return "gamma";
    default:
      return {};
  }
}

struct ToyPosition {
  std::shared_ptr<const ToySpec> spec;
  AdapterBehavior behavior;
  int step = 0;
  bool adapterB = false;

  // Terminal states have no candidates, as in dynamic-action games like 5dc.
  int actionCount() const { return isTerminal() ? 0 : 3; }

  bool isTerminal() const { return step >= spec->terminalStep; }

  PlayerId currentPlayer() const {
    if (behavior.unknownPlayer) {
      return 1000003;
    }
    if (isTerminal()) {
      return spec->terminalPlayer;
    }
    if (step <= 1) {
      return spec->firstPlayer;
    }
    return spec->secondPlayer;
  }

  MatchAction normalMatchAction(ActionId action) const {
    if (action < 0 || action >= actionCount()) {
      return "invalid-local-action";
    }
    return localOrdering(adapterB)[static_cast<std::size_t>(action)];
  }

  ActionId normalLocalAction(const MatchAction& action) const {
    const auto& ordering = localOrdering(adapterB);
    for (std::size_t i = 0; i < ordering.size(); ++i) {
      if (ordering[i] == action) {
        return static_cast<ActionId>(i);
      }
    }
    return kInvalidAction;
  }

  bool isLegal(ActionId action) const {
    if (isTerminal() || behavior.noLegalActions || action < 0 ||
        action >= actionCount()) {
      return false;
    }
    return normalMatchAction(action) == requiredAction(step);
  }

  void applyAction(ActionId action) {
    // Deliberately do not throw for an illegal action. IGameState says callers
    // may only apply legal actions; accepting a bad call here makes the negative
    // tests verify MatchRunner's validation rather than this fixture's checks.
    (void)action;
    if (!isTerminal()) {
      ++step;
    }
  }

  StateKey stateKey() const {
    return std::string(adapterB ? "toy-b:" : "toy-a:") +
           std::to_string(step) + ":" + std::to_string(currentPlayer());
  }

  StateKey canonicalStateKey() const {
    std::string key = "toy-canonical:" + std::to_string(step);
    if (behavior.divergeCanonicalInitially && step == spec->initialStep) {
      key += ":bad-initial";
    }
    if (behavior.divergeCanonicalAfterMove && step > spec->initialStep) {
      key += ":bad-after-move";
    }
    return key;
  }

  std::string rulesKey() const {
    return behavior.overrideRules ? behavior.rules : spec->rules;
  }

  MatchAction toMatchAction(ActionId action) const {
    MatchAction logical = normalMatchAction(action);
    if (behavior.mappingFault == MappingFault::kRoundTripMismatch &&
        logical == "alpha") {
      return "alpha-with-broken-round-trip";
    }
    return logical;
  }

  ActionId fromMatchAction(const MatchAction& action) const {
    if (action == "alpha") {
      if (behavior.mappingFault == MappingFault::kUnmappable) {
        return kInvalidAction;
      }
      if (behavior.mappingFault == MappingFault::kMappedToIllegalAction) {
        const ActionId correct = normalLocalAction(action);
        return static_cast<ActionId>((correct + 1) % actionCount());
      }
    }
    return normalLocalAction(action);
  }

  double terminalUtility() const {
    return behavior.overrideTerminalUtility ? behavior.terminalUtility
                                            : spec->terminalUtility;
  }

  std::string terminalReason() const {
    return behavior.overrideTerminalReason ? behavior.terminalReason
                                           : spec->terminalReason;
  }
};

class PlainToyState final : public IGameState {
 public:
  explicit PlainToyState(ToyPosition position)
      : position_(std::move(position)) {}

  std::unique_ptr<IGameState> clone() const override {
    return std::make_unique<PlainToyState>(*this);
  }
  int actionCount() const override { return position_.actionCount(); }
  bool isLegal(ActionId action) const override {
    return position_.isLegal(action);
  }
  void applyAction(ActionId action) override { position_.applyAction(action); }
  PlayerId currentPlayer() const override { return position_.currentPlayer(); }
  bool isTerminal() const override { return position_.isTerminal(); }
  double terminalUtility() const override {
    return position_.terminalUtility();
  }
  StateKey stateKey() const override { return position_.stateKey(); }
  int moveCount() const override { return position_.step; }

 private:
  ToyPosition position_;
};

class ToyMatchGameBase : public IMatchGame {
 public:
  int actionCount() const override { return position_.actionCount(); }
  bool isLegal(ActionId action) const override {
    return position_.isLegal(action);
  }
  void applyAction(ActionId action) override { position_.applyAction(action); }
  PlayerId currentPlayer() const override { return position_.currentPlayer(); }
  bool isTerminal() const override { return position_.isTerminal(); }
  double terminalUtility() const override {
    return position_.terminalUtility();
  }
  StateKey stateKey() const override { return position_.stateKey(); }
  int moveCount() const override { return position_.step; }

  std::string rulesKey() const override { return position_.rulesKey(); }
  StateKey canonicalStateKey() const override {
    return position_.canonicalStateKey();
  }
  MatchAction toMatchAction(ActionId action) const override {
    return position_.toMatchAction(action);
  }
  ActionId fromMatchAction(const MatchAction& action) const override {
    return position_.fromMatchAction(action);
  }
  std::string terminalReason() const override {
    return position_.terminalReason();
  }

  int step() const { return position_.step; }

 protected:
  ToyMatchGameBase(std::shared_ptr<const ToySpec> spec,
                   AdapterBehavior behavior, bool adapter_b)
      : position_{spec, std::move(behavior), spec->initialStep, adapter_b} {}

  const ToyPosition& position() const { return position_; }

 private:
  ToyPosition position_;
};

class ToyMatchGameA final : public ToyMatchGameBase {
 public:
  ToyMatchGameA(std::shared_ptr<const ToySpec> spec,
                AdapterBehavior behavior = {})
      : ToyMatchGameBase(std::move(spec), std::move(behavior), false) {}

  std::unique_ptr<IGameState> clone() const override {
    if (position().behavior.cloneDropsMatchInterface) {
      return std::make_unique<PlainToyState>(position());
    }
    return std::make_unique<ToyMatchGameA>(*this);
  }
};

class ToyMatchGameB final : public ToyMatchGameBase {
 public:
  ToyMatchGameB(std::shared_ptr<const ToySpec> spec,
                AdapterBehavior behavior = {})
      : ToyMatchGameBase(std::move(spec), std::move(behavior), true) {}

  std::unique_ptr<IGameState> clone() const override {
    if (position().behavior.cloneDropsMatchInterface) {
      return std::make_unique<PlainToyState>(position());
    }
    return std::make_unique<ToyMatchGameB>(*this);
  }
};

static_assert(!std::is_same<ToyMatchGameA, ToyMatchGameB>::value,
              "the match fixture must use genuinely distinct adapter types");

struct EvaluatorLog {
  std::mutex mutex;
  int calls = 0;
  bool sawWrongConcreteType = false;
  std::vector<int> steps;
  std::vector<PlayerId> players;
};

template <typename ExpectedGame>
class RecordingEvaluator final : public IEvaluator {
 public:
  explicit RecordingEvaluator(std::shared_ptr<EvaluatorLog> log)
      : log_(std::move(log)) {}

  Evaluation evaluate(const IGameState& state) override {
    std::lock_guard<std::mutex> lock(log_->mutex);
    ++log_->calls;
    const auto* game = dynamic_cast<const ExpectedGame*>(&state);
    if (game == nullptr) {
      log_->sawWrongConcreteType = true;
      throw std::runtime_error("evaluator received the other adapter type");
    }

    log_->steps.push_back(game->step());
    log_->players.push_back(game->currentPlayer());

    Evaluation evaluation;
    evaluation.policy.assign(static_cast<std::size_t>(game->actionCount()),
                             -1.0f);
    for (int action = 0; action < game->actionCount(); ++action) {
      const ActionId local_action = static_cast<ActionId>(action);
      if (game->isLegal(local_action)) {
        evaluation.policy[static_cast<std::size_t>(action)] = 1.0f;
      }
    }
    evaluation.utility = 0.0;
    return evaluation;
  }

 private:
  std::shared_ptr<EvaluatorLog> log_;
};

struct FactoryLog {
  std::atomic<int> adapterACalls{0};
  std::atomic<int> adapterBCalls{0};
};

struct MatchFixture {
  std::shared_ptr<const ToySpec> spec;
  std::shared_ptr<EvaluatorLog> evaluatorALog;
  std::shared_ptr<EvaluatorLog> evaluatorBLog;
  std::shared_ptr<FactoryLog> factoryLog;
  std::shared_ptr<RecordingEvaluator<ToyMatchGameA>> evaluatorA;
  std::shared_ptr<RecordingEvaluator<ToyMatchGameB>> evaluatorB;
  std::array<MatchParticipant, 2> participants;
};

MatchFixture makeFixture(
    ToySpec spec = {}, AdapterBehavior adapter_a_behavior = {},
    AdapterBehavior adapter_b_behavior = {},
    SearchParams adapter_a_params = rootEvaluationOnlyParams(),
    SearchParams adapter_b_params = twoPlayoutParams()) {
  MatchFixture fixture;
  fixture.spec = std::make_shared<const ToySpec>(std::move(spec));
  fixture.evaluatorALog = std::make_shared<EvaluatorLog>();
  fixture.evaluatorBLog = std::make_shared<EvaluatorLog>();
  fixture.factoryLog = std::make_shared<FactoryLog>();
  fixture.evaluatorA =
      std::make_shared<RecordingEvaluator<ToyMatchGameA>>(
          fixture.evaluatorALog);
  fixture.evaluatorB =
      std::make_shared<RecordingEvaluator<ToyMatchGameB>>(
          fixture.evaluatorBLog);

  MatchParticipant adapter_a;
  adapter_a.name = "adapter-a";
  adapter_a.player = fixture.spec->firstPlayer;
  adapter_a.stateFactory =
      [spec_ptr = fixture.spec, adapter_a_behavior,
       factory_log = fixture.factoryLog]() {
        ++factory_log->adapterACalls;
        return std::make_shared<ToyMatchGameA>(spec_ptr,
                                               adapter_a_behavior);
      };
  adapter_a.evaluator = fixture.evaluatorA;
  adapter_a.searchParams = adapter_a_params;

  MatchParticipant adapter_b;
  adapter_b.name = "adapter-b";
  adapter_b.player = fixture.spec->secondPlayer;
  adapter_b.stateFactory =
      [spec_ptr = fixture.spec, adapter_b_behavior,
       factory_log = fixture.factoryLog]() {
        ++factory_log->adapterBCalls;
        return std::make_shared<ToyMatchGameB>(spec_ptr,
                                               adapter_b_behavior);
      };
  adapter_b.evaluator = fixture.evaluatorB;
  adapter_b.searchParams = adapter_b_params;

  fixture.participants = {{std::move(adapter_a), std::move(adapter_b)}};
  return fixture;
}

MatchConfig matchConfig(int max_moves) {
  MatchConfig config;
  config.maxMoves = max_moves;
  config.seed = 0x123456789abcdef0ULL;
  return config;
}

MatchResult runFixture(const MatchFixture& fixture, int max_moves) {
  MatchRunner runner(matchConfig(max_moves), fixture.participants);
  return runner.run();
}

void expectRunFailure(MatchFixture fixture, MatchConfig config,
                      const char* case_name, int line) {
  ++g_checks;
  if (!throwsAnything([&]() {
        MatchRunner runner(config, fixture.participants);
        (void)runner.run();
      })) {
    std::cerr << "FAIL " << __FILE__ << ":" << line
              << ": expected MatchRunner failure for " << case_name << "\n";
    ++g_failures;
  }
}

void expectRunFailure(MatchFixture fixture, const char* case_name, int line) {
  expectRunFailure(std::move(fixture), matchConfig(8), case_name, line);
}

void checkOptionalNear(const std::optional<double>& actual, double expected) {
  CHECK(actual.has_value());
  if (actual.has_value()) {
    CHECK_NEAR(*actual, expected, 1e-12);
  }
}

void checkMove(const MatchResult& result, std::size_t index, PlayerId player,
               int participant_index, const MatchAction& action,
               ActionId adapter_a_action, ActionId adapter_b_action) {
  CHECK(index < result.moves.size());
  if (index >= result.moves.size()) {
    return;
  }
  const MatchMove& move = result.moves[index];
  CHECK(move.player == player);
  CHECK(move.participantIndex == participant_index);
  CHECK(move.action == action);
  CHECK(move.localActions[0] == adapter_a_action);
  CHECK(move.localActions[1] == adapter_b_action);
}

void checkScore(const MatchScore& score, int wins, int losses, int draws,
                int truncated) {
  CHECK(score.wins == wins);
  CHECK(score.losses == losses);
  CHECK(score.draws == draws);
  CHECK(score.truncated == truncated);
}

void testHeterogeneousMappingSynchronizationAndSearch() {
  MatchFixture fixture = makeFixture();
  CHECK(fixture.participants[0].evaluator.get() !=
        fixture.participants[1].evaluator.get());
  CHECK(fixture.participants[0].searchParams.maxVisits == 1);
  CHECK(fixture.participants[0].searchParams.maxPlayouts == 0);
  CHECK(fixture.participants[1].searchParams.maxVisits == 0);
  CHECK(fixture.participants[1].searchParams.maxPlayouts == 2);

  // Three is both the exact terminal length and the move cap. Terminality must
  // win over truncation on the last allowed action.
  const MatchResult result = runFixture(fixture, 3);
  CHECK(result.terminal);
  CHECK(!result.truncated);
  CHECK(result.moves.size() == 3);
  CHECK(result.finalPlayer == fixture.spec->terminalPlayer);
  checkOptionalNear(result.finalUtility, 0.75);
  checkOptionalNear(result.participantUtilities[0], 0.75);
  checkOptionalNear(result.participantUtilities[1], -0.75);
  CHECK(result.terminalReason == fixture.spec->terminalReason);

  // A and B expose alpha/beta/gamma at [0,1,2] and [1,2,0].
  checkMove(result, 0, fixture.spec->firstPlayer, 0, "alpha", 0, 1);
  checkMove(result, 1, fixture.spec->firstPlayer, 0, "beta", 1, 2);
  checkMove(result, 2, fixture.spec->secondPlayer, 1, "gamma", 2, 0);

  CHECK(result.moves[0].player == result.moves[1].player);
  CHECK(result.moves[0].participantIndex ==
        result.moves[1].participantIndex);
  CHECK(fixture.factoryLog->adapterACalls == 1);
  CHECK(fixture.factoryLog->adapterBCalls == 1);

  const std::vector<int> expected_a_steps{0, 1};
  const std::vector<int> expected_b_steps{2};
  CHECK(fixture.evaluatorALog->steps == expected_a_steps);
  CHECK(fixture.evaluatorBLog->steps == expected_b_steps);
  CHECK(fixture.evaluatorALog->calls > 0);
  CHECK(fixture.evaluatorBLog->calls > 0);
  CHECK(!fixture.evaluatorALog->sawWrongConcreteType);
  CHECK(!fixture.evaluatorBLog->sawWrongConcreteType);
}

void testTerminalPerspectiveWithNonBinaryPlayerIds() {
  ToySpec spec;
  spec.firstPlayer = 37;
  spec.secondPlayer = -12;
  spec.terminalPlayer = spec.secondPlayer;
  spec.terminalUtility = -0.625;

  MatchFixture fixture = makeFixture(spec);
  const MatchResult result = runFixture(fixture, 10);

  CHECK(result.terminal);
  CHECK(!result.truncated);
  CHECK(result.finalPlayer == -12);
  checkOptionalNear(result.finalUtility, -0.625);
  // terminalUtility is from finalPlayer's perspective, not the last mover's or
  // a hard-coded player 0/1 perspective.
  checkOptionalNear(result.participantUtilities[0], 0.625);
  checkOptionalNear(result.participantUtilities[1], -0.625);
}

void testInitiallyTerminalGameSkipsSearch() {
  ToySpec spec;
  spec.initialStep = spec.terminalStep;
  spec.terminalPlayer = spec.secondPlayer;
  spec.terminalUtility = -0.4;

  MatchFixture fixture = makeFixture(spec);
  const MatchResult result = runFixture(fixture, 1);

  CHECK(result.terminal);
  CHECK(!result.truncated);
  CHECK(result.moves.empty());
  CHECK(result.finalPlayer == spec.secondPlayer);
  checkOptionalNear(result.finalUtility, -0.4);
  checkOptionalNear(result.participantUtilities[0], 0.4);
  checkOptionalNear(result.participantUtilities[1], -0.4);
  CHECK(result.terminalReason == spec.terminalReason);
  CHECK(fixture.evaluatorALog->calls == 0);
  CHECK(fixture.evaluatorBLog->calls == 0);
}

void testMoveCapProducesUtilityFreeTruncation() {
  MatchFixture fixture = makeFixture();
  const MatchResult result = runFixture(fixture, 1);

  CHECK(!result.terminal);
  CHECK(result.truncated);
  CHECK(result.moves.size() == 1);
  CHECK(result.finalPlayer == fixture.spec->firstPlayer);
  CHECK(!result.finalUtility.has_value());
  CHECK(!result.participantUtilities[0].has_value());
  CHECK(!result.participantUtilities[1].has_value());
}

void testRulesAndCanonicalStateValidation() {
  AdapterBehavior incompatible_rules;
  incompatible_rules.overrideRules = true;
  incompatible_rules.rules = "different-rules";
  expectRunFailure(makeFixture({}, {}, incompatible_rules),
                   "incompatible rulesKey", __LINE__);

  AdapterBehavior bad_initial_key;
  bad_initial_key.divergeCanonicalInitially = true;
  expectRunFailure(makeFixture({}, {}, bad_initial_key),
                   "initial canonical state divergence", __LINE__);

  AdapterBehavior bad_advanced_key;
  bad_advanced_key.divergeCanonicalAfterMove = true;
  expectRunFailure(makeFixture({}, {}, bad_advanced_key),
                   "post-move canonical state divergence", __LINE__);
}

void testActionMappingValidation() {
  struct FaultCase {
    MappingFault fault;
    const char* name;
  };
  const std::array<FaultCase, 3> cases{{
      {MappingFault::kUnmappable, "unmappable logical action"},
      {MappingFault::kMappedToIllegalAction, "mapped local action is illegal"},
      {MappingFault::kRoundTripMismatch, "logical/local round trip mismatch"},
  }};

  // Put faults on the non-acting adapter at step zero. Its applyAction accepts
  // bad calls, so these only fail when MatchRunner validates both mappings.
  for (const FaultCase& fault_case : cases) {
    AdapterBehavior bad_mapping;
    bad_mapping.mappingFault = fault_case.fault;
    expectRunFailure(makeFixture({}, {}, bad_mapping), fault_case.name,
                     __LINE__);
  }
}

void testTerminalMetadataValidation() {
  ToySpec terminal_spec;
  terminal_spec.initialStep = terminal_spec.terminalStep;

  AdapterBehavior mismatched_utility;
  mismatched_utility.overrideTerminalUtility = true;
  mismatched_utility.terminalUtility = -0.25;
  expectRunFailure(makeFixture(terminal_spec, {}, mismatched_utility),
                   "mismatched terminal utility", __LINE__);

  for (double non_finite :
       {std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()}) {
    AdapterBehavior bad_a;
    bad_a.overrideTerminalUtility = true;
    bad_a.terminalUtility = non_finite;
    AdapterBehavior bad_b = bad_a;
    expectRunFailure(makeFixture(terminal_spec, bad_a, bad_b),
                     "non-finite terminal utility", __LINE__);
  }

  AdapterBehavior mismatched_reason;
  mismatched_reason.overrideTerminalReason = true;
  mismatched_reason.terminalReason = "different-terminal-reason";
  expectRunFailure(makeFixture(terminal_spec, {}, mismatched_reason),
                   "mismatched nonempty terminal reason", __LINE__);
}

void testUnknownPlayerAndNoLegalActionValidation() {
  AdapterBehavior unknown;
  unknown.unknownPlayer = true;
  MatchFixture unknown_fixture = makeFixture({}, unknown, unknown);
  expectRunFailure(unknown_fixture, "unknown currentPlayer", __LINE__);
  CHECK(unknown_fixture.evaluatorALog->calls == 0);
  CHECK(unknown_fixture.evaluatorBLog->calls == 0);

  AdapterBehavior stuck;
  stuck.noLegalActions = true;
  expectRunFailure(makeFixture({}, stuck, stuck),
                   "nonterminal state with no legal action", __LINE__);
}

void testConfigurationAndNullDependencyValidation() {
  for (int max_moves : {-1, -3}) {
    expectRunFailure(makeFixture(), matchConfig(max_moves),
                     "negative maxMoves", __LINE__);
  }
  const MatchResult unlimited = runFixture(makeFixture(), 0);
  CHECK(unlimited.terminal);
  CHECK(!unlimited.truncated);
  CHECK(unlimited.moves.size() == 3);

  for (int index = 0; index < 2; ++index) {
    MatchFixture missing_factory = makeFixture();
    missing_factory.participants[static_cast<std::size_t>(index)].stateFactory =
        {};
    expectRunFailure(std::move(missing_factory), "empty state factory",
                     __LINE__);

    MatchFixture null_state = makeFixture();
    null_state.participants[static_cast<std::size_t>(index)].stateFactory =
        []() -> std::shared_ptr<IMatchGame> { return nullptr; };
    expectRunFailure(std::move(null_state), "factory returned null state",
                     __LINE__);

    MatchFixture null_evaluator = makeFixture();
    null_evaluator.participants[static_cast<std::size_t>(index)].evaluator
        .reset();
    expectRunFailure(std::move(null_evaluator), "null evaluator", __LINE__);
  }
}

void testCloneMustPreserveMatchInterface() {
  for (int bad_index = 0; bad_index < 2; ++bad_index) {
    AdapterBehavior behavior_a;
    AdapterBehavior behavior_b;
    if (bad_index == 0) {
      behavior_a.cloneDropsMatchInterface = true;
    } else {
      behavior_b.cloneDropsMatchInterface = true;
    }

    MatchFixture fixture = makeFixture(
        {}, behavior_a, behavior_b, rootEvaluationOnlyParams(),
        rootEvaluationOnlyParams());
    const bool threw = throwsAnything([&]() {
      MatchRunner runner(matchConfig(8), fixture.participants);
      (void)runner.run();
    });
    CHECK(threw);
    // The runner should reject the malformed clone before handing it to the
    // adapter-specific evaluator on a later move.
    if (fixture.evaluatorALog->sawWrongConcreteType ||
        fixture.evaluatorBLog->sawWrongConcreteType) {
      std::cerr << "CONTEXT malformed clone from adapter "
                << (bad_index == 0 ? "A" : "B")
                << " reached evaluator (wrongA="
                << fixture.evaluatorALog->sawWrongConcreteType
                << ", wrongB="
                << fixture.evaluatorBLog->sawWrongConcreteType << ")\n";
    }
    CHECK(!fixture.evaluatorALog->sawWrongConcreteType);
    CHECK(!fixture.evaluatorBLog->sawWrongConcreteType);
  }
}

void testSeriesSwapsIdsButKeepsParticipantIndicesStable() {
  MatchFixture fixture = makeFixture();
  const PlayerId original_a_player = fixture.participants[0].player;
  const PlayerId original_b_player = fixture.participants[1].player;

  MatchSeriesConfig config;
  config.pairs = 1;
  config.game = matchConfig(3);
  const MatchSeriesResult series = runMatchSeries(config, fixture.participants);

  CHECK(series.games.size() == 2);
  CHECK(fixture.participants[0].player == original_a_player);
  CHECK(fixture.participants[1].player == original_b_player);
  CHECK(fixture.factoryLog->adapterACalls == 2);
  CHECK(fixture.factoryLog->adapterBCalls == 2);
  CHECK(fixture.evaluatorALog->calls > 0);
  CHECK(fixture.evaluatorBLog->calls > 0);
  CHECK(!fixture.evaluatorALog->sawWrongConcreteType);
  CHECK(!fixture.evaluatorBLog->sawWrongConcreteType);

  if (series.games.size() == 2) {
    const MatchResult& first = series.games[0];
    const MatchResult& swapped = series.games[1];
    CHECK(first.terminal);
    CHECK(swapped.terminal);

    checkMove(first, 0, fixture.spec->firstPlayer, 0, "alpha", 0, 1);
    checkMove(first, 1, fixture.spec->firstPlayer, 0, "beta", 1, 2);
    checkMove(first, 2, fixture.spec->secondPlayer, 1, "gamma", 2, 0);

    // The same game PlayerIds act in the same order, while the input model
    // indices are exchanged exactly. localActions remains in input order.
    checkMove(swapped, 0, fixture.spec->firstPlayer, 1, "alpha", 0, 1);
    checkMove(swapped, 1, fixture.spec->firstPlayer, 1, "beta", 1, 2);
    checkMove(swapped, 2, fixture.spec->secondPlayer, 0, "gamma", 2, 0);

    checkOptionalNear(first.participantUtilities[0], 0.75);
    checkOptionalNear(first.participantUtilities[1], -0.75);
    checkOptionalNear(swapped.participantUtilities[0], -0.75);
    checkOptionalNear(swapped.participantUtilities[1], 0.75);
  }

  checkScore(series.totals[0], 1, 1, 0, 0);
  checkScore(series.totals[1], 1, 1, 0, 0);
  checkScore(series.bySeat[0][0], 1, 0, 0, 0);
  checkScore(series.bySeat[0][1], 0, 1, 0, 0);
  checkScore(series.bySeat[1][0], 1, 0, 0, 0);
  checkScore(series.bySeat[1][1], 0, 1, 0, 0);
}

void testSeriesDrawAndTruncationAreScoredSeparately() {
  ToySpec draw_spec;
  draw_spec.terminalUtility = 0.0;
  MatchFixture draw_fixture = makeFixture(draw_spec);
  MatchSeriesConfig draw_config;
  draw_config.pairs = 1;
  draw_config.game = matchConfig(3);
  const MatchSeriesResult draws =
      runMatchSeries(draw_config, draw_fixture.participants);

  CHECK(draws.games.size() == 2);
  checkScore(draws.totals[0], 0, 0, 2, 0);
  checkScore(draws.totals[1], 0, 0, 2, 0);
  for (int participant = 0; participant < 2; ++participant) {
    for (int seat = 0; seat < 2; ++seat) {
      checkScore(draws.bySeat[static_cast<std::size_t>(participant)]
                              [static_cast<std::size_t>(seat)],
                 0, 0, 1, 0);
    }
  }

  MatchFixture truncated_fixture = makeFixture();
  MatchSeriesConfig truncated_config;
  truncated_config.pairs = 1;
  truncated_config.game = matchConfig(1);
  const MatchSeriesResult truncated =
      runMatchSeries(truncated_config, truncated_fixture.participants);

  CHECK(truncated.games.size() == 2);
  for (const MatchResult& game : truncated.games) {
    CHECK(!game.terminal);
    CHECK(game.truncated);
    CHECK(!game.finalUtility.has_value());
    CHECK(!game.participantUtilities[0].has_value());
    CHECK(!game.participantUtilities[1].has_value());
  }
  checkScore(truncated.totals[0], 0, 0, 0, 2);
  checkScore(truncated.totals[1], 0, 0, 0, 2);
  for (int participant = 0; participant < 2; ++participant) {
    for (int seat = 0; seat < 2; ++seat) {
      checkScore(truncated.bySeat[static_cast<std::size_t>(participant)]
                                  [static_cast<std::size_t>(seat)],
                 0, 0, 0, 1);
    }
  }
}

void testConcurrentSeriesPreservesOrderAndUsesGameThreads() {
  for (int workers : {1, 3, 20}) {
    MatchFixture fixture = makeFixture();
    MatchSeriesConfig config;
    config.pairs = 2;
    config.game = matchConfig(3);
    const auto sequential = runMatchSeries(config, fixture.participants);
    config.numGameThreads = workers;
    const int thread_count = std::min(workers, 4);
    const auto caller = std::this_thread::get_id();
    std::mutex mutex;
    std::condition_variable ready;
    int arrivals = 0;
    std::vector<std::thread::id> thread_ids;
    const auto factory = fixture.participants[0].stateFactory;
    fixture.participants[0].stateFactory = [&]() {
      {
        std::unique_lock<std::mutex> lock(mutex);
        ++arrivals;
        thread_ids.push_back(std::this_thread::get_id());
        ready.notify_all();
        if (!ready.wait_for(lock, std::chrono::seconds(5),
                            [&]() { return arrivals >= thread_count; })) {
          throw std::runtime_error("games did not start concurrently");
        }
      }
      return factory();
    };
    std::vector<int> callbacks;
    const auto series = runMatchSeries(
        config, fixture.participants, [&](int index, const MatchResult& game) {
          CHECK(std::this_thread::get_id() == caller);
          callbacks.push_back(index);
          CHECK(game.moves.front().participantIndex == index % 2);
        });
    std::sort(thread_ids.begin(), thread_ids.end());
    thread_ids.erase(std::unique(thread_ids.begin(), thread_ids.end()),
                     thread_ids.end());
    CHECK(thread_ids.size() == static_cast<std::size_t>(thread_count));
    std::sort(callbacks.begin(), callbacks.end());
    CHECK(callbacks == std::vector<int>({0, 1, 2, 3}));
    CHECK(series.games.size() == sequential.games.size());
    for (std::size_t i = 0; i < series.games.size(); ++i) {
      const auto& actual = series.games[i];
      const auto& expected = sequential.games[i];
      CHECK(actual.terminal == expected.terminal);
      CHECK(actual.truncated == expected.truncated);
      CHECK(actual.finalPlayer == expected.finalPlayer);
      CHECK(actual.finalUtility == expected.finalUtility);
      CHECK(actual.participantUtilities == expected.participantUtilities);
      CHECK(actual.moves.size() == expected.moves.size());
      for (std::size_t j = 0; j < actual.moves.size(); ++j) {
        CHECK(actual.moves[j].action == expected.moves[j].action);
        CHECK(actual.moves[j].participantIndex == expected.moves[j].participantIndex);
        CHECK(actual.moves[j].localActions == expected.moves[j].localActions);
      }
    }
    for (int i = 0; i < 2; ++i) {
      checkScore(series.totals[static_cast<std::size_t>(i)], 2, 2, 0, 0);
      checkScore(series.bySeat[static_cast<std::size_t>(i)][0], 2, 0, 0, 0);
      checkScore(series.bySeat[static_cast<std::size_t>(i)][1], 0, 2, 0, 0);
    }
  }
}

void testConcurrentSeriesPropagatesErrors() {
  for (bool callback_error : {false, true}) {
    MatchFixture fixture = makeFixture();
    MatchSeriesConfig config;
    config.pairs = 4;
    config.numGameThreads = 3;
    config.game = matchConfig(3);
    if (!callback_error) {
      fixture.participants[0].stateFactory = []() -> std::shared_ptr<IMatchGame> {
        throw std::runtime_error("worker failure");
      };
    }
    int callbacks = 0;
    bool caught = false;
    try {
      (void)runMatchSeries(config, fixture.participants,
                          [&](int, const MatchResult&) {
                            ++callbacks;
                            throw std::runtime_error("callback failure");
                          });
    } catch (const std::runtime_error& error) {
      caught = true;
      CHECK(std::string(error.what()) ==
            (callback_error ? "callback failure" : "worker failure"));
    }
    CHECK(caught);
    CHECK(callbacks == (callback_error ? 1 : 0));
  }
  for (int workers : {0, -1}) {
    MatchFixture fixture = makeFixture();
    MatchSeriesConfig config;
    config.numGameThreads = workers;
    CHECK(throwsAnything(
        [&]() { (void)runMatchSeries(config, fixture.participants); }));
    CHECK(fixture.factoryLog->adapterACalls == 0);
    CHECK(fixture.evaluatorALog->calls == 0);
  }
}

void testSeriesRejectsInvalidPairCounts() {
  for (int pairs : {0, -1}) {
    MatchFixture fixture = makeFixture();
    MatchSeriesConfig config;
    config.pairs = pairs;
    config.game = matchConfig(3);
    CHECK(throwsAnything(
        [&]() { (void)runMatchSeries(config, fixture.participants); }));
  }
}

}  // namespace

int main() {
  testHeterogeneousMappingSynchronizationAndSearch();
  testTerminalPerspectiveWithNonBinaryPlayerIds();
  testInitiallyTerminalGameSkipsSearch();
  testMoveCapProducesUtilityFreeTruncation();
  testRulesAndCanonicalStateValidation();
  testActionMappingValidation();
  testTerminalMetadataValidation();
  testUnknownPlayerAndNoLegalActionValidation();
  testConfigurationAndNullDependencyValidation();
  testCloneMustPreserveMatchInterface();
  testSeriesSwapsIdsButKeepsParticipantIndicesStable();
  testSeriesDrawAndTruncationAreScoredSeparately();
  testConcurrentSeriesPreservesOrderAndUsesGameThreads();
  testConcurrentSeriesPropagatesErrors();
  testSeriesRejectsInvalidPairCounts();

  if (g_failures == 0) {
    std::cout << "All " << g_checks << " checks passed.\n";
    return 0;
  }
  std::cerr << g_failures << " of " << g_checks << " checks failed.\n";
  return 1;
}
