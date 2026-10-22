// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-02 contributors
//
// Unit tests for the game-agnostic PUCT search. Toy games only; no engine is
// linked or modified.

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "suzero/mcts/puct_search.h"
#include "toy_games.h"

namespace {

using suzero::mcts::ActionStats;
using suzero::mcts::PuctSearch;
using suzero::mcts::SearchParams;
using suzero::mcts::SearchProgress;
using suzero::mcts::SearchResult;
using namespace suzero::mcts::test;

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
    const double actual_value = (actual);                                    \
    const double expected_value = (expected);                                \
    if (!(std::fabs(actual_value - expected_value) <= (tolerance))) {        \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": "            \
                << #actual << " = " << actual_value << ", expected "         \
                << expected_value << " +/- " << (tolerance) << "\n";         \
      ++g_failures;                                                          \
    }                                                                        \
  } while (0)

SearchParams paramsWithPlayouts(std::int64_t playouts) {
  SearchParams params;
  params.maxPlayouts = playouts;
  params.maxVisits = 0;
  params.maxTimeSeconds = 0.0;
  return params;
}

SearchResult runSearch(PuctSearch& search) { return search.run(); }

double shareOf(const SearchResult& result, int action) {
  for (const ActionStats& stats : result.actions) {
    if (stats.action == action) {
      return stats.visitShare;
    }
  }
  return 0.0;
}

void testFindsForcedWin() {
  auto state = std::make_shared<SubtractionGame>(0, 0);
  auto evaluator = std::make_shared<UniformEvaluator>();
  PuctSearch search(state, evaluator, paramsWithPlayouts(4000));
  const SearchResult result = runSearch(search);

  CHECK(result.actions.size() == 2);
  CHECK(result.bestAction == 1);
  CHECK(result.rootUtility > 0.9);
  CHECK(result.rootVisits == result.playouts + 1);
  CHECK(shareOf(result, 1) > 0.9);
  CHECK(search.chooseAction(0.0) == 1);

  double share_sum = 0.0;
  for (const ActionStats& stats : result.actions) {
    share_sum += stats.visitShare;
  }
  CHECK_NEAR(share_sum, 1.0, 1e-9);

  const std::vector<double> dense = result.denseVisitDistribution(2);
  CHECK(dense.size() == 2);
  CHECK(dense[1] > 0.9);
  CHECK(dense[0] < 0.1);
  CHECK_NEAR(dense[0] + dense[1], 1.0, 1e-9);
}

void testTerminalRoot() {
  // The root is already won for player 0.
  auto state = std::make_shared<OneMoveWinGame>(true, 0, 0);
  auto evaluator = std::make_shared<UniformEvaluator>();
  PuctSearch search(state, evaluator, paramsWithPlayouts(100));
  const SearchResult result = runSearch(search);

  CHECK(result.playouts == 0);
  CHECK(result.actions.empty());
  CHECK(result.bestAction == suzero::mcts::kInvalidAction);
  CHECK_NEAR(result.rootUtility, 1.0, 1e-12);
  CHECK(result.rootVisits == 1);
  CHECK(search.chooseAction(0.0) == suzero::mcts::kInvalidAction);
}

void testExploresDespiteSkewedPrior() {
  auto state = std::make_shared<OneMoveWinGame>(false, 0, 0);
  auto evaluator = std::make_shared<SkewedWinEvaluator>();
  PuctSearch search(state, evaluator, paramsWithPlayouts(1000));
  const SearchResult result = runSearch(search);

  // Action 2 has the smallest prior but is the only winning move.
  CHECK(result.bestAction == 2);
  CHECK(result.rootUtility > 0.9);
  CHECK(shareOf(result, 2) > 0.8);
}

void testNonAlternatingPerspective() {
  auto state = std::make_shared<SamePlayerGame>();
  auto evaluator = std::make_shared<UniformEvaluator>();
  PuctSearch search(state, evaluator, paramsWithPlayouts(4000));
  SearchResult result = runSearch(search);

  // Only the line 0 -> 1 wins. If the engine negated values on every level,
  // the terminal +1 would be treated as -1 and this would fail.
  CHECK(result.bestAction == 0);
  CHECK(result.rootUtility > 0.9);
  CHECK(search.chooseAction(0.0) == 0);

  CHECK(search.advanceRoot(0));
  CHECK(search.rootState().currentPlayer() == 0);
  const std::int64_t visits_before = search.rootNode().stats().visits();
  result = runSearch(search);
  CHECK(result.bestAction == 1);
  CHECK(result.rootUtility > 0.9);
  CHECK(search.chooseAction(0.0) == 1);
  CHECK(search.rootNode().stats().visits() >= visits_before);
}

void testActionMasking() {
  // At position 4 only +1 lands exactly on 5.
  auto state = std::make_shared<SubtractionGame>(4, 0);
  auto evaluator = std::make_shared<UniformEvaluator>();
  PuctSearch search(state, evaluator, paramsWithPlayouts(200));
  const SearchResult result = runSearch(search);

  CHECK(result.actions.size() == 1);
  CHECK(result.actions[0].action == 0);
  CHECK(result.bestAction == 0);
  CHECK(result.rootUtility > 0.99);
  for (const ActionStats& stats : result.actions) {
    CHECK(stats.action == 0 || stats.action == 1);
  }
}

void testDeterministicWithNoise() {
  SearchParams params = paramsWithPlayouts(500);
  params.rootNoiseEnabled = true;
  params.rootDirichletNoiseWeight = 0.25;

  auto state1 = std::make_shared<SubtractionGame>(0, 0);
  auto state2 = std::make_shared<SubtractionGame>(0, 0);
  auto evaluator = std::make_shared<UniformEvaluator>();

  PuctSearch first(state1, evaluator, params, 12345);
  PuctSearch second(state2, evaluator, params, 12345);
  const SearchResult a = runSearch(first);
  const SearchResult b = runSearch(second);

  CHECK(a.actions.size() == b.actions.size());
  for (std::size_t i = 0; i < a.actions.size() && i < b.actions.size(); ++i) {
    CHECK(a.actions[i].action == b.actions[i].action);
    CHECK(a.actions[i].visits == b.actions[i].visits);
    CHECK_NEAR(a.actions[i].prior, b.actions[i].prior, 0.0);
  }
  CHECK(a.bestAction == b.bestAction);
  CHECK_NEAR(a.rootUtility, b.rootUtility, 0.0);
}

void testStopCriterion() {
  SearchParams params;
  params.maxPlayouts = 0;
  params.maxVisits = 0;
  params.maxTimeSeconds = 0.0;

  auto state = std::make_shared<SubtractionGame>(0, 0);
  auto evaluator = std::make_shared<UniformEvaluator>();
  PuctSearch search(state, evaluator, params);

  const SearchResult result =
      search.run([](const SearchProgress& progress) {
        return progress.playouts >= 10;
      });
  CHECK(!result.completed);
  CHECK(result.playouts == 10);
}

void testMaxVisits() {
  SearchParams params;
  params.maxVisits = 100;
  params.maxPlayouts = 0;

  auto state = std::make_shared<SubtractionGame>(0, 0);
  auto evaluator = std::make_shared<UniformEvaluator>();
  PuctSearch search(state, evaluator, params);
  const SearchResult result = runSearch(search);

  CHECK(result.completed);
  CHECK(result.rootVisits == 100);
  CHECK(result.playouts == 99);
}

void testDirectPolicyFallback() {
  SearchParams params;
  params.maxVisits = 1;  // Stop after the root evaluation.
  params.maxPlayouts = 0;

  auto state = std::make_shared<OneMoveWinGame>(false, 0, 0);
  auto evaluator = std::make_shared<SkewedWinEvaluator>();
  PuctSearch search(state, evaluator, params);
  const SearchResult result = runSearch(search);

  CHECK(result.playouts == 0);
  CHECK(result.bestAction == 0);  // Highest prior before any search.
  CHECK(search.chooseAction(0.0) == 0);
}

void testTemperatureSamplingIsLegal() {
  auto state = std::make_shared<OneMoveWinGame>(false, 0, 0);
  auto evaluator = std::make_shared<SkewedWinEvaluator>();
  PuctSearch search(state, evaluator, paramsWithPlayouts(300));
  runSearch(search);

  CHECK(search.chooseAction(0.0) == 2);
  for (int i = 0; i < 100; ++i) {
    const auto action = search.chooseAction(1.0);
    CHECK(action >= 0 && action < 3);
  }
}

void testEvaluatorValidation() {
  auto state = std::make_shared<SubtractionGame>(0, 0);
  auto evaluator = std::make_shared<BrokenEvaluator>();
  PuctSearch search(state, evaluator, paramsWithPlayouts(10));

  bool threw = false;
  try {
    runSearch(search);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw);
}

void testTranspositionMerging() {
  // Playing 0 then 1 and 1 then 0 reaches the same state. Graph search must
  // allocate a single node for it instead of one per path.
  auto state = std::make_shared<CommutingGame>();
  auto evaluator = std::make_shared<UniformEvaluator>();
  PuctSearch search(state, evaluator, paramsWithPlayouts(800));
  const SearchResult result = runSearch(search);

  // root + two depth-1 states + one shared depth-2 terminal state.
  CHECK(search.nodeCount() == 4);
  CHECK(result.completed);

  const suzero::mcts::SearchNode* first = search.rootNode().childForAction(0);
  const suzero::mcts::SearchNode* second = search.rootNode().childForAction(1);
  CHECK(first != nullptr);
  CHECK(second != nullptr);
  if (first != nullptr && second != nullptr) {
    CHECK(first != second);
    CHECK(first->childForAction(1) != nullptr);
    CHECK(first->childForAction(1) == second->childForAction(0));
    CHECK(first->childForAction(1)->isTerminal());
  }
}

void testCycleTerminates() {
  // Action 0 loops between two positions. The graph keeps the cycle-closing
  // edge, so termination relies on the per-playout path check.
  auto state = std::make_shared<CycleGame>();
  auto evaluator = std::make_shared<UniformEvaluator>();
  PuctSearch search(state, evaluator, paramsWithPlayouts(300));
  const SearchResult result = runSearch(search);

  CHECK(result.completed);
  CHECK(result.playouts == 300);
  CHECK(result.rootVisits == result.playouts + 1);
  // root + two phase nodes + one duplicate of the root phase + two terminals
  CHECK(search.nodeCount() <= 5);

  // Promote across a cycle edge: the copied subtree keeps the cycle, and the
  // new root must remain unreachable from it.
  const suzero::mcts::SearchNode* promoted =
      search.rootNode().childForAction(0);
  CHECK(promoted != nullptr);
  if (promoted != nullptr) {
    const std::int64_t child_visits = promoted->stats().visits();
    CHECK(search.advanceRoot(0));
    CHECK(search.rootNode().stats().visits() == child_visits);
    const SearchResult reused = runSearch(search);
    CHECK(reused.completed);
    CHECK(reused.playouts == 300);
    CHECK(reused.rootVisits == child_visits + reused.playouts);
  }
}

void testCycleCheckToggleOnAcyclicGame() {
  // On a DAG the path set only costs time, so both settings must agree.
  auto make_params = [](bool enabled) {
    SearchParams params = paramsWithPlayouts(500);
    params.graphSearchCycleCheckEnabled = enabled;
    return params;
  };

  PuctSearch with_check(std::make_shared<CommutingGame>(),
                        std::make_shared<UniformEvaluator>(),
                        make_params(true), 4242);
  PuctSearch without_check(std::make_shared<CommutingGame>(),
                           std::make_shared<UniformEvaluator>(),
                           make_params(false), 4242);

  const SearchResult a = runSearch(with_check);
  const SearchResult b = runSearch(without_check);

  CHECK(a.bestAction == b.bestAction);
  CHECK(a.actions.size() == b.actions.size());
  for (std::size_t i = 0; i < a.actions.size() && i < b.actions.size(); ++i) {
    CHECK(a.actions[i].action == b.actions[i].action);
    CHECK(a.actions[i].visits == b.actions[i].visits);
    CHECK_NEAR(a.actions[i].prior, b.actions[i].prior, 0.0);
  }
  CHECK_NEAR(a.rootUtility, b.rootUtility, 0.0);
  CHECK(with_check.nodeCount() == without_check.nodeCount());
}

void testCycleCheckDisabledWithDepthLimit() {
  // Without the path check a cyclic game relies on maxDepth to terminate.
  SearchParams params = paramsWithPlayouts(50);
  params.graphSearchCycleCheckEnabled = false;
  params.maxDepth = 8;

  PuctSearch search(std::make_shared<CycleGame>(),
                    std::make_shared<UniformEvaluator>(), params);
  const SearchResult result = runSearch(search);

  CHECK(result.completed);
  CHECK(result.playouts == 50);
}

void testAdvanceRootInvalid() {
  auto state = std::make_shared<SubtractionGame>(0, 0);
  auto evaluator = std::make_shared<UniformEvaluator>();
  PuctSearch search(state, evaluator, paramsWithPlayouts(10));

  CHECK(!search.advanceRoot(999));
  runSearch(search);
  CHECK(!search.advanceRoot(999));
}

void testResetRoot() {
  auto state = std::make_shared<SubtractionGame>(0, 0);
  auto evaluator = std::make_shared<UniformEvaluator>();
  PuctSearch search(state, evaluator, paramsWithPlayouts(200));
  runSearch(search);

  auto new_state = std::make_shared<SubtractionGame>(4, 0);
  search.resetRoot(new_state);
  const SearchResult result = runSearch(search);
  CHECK(result.actions.size() == 1);
  CHECK(result.actions[0].action == 0);
  CHECK(result.rootUtility > 0.99);
}

}  // namespace

int main() {
  testFindsForcedWin();
  testTerminalRoot();
  testExploresDespiteSkewedPrior();
  testNonAlternatingPerspective();
  testActionMasking();
  testDeterministicWithNoise();
  testStopCriterion();
  testMaxVisits();
  testDirectPolicyFallback();
  testTemperatureSamplingIsLegal();
  testEvaluatorValidation();
  testTranspositionMerging();
  testCycleTerminates();
  testCycleCheckToggleOnAcyclicGame();
  testCycleCheckDisabledWithDepthLimit();
  testAdvanceRootInvalid();
  testResetRoot();

  if (g_failures == 0) {
    std::cout << "All " << g_checks << " checks passed.\n";
    return 0;
  }
  std::cerr << g_failures << " of " << g_checks << " checks failed.\n";
  return 1;
}
