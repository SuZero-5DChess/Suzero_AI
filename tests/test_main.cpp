// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-02 contributors
//
// Unit tests for the game-agnostic PUCT search. Toy games only; no engine is
// linked or modified.

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "suzero/mcts/batch_evaluator.h"
#include "suzero/mcts/puct_search.h"
#include "suzero/mcts/selfplay.h"
#include "internal_math.h"
#include "toy_games.h"

namespace {

using suzero::mcts::ActionStats;
using suzero::mcts::BatchEvaluatorConfig;
using suzero::mcts::BatchedEvaluator;
using suzero::mcts::Evaluation;
using suzero::mcts::IBatchBackend;
using suzero::mcts::PuctSearch;
using suzero::mcts::SearchParams;
using suzero::mcts::SearchProgress;
using suzero::mcts::SearchResult;
using suzero::mcts::SelfPlayConfig;
using suzero::mcts::SelfPlayGame;
using suzero::mcts::SelfPlayRunner;
using suzero::mcts::SelfPlaySample;
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
    const double suzero_check_actual = (actual);                             \
    const double suzero_check_expected = (expected);                         \
    if (!(std::fabs(suzero_check_actual - suzero_check_expected) <=           \
          (tolerance))) {                                                    \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": "            \
                << #actual << " = " << suzero_check_actual                   \
                << ", expected " << suzero_check_expected << " +/- "         \
                << (tolerance) << "\n";                                      \
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

/// Synchronous evaluator wrapped in a batch backend. The delay widens the
/// window in which other client threads can enqueue, so batch sizes above one
/// are reliably observable.
class CountingBatchBackend final : public IBatchBackend {
 public:
  CountingBatchBackend(std::shared_ptr<suzero::mcts::IEvaluator> inner,
                       int max_batch,
                       std::atomic<std::uint64_t>* batches,
                       std::atomic<std::uint64_t>* max_observed,
                       std::chrono::microseconds delay)
      : inner_(std::move(inner)),
        max_batch_(max_batch),
        batches_(batches),
        max_observed_(max_observed),
        delay_(delay) {}

  int maxBatchSize() const override { return max_batch_; }

  void evaluateBatch(const std::vector<const suzero::mcts::IGameState*>& states,
                     std::vector<Evaluation>& results) override {
    if (batches_ != nullptr) {
      batches_->fetch_add(1, std::memory_order_relaxed);
    }
    if (max_observed_ != nullptr) {
      updateMax(static_cast<std::uint64_t>(states.size()));
    }
    if (delay_.count() > 0) {
      std::this_thread::sleep_for(delay_);
    }
    results.clear();
    results.reserve(states.size());
    for (const suzero::mcts::IGameState* state : states) {
      results.push_back(inner_->evaluate(*state));
    }
  }

 private:
  void updateMax(std::uint64_t observed) {
    std::uint64_t current = max_observed_->load(std::memory_order_relaxed);
    while (observed > current &&
           !max_observed_->compare_exchange_weak(
               current, observed, std::memory_order_relaxed)) {
    }
  }

  std::shared_ptr<suzero::mcts::IEvaluator> inner_;
  int max_batch_;
  std::atomic<std::uint64_t>* batches_;
  std::atomic<std::uint64_t>* max_observed_;
  std::chrono::microseconds delay_;
};

class PoisonBatchBackend final : public IBatchBackend {
 public:
  explicit PoisonBatchBackend(int poison_position)
      : poison_position_(poison_position) {}

  int maxBatchSize() const override { return 16; }

  void evaluateBatch(const std::vector<const suzero::mcts::IGameState*>& states,
                     std::vector<Evaluation>& results) override {
    for (const suzero::mcts::IGameState* state : states) {
      const auto* game = dynamic_cast<const SubtractionGame*>(state);
      if (game != nullptr && game->position() == poison_position_) {
        throw std::runtime_error("poisoned batch");
      }
    }
    results.clear();
    results.reserve(states.size());
    for (const suzero::mcts::IGameState* state : states) {
      UniformEvaluator inner;
      results.push_back(inner.evaluate(*state));
    }
  }

 private:
  int poison_position_;
};

/// Runs the same concurrent workload against `server_threads` servers, which
/// also exercises several threads draining one shared queue.
void checkBatchedEvaluatorConcurrently(int server_threads) {
  auto inner = std::make_shared<UniformEvaluator>();
  std::atomic<std::uint64_t> batches{0};
  std::atomic<std::uint64_t> max_observed{0};

  BatchEvaluatorConfig config;
  config.maxBatchSize = 8;
  config.numServerThreads = server_threads;
  config.maxQueuedRequests = 64;

  BatchedEvaluator evaluator(
      config, [&]() {
        return std::make_unique<CountingBatchBackend>(
            inner, 8, &batches, &max_observed, std::chrono::milliseconds(5));
      });

  std::vector<std::shared_ptr<SubtractionGame>> states;
  std::vector<Evaluation> expected;
  for (int position = 0; position < 5; ++position) {
    states.push_back(
        std::make_shared<SubtractionGame>(position, position % 2));
    expected.push_back(inner->evaluate(*states.back()));
  }

  constexpr int kThreads = 8;
  constexpr int kPerThread = 24;
  std::atomic<int> mismatches{0};
  std::atomic<bool> start{false};

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int thread_index = 0; thread_index < kThreads; ++thread_index) {
    threads.emplace_back([&, thread_index]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int iteration = 0; iteration < kPerThread; ++iteration) {
        const std::size_t index =
            static_cast<std::size_t>((thread_index + iteration) %
                                     static_cast<int>(states.size()));
        const Evaluation actual = evaluator.evaluate(*states[index]);
        const Evaluation& reference = expected[index];
        if (actual.policy.size() != reference.policy.size()) {
          mismatches.fetch_add(1, std::memory_order_relaxed);
          continue;
        }
        for (std::size_t i = 0; i < actual.policy.size(); ++i) {
          if (actual.policy[i] != reference.policy[i]) {
            mismatches.fetch_add(1, std::memory_order_relaxed);
            break;
          }
        }
        if (actual.utility != reference.utility) {
          mismatches.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (std::thread& thread : threads) {
    thread.join();
  }

  CHECK(mismatches.load() == 0);
  const auto stats = evaluator.stats();
  CHECK(stats.requests == kThreads * kPerThread);
  CHECK(stats.rows == kThreads * kPerThread);
  CHECK(stats.batches > 0);
  CHECK(stats.maxObservedBatch >= 2);
  CHECK(stats.maxObservedBatch <=
        static_cast<std::uint64_t>(evaluator.maxBatchSize()));
  CHECK(evaluator.numServerThreads() == server_threads);
}

void testBatchedEvaluatorMatchesDirectEvaluation() {
  checkBatchedEvaluatorConcurrently(1);
}

void testBatchedEvaluatorMultipleServerThreads() {
  checkBatchedEvaluatorConcurrently(2);
}

void testBatchedEvaluatorFastCompletionLifetime() {
  auto inner = std::make_shared<UniformEvaluator>();
  std::atomic<std::uint64_t> backend_requests{0};
  std::atomic<std::uint64_t> max_observed{0};

  BatchEvaluatorConfig config;
  config.maxBatchSize = 1;
  config.numServerThreads = 2;
  config.maxQueuedRequests = 64;

  BatchedEvaluator evaluator(
      config, [&]() {
        return std::make_unique<CountingBatchBackend>(
            inner, 1, &backend_requests, &max_observed,
            std::chrono::microseconds(0));
      });

  std::vector<std::shared_ptr<SubtractionGame>> states;
  std::vector<Evaluation> expected;
  for (int position = 0; position < 5; ++position) {
    states.push_back(
        std::make_shared<SubtractionGame>(position, position % 2));
    expected.push_back(inner->evaluate(*states.back()));
  }

  constexpr int kThreads = 8;
  constexpr int kPerThread = 1000;
  constexpr std::uint64_t kTotalRequests = kThreads * kPerThread;
  std::atomic<std::uint64_t> completed{0};
  std::atomic<int> mismatches{0};
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int thread_index = 0; thread_index < kThreads; ++thread_index) {
    threads.emplace_back([&, thread_index]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int iteration = 0; iteration < kPerThread; ++iteration) {
        const std::size_t index =
            static_cast<std::size_t>((thread_index + iteration) %
                                     static_cast<int>(states.size()));
        const Evaluation actual = evaluator.evaluate(*states[index]);
        if (actual.policy != expected[index].policy ||
            actual.utility != expected[index].utility) {
          mismatches.fetch_add(1, std::memory_order_relaxed);
        }
        completed.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (std::thread& thread : threads) {
    thread.join();
  }

  CHECK(mismatches.load() == 0);
  CHECK(completed.load() == kTotalRequests);
  CHECK(backend_requests.load() == kTotalRequests);
  CHECK(evaluator.stats().requests == kTotalRequests);
  CHECK(max_observed.load() == 1);
}

void testBatchedEvaluatorPropagatesBackendErrors() {
  BatchEvaluatorConfig config;
  config.maxBatchSize = 4;
  config.numServerThreads = 1;

  BatchedEvaluator evaluator(
      config, []() { return std::make_unique<PoisonBatchBackend>(3); });

  SubtractionGame poisoned(3, 0);
  bool threw = false;
  try {
    (void)evaluator.evaluate(poisoned);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw);

  // A failed batch must not kill the server thread.
  SubtractionGame healthy(1, 0);
  const Evaluation result = evaluator.evaluate(healthy);
  CHECK(result.policy.size() ==
        static_cast<std::size_t>(healthy.actionCount()));
}

void testBatchedEvaluatorValidatesConfig() {
  BatchEvaluatorConfig bad_batch;
  bad_batch.maxBatchSize = 0;
  bool threw = false;
  try {
    BatchedEvaluator evaluator(bad_batch, []() {
      return std::make_unique<CountingBatchBackend>(
          std::make_shared<UniformEvaluator>(), 0, nullptr, nullptr,
          std::chrono::microseconds(0));
    });
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);

  threw = false;
  try {
    BatchEvaluatorConfig config;
    BatchedEvaluator evaluator(config, []() { return nullptr; });
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

SelfPlayConfig selfPlayConfigForTests() {
  SelfPlayConfig config;
  config.numGameThreads = 4;
  config.maxGames = 12;
  config.maxMovesPerGame = 20;
  config.searchParams = paramsWithPlayouts(24);
  config.searchParams.chosenMoveTemperature = 1.0;
  config.seed = 12345;
  return config;
}

std::shared_ptr<BatchedEvaluator> makeSelfPlayEvaluator(
    std::atomic<std::uint64_t>* max_observed) {
  BatchEvaluatorConfig config;
  config.maxBatchSize = 8;
  config.numServerThreads = 1;
  return std::make_shared<BatchedEvaluator>(
      config, [max_observed]() {
        return std::make_unique<CountingBatchBackend>(
            std::make_shared<UniformEvaluator>(), 8, nullptr, max_observed,
            std::chrono::microseconds(200));
      });
}

void checkSelfPlayGame(const SelfPlayGame& game) {
  CHECK(game.gameIndex >= 0);
  CHECK(game.terminal);
  CHECK(!game.truncated);
  CHECK(game.movesPlayed > 0);
  CHECK(!game.samples.empty());
  CHECK(std::isfinite(game.finalUtility));
  for (const auto& sample : game.samples) {
    CHECK(sample.state != nullptr);
    if (sample.state == nullptr) {
      continue;
    }
    CHECK(sample.policy.size() ==
          static_cast<std::size_t>(sample.state->actionCount()));
    CHECK(sample.player == sample.state->currentPlayer());
    double sum = 0.0;
    for (float probability : sample.policy) {
      CHECK(std::isfinite(probability));
      CHECK(probability >= 0.0f);
      sum += probability;
    }
    CHECK_NEAR(sum, 1.0, 1e-4);
    const double expected_value =
        sample.player == game.finalPlayer ? game.finalUtility
                                          : -game.finalUtility;
    CHECK_NEAR(sample.value, expected_value, 1e-12);
  }
}

void testSelfPlayRunner() {
  std::atomic<std::uint64_t> max_observed{0};
  auto evaluator = makeSelfPlayEvaluator(&max_observed);
  SelfPlayRunner runner(
      selfPlayConfigForTests(),
      []() { return std::make_shared<SubtractionGame>(0, 0); }, evaluator);

  const std::vector<SelfPlayGame> games = runner.run();
  CHECK(games.size() == 12);
  for (const SelfPlayGame& game : games) {
    checkSelfPlayGame(game);
  }
  CHECK(evaluator->stats().rows > 0);
  CHECK(max_observed.load() >= 1);

  // Per-game seeds make results independent of thread scheduling.
  std::atomic<std::uint64_t> unused_max{0};
  auto second_evaluator = makeSelfPlayEvaluator(&unused_max);
  SelfPlayRunner second(
      selfPlayConfigForTests(),
      []() { return std::make_shared<SubtractionGame>(0, 0); },
      second_evaluator);
  const std::vector<SelfPlayGame> second_games = second.run();
  CHECK(second_games.size() == games.size());
  for (std::size_t game_index = 0;
       game_index < games.size() && game_index < second_games.size();
       ++game_index) {
    const SelfPlayGame& first_game = games[game_index];
    const SelfPlayGame& second_game = second_games[game_index];
    CHECK(first_game.samples.size() == second_game.samples.size());
    CHECK(first_game.movesPlayed == second_game.movesPlayed);
    for (std::size_t sample_index = 0;
         sample_index < first_game.samples.size() &&
         sample_index < second_game.samples.size();
         ++sample_index) {
      CHECK(first_game.samples[sample_index].state->stateKey() ==
            second_game.samples[sample_index].state->stateKey());
      CHECK(first_game.samples[sample_index].policy ==
            second_game.samples[sample_index].policy);
    }
  }
}

void testSelfPlayForcedMoveToZeroActionTerminal() {
  class ForcedEndGame final : public suzero::mcts::IGameState {
   public:
    explicit ForcedEndGame(int remaining) : remaining_(remaining) {}
    std::unique_ptr<IGameState> clone() const override {
      return std::make_unique<ForcedEndGame>(*this);
    }
    int actionCount() const override { return remaining_; }
    bool isLegal(suzero::mcts::ActionId action) const override {
      return action >= 0 && action < actionCount();
    }
    void applyAction(suzero::mcts::ActionId) override {
      --remaining_;
      player_ = 1 - player_;
    }
    suzero::mcts::PlayerId currentPlayer() const override { return player_; }
    bool isTerminal() const override { return remaining_ == 0; }
    double terminalUtility() const override { return -1.0; }
    suzero::mcts::StateKey stateKey() const override {
      return "forced-end:" + std::to_string(remaining_) + ":" +
             std::to_string(player_);
    }

   private:
    int remaining_;
    suzero::mcts::PlayerId player_ = 0;
  };

  for (const int remaining : {1, 2}) {
    for (const int playouts : {1, 24}) {
      SelfPlayConfig config = selfPlayConfigForTests();
      config.maxGames = 2;
      config.numGameThreads = 1;
      config.searchParams = paramsWithPlayouts(playouts);
      SelfPlayRunner runner(
          config, [remaining]() { return std::make_shared<ForcedEndGame>(remaining); },
          std::make_shared<UniformEvaluator>());
      const auto games = runner.run();
      CHECK(games.size() == 2);
      for (const auto& game : games) {
        CHECK(game.terminal);
        CHECK(!game.truncated);
        CHECK(game.movesPlayed == remaining);
        CHECK(game.finalPlayer == remaining % 2);
        CHECK_NEAR(game.finalUtility, -1.0, 1e-12);
        if (remaining == 1) {
          CHECK(game.samples.empty());
        } else {
          CHECK(game.samples.size() == 1);
          for (const auto& sample : game.samples) {
            CHECK(sample.player == 0);
            CHECK_NEAR(sample.value, -1.0, 1e-12);
          }
        }
      }
    }
  }
}

void testSelfPlayRunnerStreaming() {
  std::atomic<std::uint64_t> max_observed{0};
  auto evaluator = makeSelfPlayEvaluator(&max_observed);
  SelfPlayConfig config = selfPlayConfigForTests();
  config.maxGames = 5;
  SelfPlayRunner runner(
      config, []() { return std::make_shared<SubtractionGame>(0, 0); },
      evaluator);

  std::mutex mutex;
  std::vector<std::int64_t> indices;
  std::vector<SelfPlayGame> games;
  const std::size_t completed =
      runner.run([&](std::int64_t index, SelfPlayGame&& game) {
        std::lock_guard<std::mutex> lock(mutex);
        indices.push_back(index);
        games.push_back(std::move(game));
      });

  CHECK(completed == 5);
  CHECK(indices.size() == 5);
  CHECK(games.size() == 5);
  for (const SelfPlayGame& game : games) {
    checkSelfPlayGame(game);
  }
}

void testSelfPlayRunnerStop() {
  std::atomic<std::uint64_t> max_observed{0};
  SelfPlayConfig config = selfPlayConfigForTests();
  config.maxGames = 0;  // Unlimited: only requestStop() ends the run.
  config.numGameThreads = 2;
  SelfPlayRunner runner(
      config, []() { return std::make_shared<SubtractionGame>(0, 0); },
      makeSelfPlayEvaluator(&max_observed));

  std::mutex mutex;
  std::vector<SelfPlayGame> games;
  const std::size_t completed =
      runner.run([&](std::int64_t, SelfPlayGame&& game) {
        std::lock_guard<std::mutex> lock(mutex);
        games.push_back(std::move(game));
        if (games.size() == 3) {
          runner.requestStop();
        }
      });

  CHECK(completed == games.size());
  CHECK(games.size() >= 3);
  // At most one in-flight game per worker can still finish after the stop.
  CHECK(games.size() <= 3 + static_cast<std::size_t>(config.numGameThreads));
  CHECK(runner.stopRequested());
  for (const SelfPlayGame& game : games) {
    // Stop-produced and naturally finished games are both valid outputs, but
    // a game cannot be terminal and truncated at the same time.
    CHECK(game.terminal != game.truncated);
  }
}

void testSelfPlayMoveCapTruncates() {
  std::atomic<std::uint64_t> max_observed{0};
  SelfPlayConfig config = selfPlayConfigForTests();
  config.numGameThreads = 1;
  config.maxGames = 1;
  config.maxMovesPerGame = 1;  // Reaching 5 takes at least three moves.
  SelfPlayRunner runner(
      config, []() { return std::make_shared<SubtractionGame>(0, 0); },
      makeSelfPlayEvaluator(&max_observed));

  const std::vector<SelfPlayGame> games = runner.run();
  CHECK(games.size() == 1);
  CHECK(games.front().truncated);
  CHECK(!games.front().terminal);
  CHECK(games.front().movesPlayed == 1);
  CHECK_NEAR(games.front().finalUtility, 0.0, 1e-12);
  for (const SelfPlaySample& sample : games.front().samples) {
    CHECK_NEAR(sample.value, 0.0, 1e-12);
  }
}

void testSelfPlayRunnerRejectsUnboundedCollect() {
  SelfPlayConfig config = selfPlayConfigForTests();
  config.maxGames = 0;
  std::atomic<std::uint64_t> max_observed{0};
  SelfPlayRunner runner(
      config, []() { return std::make_shared<SubtractionGame>(0, 0); },
      makeSelfPlayEvaluator(&max_observed));

  bool threw = false;
  try {
    (void)runner.run();
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
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

void testStudentTCdf() {
  using suzero::mcts::internal::studentTCdf3;
  CHECK_NEAR(studentTCdf3(0.0), 0.5, 1e-12);
  CHECK_NEAR(studentTCdf3(1.0), 0.804498891, 1e-8);
  CHECK_NEAR(studentTCdf3(-1.0), 0.195501109, 1e-8);
  CHECK_NEAR(studentTCdf3(2.0), 0.930337016, 1e-8);
  CHECK_NEAR(studentTCdf3(1.0) + studentTCdf3(-1.0), 1.0, 1e-12);
}

void testZeroProbabilityLegalActionsRemainVisible() {
  PuctSearch search(std::make_shared<OneMoveWinGame>(),
                    std::make_shared<ZeroActionPriorEvaluator>(),
                    paramsWithPlayouts(300), 7);
  const SearchResult result = runSearch(search);

  CHECK(result.actions.size() == 3);
  bool found_zero = false;
  for (const ActionStats& stats : result.actions) {
    if (stats.action == 0) {
      found_zero = true;
      CHECK_NEAR(stats.prior, 0.0, 0.0);
    }
  }
  CHECK(found_zero);
  CHECK(result.bestAction == 2);
}

void testAllZeroPolicyStillSearches() {
  PuctSearch search(std::make_shared<OneMoveWinGame>(),
                    std::make_shared<AllZeroPriorEvaluator>(),
                    paramsWithPlayouts(100), 11);
  const SearchResult result = runSearch(search);

  CHECK(result.actions.size() == 3);
  CHECK(result.bestAction == 2);
  CHECK(search.chooseAction(0.0) == 2);
}

void testRootPolicyTemperatureKeepsNearTies() {
  SearchParams params;
  params.maxVisits = 1;
  params.maxPlayouts = 0;
  params.rootPolicyTemperature = 1e-4;
  params.rootPolicyTemperatureEarly = 1e-4;

  PuctSearch search(std::make_shared<OneMoveWinGame>(),
                    std::make_shared<UniformEvaluator>(), params, 3);
  const SearchResult result = runSearch(search);

  // Equal priors must stay equal. A hard one-hot limit would make action 0
  // certain and the others zero.
  CHECK(result.actions.size() == 3);
  for (const ActionStats& stats : result.actions) {
    CHECK_NEAR(stats.prior, 1.0 / 3.0, 1e-6);
  }
}

void testDirichletNoiseIsUniformForPlausibleMoves() {
  SearchParams params;
  params.maxVisits = 1;
  params.maxPlayouts = 0;
  params.rootNoiseEnabled = true;
  params.rootDirichletNoiseWeight = 1.0;
  params.rootDirichletNoiseTotalConcentration = 10.83;

  // All three priors are above 1%, so KataGo's min(0.01, p) alpha is equal
  // for all of them and the noise is a symmetric Dirichlet draw. The old
  // max(0.01, p) implementation would concentrate the noise on action 0.
  constexpr int kTrials = 300;
  double sums[3] = {0.0, 0.0, 0.0};
  for (int trial = 0; trial < kTrials; ++trial) {
    PuctSearch search(
        std::make_shared<OneMoveWinGame>(),
        std::make_shared<SkewedWinEvaluator>(), params,
        9000 + static_cast<std::uint64_t>(trial));
    const SearchResult result = runSearch(search);
    CHECK(result.actions.size() == 3);
    for (std::size_t i = 0; i < result.actions.size() && i < 3; ++i) {
      sums[i] += result.actions[i].prior;
    }
  }
  for (int i = 0; i < 3; ++i) {
    CHECK_NEAR(sums[i] / kTrials, 1.0 / 3.0, 0.035);
  }
}

void testAdvanceRootDropsUnreachablePromotedCopy() {
  PuctSearch search(std::make_shared<CommutingGame>(),
                    std::make_shared<UniformEvaluator>(),
                    paramsWithPlayouts(800), 5);
  runSearch(search);
  CHECK(search.nodeCount() == 4);

  CHECK(search.advanceRoot(0));
  // New root + the shared terminal child. The promoted node's old table copy
  // is unreachable and must be dropped, as KataGo's sweep does.
  CHECK(search.nodeCount() == 2);
}

void testBestActionMatchesGreedyChoice() {
  PuctSearch search(std::make_shared<SubtractionGame>(0, 0),
                    std::make_shared<UniformEvaluator>(),
                    paramsWithPlayouts(1000), 13);
  const SearchResult result = runSearch(search);
  CHECK(result.bestAction == search.chooseAction(0.0));

  PuctSearch win(std::make_shared<OneMoveWinGame>(),
                 std::make_shared<SkewedWinEvaluator>(),
                 paramsWithPlayouts(500), 17);
  const SearchResult win_result = runSearch(win);
  CHECK(win_result.bestAction == win.chooseAction(0.0));
}

void testEarlyTemperatureInterpolation() {
  SearchParams params;
  params.maxVisits = 1;
  params.maxPlayouts = 0;
  params.rootPolicyTemperature = 1.0;
  params.rootPolicyTemperatureEarly = 1e-4;
  params.chosenMoveTemperatureHalflife = 19.0;

  PuctSearch at_start(std::make_shared<SubtractionGame>(0, 0, 0),
                      std::make_shared<SkewedWinEvaluator>(), params, 19);
  const SearchResult start_result = runSearch(at_start);
  CHECK(start_result.actions.size() == 2);
  CHECK(start_result.actions[0].prior > 0.999);

  PuctSearch after_halflife(std::make_shared<SubtractionGame>(0, 0, 19),
                            std::make_shared<SkewedWinEvaluator>(), params,
                            19);
  const SearchResult middle_result = runSearch(after_halflife);
  // After one halflife the temperature is about 0.5, so the priors are
  // proportional to p^2: 0.36 : 0.0625.
  CHECK(middle_result.actions.size() == 2);
  CHECK_NEAR(middle_result.actions[0].prior, 0.36 / (0.36 + 0.0625), 2e-3);
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
  testStudentTCdf();
  testZeroProbabilityLegalActionsRemainVisible();
  testAllZeroPolicyStillSearches();
  testRootPolicyTemperatureKeepsNearTies();
  testDirichletNoiseIsUniformForPlausibleMoves();
  testAdvanceRootDropsUnreachablePromotedCopy();
  testBestActionMatchesGreedyChoice();
  testEarlyTemperatureInterpolation();
  testBatchedEvaluatorMatchesDirectEvaluation();
  testBatchedEvaluatorMultipleServerThreads();
  testBatchedEvaluatorFastCompletionLifetime();
  testBatchedEvaluatorPropagatesBackendErrors();
  testBatchedEvaluatorValidatesConfig();
  testSelfPlayRunner();
  testSelfPlayForcedMoveToZeroActionTerminal();
  testSelfPlayRunnerStreaming();
  testSelfPlayRunnerStop();
  testSelfPlayMoveCapTruncates();
  testSelfPlayRunnerRejectsUnboundedCollect();

  if (g_failures == 0) {
    std::cout << "All " << g_checks << " checks passed.\n";
    return 0;
  }
  std::cerr << g_failures << " of " << g_checks << " checks failed.\n";
  return 1;
}
