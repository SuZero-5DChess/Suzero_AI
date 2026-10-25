// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-02 contributors

#include "suzero/mcts/selfplay.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#include "suzero/mcts/puct_search.h"

namespace suzero::mcts {

namespace {

/// Advances the search root, or returns a terminal state without resetting it.
std::unique_ptr<IGameState> applyRootAction(PuctSearch& search, ActionId action) {
  if (search.advanceRoot(action)) {
    return nullptr;
  }

  std::unique_ptr<IGameState> next = search.rootState().clone();
  if (!next) {
    throw std::runtime_error("SelfPlayRunner: IGameState::clone returned null");
  }
  next->applyAction(action);
  if (next->isTerminal()) {
    return next;
  }
  search.resetRoot(std::shared_ptr<IGameState>(std::move(next)));
  return nullptr;
}

bool hasPositiveVisit(const std::vector<double>& distribution) {
  for (const double value : distribution) {
    if (value > 0.0) {
      return true;
    }
  }
  return false;
}

}  // namespace

SelfPlayRunner::SelfPlayRunner(SelfPlayConfig config, StateFactory stateFactory,
                               std::shared_ptr<IEvaluator> evaluator)
    : config_(std::move(config)),
      state_factory_(std::move(stateFactory)),
      evaluator_(std::move(evaluator)) {
  if (config_.numGameThreads <= 0) {
    throw std::invalid_argument(
        "SelfPlayRunner: numGameThreads must be positive");
  }
  if (config_.maxMovesPerGame < 0) {
    throw std::invalid_argument(
        "SelfPlayRunner: maxMovesPerGame must be non-negative");
  }
  if (!state_factory_) {
    throw std::invalid_argument("SelfPlayRunner: stateFactory is required");
  }
  if (!evaluator_) {
    throw std::invalid_argument("SelfPlayRunner: evaluator is required");
  }
}

SelfPlayRunner::~SelfPlayRunner() = default;

std::uint64_t SelfPlayRunner::seedForGame(std::int64_t gameIndex) const {
  // SplitMix64 finalizer: decorrelates adjacent game indices.
  std::uint64_t z =
      config_.seed +
      0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(gameIndex + 1);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

SelfPlayGame SelfPlayRunner::playGame(std::int64_t gameIndex) {
  std::shared_ptr<IGameState> initial = state_factory_();
  if (!initial) {
    throw std::runtime_error("SelfPlayRunner: stateFactory returned null");
  }
  std::unique_ptr<IGameState> root = initial->clone();
  if (!root) {
    throw std::runtime_error("SelfPlayRunner: IGameState::clone returned null");
  }

  SelfPlayGame game;
  game.gameIndex = gameIndex;

  PuctSearch search(std::shared_ptr<IGameState>(std::move(root)), evaluator_,
                    config_.searchParams, seedForGame(gameIndex));

  std::unique_ptr<IGameState> terminal_state;
  int moves = 0;
  for (;;) {
    const IGameState& root_state = search.rootState();
    if (root_state.isTerminal()) {
      game.terminal = true;
      break;
    }
    if (stop_requested_.load(std::memory_order_relaxed)) {
      game.truncated = true;
      break;
    }
    if (config_.maxMovesPerGame > 0 && moves >= config_.maxMovesPerGame) {
      game.truncated = true;
      break;
    }

    const int action_count = root_state.actionCount();
    if (action_count <= 0) {
      game.truncated = true;
      break;
    }

    // A single candidate (for 5D chess: a forced SUBMIT) carries no decision
    // and no useful policy target, so it needs neither search nor a sample.
    if (action_count == 1) {
      if (!root_state.isLegal(0)) {
        game.truncated = true;
        break;
      }
      terminal_state = applyRootAction(search, 0);
      ++moves;
      if (terminal_state) {
        game.terminal = true;
        break;
      }
      continue;
    }

    const SearchResult result = search.run();
    const std::vector<double> visits =
        result.denseVisitDistribution(action_count);
    if (hasPositiveVisit(visits)) {
      SelfPlaySample sample;
      sample.state = root_state.clone();
      if (!sample.state) {
        throw std::runtime_error(
            "SelfPlayRunner: IGameState::clone returned null");
      }
      sample.policy.resize(visits.size());
      for (std::size_t i = 0; i < visits.size(); ++i) {
        sample.policy[i] = static_cast<float>(visits[i]);
      }
      sample.player = root_state.currentPlayer();
      sample.moveIndex = moves;
      game.samples.push_back(std::move(sample));
    }

    ActionId action = search.chooseAction();
    if (action == kInvalidAction) {
      action = result.bestAction;
    }
    if (action == kInvalidAction) {
      // Last resort for pathological priors: any legal action keeps the game
      // moving. The position produced no sample above, so this cannot leak a
      // meaningless target into the training data.
      for (int candidate = 0; candidate < action_count; ++candidate) {
        if (root_state.isLegal(static_cast<ActionId>(candidate))) {
          action = static_cast<ActionId>(candidate);
          break;
        }
      }
    }
    if (action == kInvalidAction ||
        !root_state.isLegal(static_cast<ActionId>(action))) {
      throw std::runtime_error(
          "SelfPlayRunner: search selected no legal action");
    }

    terminal_state = applyRootAction(search, action);
    ++moves;
    if (terminal_state) {
      game.terminal = true;
      break;
    }
  }

  game.movesPlayed = moves;
  const IGameState& final_state =
      terminal_state ? *terminal_state : search.rootState();
  game.finalPlayer = final_state.currentPlayer();
  game.finalUtility = game.terminal ? final_state.terminalUtility() : 0.0;

  // Two-player zero-sum convention: the stored utility is from the final
  // player's perspective, so samples taken while the other side was to move
  // receive the negated outcome.
  for (SelfPlaySample& sample : game.samples) {
    sample.value = sample.player == game.finalPlayer ? game.finalUtility
                                                     : -game.finalUtility;
  }

  return game;
}

std::size_t SelfPlayRunner::runImpl(
    std::vector<std::unique_ptr<SelfPlayGame>>* collected,
    const GameCallback* onGameFinished) {
  stop_requested_.store(false, std::memory_order_relaxed);

  const bool limited = config_.maxGames > 0;
  const std::int64_t limit =
      limited ? static_cast<std::int64_t>(config_.maxGames)
              : std::numeric_limits<std::int64_t>::max();
  const int requested_threads = config_.numGameThreads;
  const int thread_count =
      limited ? std::min(requested_threads, config_.maxGames)
              : requested_threads;

  std::atomic<std::int64_t> next_index{0};
  std::atomic<std::size_t> completed{0};
  std::mutex slots_mutex;
  std::mutex error_mutex;
  std::exception_ptr first_error;

  auto worker = [&]() {
    for (;;) {
      if (stop_requested_.load(std::memory_order_relaxed)) {
        break;
      }
      const std::int64_t index = next_index.fetch_add(1);
      if (index >= limit) {
        break;
      }

      try {
        SelfPlayGame game = playGame(index);
        if (onGameFinished != nullptr) {
          (*onGameFinished)(index, std::move(game));
        } else {
          std::lock_guard<std::mutex> lock(slots_mutex);
          (*collected)[static_cast<std::size_t>(index)] =
              std::make_unique<SelfPlayGame>(std::move(game));
        }
        completed.fetch_add(1, std::memory_order_relaxed);
      } catch (...) {
        {
          std::lock_guard<std::mutex> lock(error_mutex);
          if (!first_error) {
            first_error = std::current_exception();
          }
        }
        stop_requested_.store(true, std::memory_order_relaxed);
        break;
      }
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(thread_count));
  for (int i = 0; i < thread_count; ++i) {
    threads.emplace_back(worker);
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  if (first_error) {
    std::rethrow_exception(first_error);
  }
  return completed.load(std::memory_order_relaxed);
}

std::vector<SelfPlayGame> SelfPlayRunner::run() {
  if (config_.maxGames <= 0) {
    throw std::invalid_argument(
        "SelfPlayRunner::run: maxGames must be positive; use the streaming "
        "overload for an unlimited run");
  }

  std::vector<std::unique_ptr<SelfPlayGame>> slots(
      static_cast<std::size_t>(config_.maxGames));
  runImpl(&slots, nullptr);

  std::vector<SelfPlayGame> games;
  games.reserve(slots.size());
  for (std::unique_ptr<SelfPlayGame>& slot : slots) {
    if (!slot) {
      throw std::runtime_error(
          "SelfPlayRunner::run: a game did not produce a result");
    }
    games.push_back(std::move(*slot));
  }
  return games;
}

std::size_t SelfPlayRunner::run(const GameCallback& onGameFinished) {
  if (!onGameFinished) {
    throw std::invalid_argument(
        "SelfPlayRunner::run: onGameFinished is required");
  }
  return runImpl(nullptr, &onGameFinished);
}

void SelfPlayRunner::requestStop() {
  stop_requested_.store(true, std::memory_order_relaxed);
}

bool SelfPlayRunner::stopRequested() const {
  return stop_requested_.load(std::memory_order_relaxed);
}

}  // namespace suzero::mcts
