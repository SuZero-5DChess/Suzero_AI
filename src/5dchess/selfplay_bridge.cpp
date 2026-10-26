// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-5DChess contributors

#include "suzero/5dchess/selfplay_bridge.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#include "suzero/5dchess/onnx_batch_backend.h"
#include "suzero/mcts/batch_evaluator.h"
#include "suzero/mcts/puct_search.h"
#include "suzero/mcts/random.h"
#include "tree_positions.h"

namespace suzero::chess5d {

namespace {

using mcts::ActionId;
using mcts::IEvaluator;
using mcts::kInvalidAction;
using mcts::PuctSearch;
using mcts::Rng;
using mcts::SearchProgress;
using mcts::SearchResult;

void validateConfig(const SelfPlayBridgeConfig& config) {
  internal::validateTreePositionConfig(config);
  auto require = [](bool condition, const char* message) {
    if (!condition) {
      throw std::invalid_argument(std::string("runSelfPlay: ") + message);
    }
  };

  require(!config.evaluator.model_path.empty(), "model_path is required");
  require(config.numGameThreads >= 1, "numGameThreads must be >= 1");
  require(config.numGames >= 1, "numGames must be >= 1");
  require(config.numSimulations >= 1, "numSimulations must be >= 1");
  require(config.fastSimulations >= 1, "fastSimulations must be >= 1");
  require(config.numServerThreads >= 1, "numServerThreads must be >= 1");
  require(config.maxBatchSize >= 1, "maxBatchSize must be >= 1");
  require(config.cpuctExploration >= 0.0,
          "cpuctExploration must be >= 0");
  require(config.maxDepth >= 1, "maxDepth must be >= 1");
  require(config.dirichletAlpha > 0.0, "dirichletAlpha must be > 0");
  require(config.dirichletEpsilon >= 0.0 &&
              config.dirichletEpsilon <= 1.0,
          "dirichletEpsilon must be in [0, 1]");
  require(std::isfinite(config.fullSearchProbability) &&
              config.fullSearchProbability >= 0.0 &&
              config.fullSearchProbability <= 1.0,
          "fullSearchProbability must be in [0, 1]");
  require(config.minBoardLimit >= 1 &&
              config.minBoardLimit <= config.maxBoardLimit,
          "board limit range is invalid");
  require(std::isfinite(config.materialScale) &&
              config.materialScale >= 0.0,
          "materialScale must be finite and >= 0");
  require(config.temperatureThreshold >= 0,
          "temperatureThreshold must be >= 0");
  for (const double value : {config.temperature, config.temperatureFinal}) {
    require(std::isfinite(value) && value >= 0.0,
            "temperatures must be finite and non-negative");
  }
}

/// SplitMix64 finalizer, matching SelfPlayRunner::seedForGame: game seeds are
/// decorrelated from adjacent indices and independent of thread scheduling.
std::uint64_t seedForGame(std::uint64_t seed, int gameIndex) {
  std::uint64_t z =
      seed + 0x9E3779B97F4A7C15ULL *
                 static_cast<std::uint64_t>(static_cast<std::int64_t>(gameIndex) + 1);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

int sampleBoardLimit(const SelfPlayBridgeConfig& config, Rng& rng) {
  const int span = config.maxBoardLimit - config.minBoardLimit + 1;
  const int offset = static_cast<int>(rng.nextUint32() % static_cast<std::uint32_t>(span));
  return config.minBoardLimit + offset;
}

bool hasPositiveVisit(const std::vector<double>& distribution) {
  for (const double value : distribution) {
    if (value > 0.0) {
      return true;
    }
  }
  return false;
}

/// Advances the search root by `action`, reusing the subtree when possible.
///
/// `next` is the position after the action. A search root must have at least
/// one action, so a terminal (or action-less) position is not promoted; the
/// game loop stops before searching again, and the root is never read after
/// that point.
void applyAction(PuctSearch& search, ActionId action,
                 const mcts::IGameState& next) {
  if (next.actionCount() <= 0) {
    return;
  }
  if (search.advanceRoot(action)) {
    return;
  }
  // The edge has no expanded child: promote a clone of the already-advanced
  // game position instead of replaying the action on the old root.
  std::unique_ptr<mcts::IGameState> promoted = next.clone();
  if (!promoted) {
    throw std::runtime_error("runSelfPlay: IGameState::clone returned null");
  }
  search.resetRoot(std::shared_ptr<mcts::IGameState>(std::move(promoted)));
}

/// Mirrors python/suzero_5dchess/self_play.py::play_game.
BridgeGame playOneGame(const SelfPlayBridgeConfig& config, int index,
                       const std::shared_ptr<IEvaluator>& evaluator) {
  const std::uint64_t seed = seedForGame(config.seed, index);
  Rng rng(seed);
  // Domain-separated from both game decisions and PUCT's own RNG.
  Rng tree_rng(seed ^ 0x54524545504F5331ULL);

  GameConfig game_config = config.game;
  game_config.board_limit = sampleBoardLimit(config, rng);
  game_config.material_scale = config.materialScale;

  auto state = std::make_shared<Chess5DGameState>(game_config);

  mcts::SearchParams params;
  // The effective playout limit is the per-search callback below; PuctSearch's
  // own cap must not truncate either the full or the fast search.
  params.maxPlayouts = 0;
  params.maxDepth = config.maxDepth;
  params.cpuctExploration = config.cpuctExploration;
  params.rootDirichletNoiseWeight = config.dirichletEpsilon;
  // Python samples directly from the raw visit distribution; disable KataGo's
  // root subtract/prune so the two backends select moves alike.
  params.chosenMoveSubtract = 0.0;
  params.chosenMovePrune = 0.0;

  std::unique_ptr<PuctSearch> search;
  if (!state->isTerminal() && !state->actions().empty()) {
    std::unique_ptr<mcts::IGameState> root = state->clone();
    if (!root) {
      throw std::runtime_error("runSelfPlay: IGameState::clone returned null");
    }
    search = std::make_unique<PuctSearch>(
        std::shared_ptr<mcts::IGameState>(std::move(root)), evaluator, params,
        seed);
  }

  BridgeGame game;
  game.index = index;
  game.boardLimit = game_config.board_limit;

  int semimoves = 0;
  int turns = 0;
  int moves = 0;
  std::string reason;

  for (;;) {
    if (state->isTerminal()) {
      reason = state->terminalReason();
      if (reason.empty()) {
        reason = "terminal";
      }
      break;
    }

    const std::vector<Action>& candidates = state->actions();
    if (candidates.empty()) {
      reason = "no_legal_action";
      break;
    }

    // A forced SUBMIT carries no decision and no useful policy target.
    if (candidates.size() == 1 && candidates.front().is_submit) {
      if (config.recordGames) {
        game.moves.push_back(BridgeMove{candidates.front(),
                                        state->currentPlayer()});
      }
      state->applyAction(0);
      applyAction(*search, 0, *state);
      ++turns;
      ++moves;
      continue;
    }

    const bool full_search =
        !config.playoutCapRandomization ||
        rng.nextDouble() < config.fullSearchProbability;
    const int simulations =
        full_search ? config.numSimulations : config.fastSimulations;
    const bool use_noise = full_search && config.dirichletEpsilon > 0.0;
    search->setRootNoise(
        use_noise,
        config.dirichletAlpha * static_cast<double>(candidates.size()),
        config.dirichletEpsilon);

    const SearchResult result = search->run(
        [simulations](const SearchProgress& progress) {
          return progress.playouts >= simulations;
        });

    if (full_search) {
      const std::vector<double> distribution =
          result.denseVisitDistribution(static_cast<int>(candidates.size()));
      if (hasPositiveVisit(distribution)) {
        BridgeSample sample;
        sample.encoded = state->encodePosition();
        sample.actions = candidates;
        sample.policy.resize(distribution.size());
        for (std::size_t i = 0; i < distribution.size(); ++i) {
          sample.policy[i] = static_cast<float>(distribution[i]);
        }
        sample.player = state->currentPlayer();
        sample.moveCount = state->moveCount();
        game.samples.push_back(std::move(sample));
      }
    }

    const double temperature = semimoves < config.temperatureThreshold
                                   ? config.temperature
                                   : config.temperatureFinal;
    ActionId action = search->chooseAction(temperature);
    if (action == kInvalidAction) {
      action = result.bestAction;
    }
    if (action == kInvalidAction ||
        action < 0 || action >= static_cast<ActionId>(candidates.size())) {
      // Last resort: any legal action keeps the game moving. No sample was
      // produced above, so this cannot leak a meaningless target.
      for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
        if (state->isLegal(static_cast<ActionId>(i))) {
          action = static_cast<ActionId>(i);
          break;
        }
      }
    }
    if (action == kInvalidAction ||
        !state->isLegal(static_cast<ActionId>(action))) {
      throw std::runtime_error("runSelfPlay: search selected no legal action");
    }

    if (full_search && config.recordTreePositions) {
      auto tree_samples = internal::extractTreePositions(
          config, *state, search->rootNode(), action, tree_rng);
      game.treeSamples += static_cast<int>(tree_samples.size());
      for (auto& sample : tree_samples) {
        game.samples.push_back(std::move(sample));
      }
    }

    const Action chosen = candidates[static_cast<std::size_t>(action)];
    const bool is_submit = chosen.is_submit;
    if (config.recordGames) {
      game.moves.push_back(BridgeMove{chosen, state->currentPlayer()});
    }
    state->applyAction(action);
    applyAction(*search, action, *state);
    if (is_submit) {
      ++turns;
    } else {
      ++semimoves;
    }
    ++moves;
  }

  const bool terminal = state->isTerminal();
  double final_utility = 0.0;
  if (terminal) {
    final_utility = state->terminalUtility();
  } else if (reason == "no_legal_action") {
    // The side to move has no move: it loses (matching the Python backend).
    final_utility = -1.0;
  }

  game.terminalReason = reason;
  game.semimoves = semimoves;
  game.turns = turns;
  game.movesPlayed = moves;
  game.finalPlayer = state->currentPlayer();
  game.boardCount = state->boardCount();
  game.whiteOutcome =
      game.finalPlayer == 0 ? final_utility : -final_utility;
  if (!terminal && reason != "no_legal_action") {
    game.whiteOutcome = 0.0;
  }
  if (config.recordGames) {
    game.finalFen = state->recordSnapshot().fen;
  }

  for (BridgeSample& sample : game.samples) {
    if (sample.source == 1) {
      continue;  // Tree utility is already labelled in the node player's frame.
    }
    const double value =
        sample.player == game.finalPlayer ? final_utility : -final_utility;
    sample.value = static_cast<float>(value);
  }

  return game;
}

}  // namespace

std::vector<BridgeGame> runSelfPlay(const SelfPlayBridgeConfig& config,
                                    const BridgeGameCallback& onGameFinished,
                                    SelfPlayStats* stats,
                                    const BridgeStopCallback& shouldStop) {
  validateConfig(config);

  using Clock = std::chrono::steady_clock;
  const Clock::time_point started = Clock::now();

  mcts::BatchEvaluatorConfig batch_config;
  batch_config.maxBatchSize = config.maxBatchSize;
  batch_config.numServerThreads = config.numServerThreads;
  const std::shared_ptr<mcts::BatchedEvaluator> evaluator =
      makeBatchedOnnxEvaluator(config.evaluator, batch_config);
  const Clock::time_point setup_done = Clock::now();

  std::vector<std::unique_ptr<BridgeGame>> slots(
      static_cast<std::size_t>(config.numGames));
  // Guards "check stop, take next index" so the played games stay a contiguous
  // prefix. A signal arriving after a worker took its index lets that game
  // finish; no later game starts.
  std::mutex start_mutex;
  int next_index = 0;
  std::atomic<int> completed{0};
  std::mutex error_mutex;
  std::exception_ptr first_error;

  const int thread_count =
      std::min(config.numGameThreads, config.numGames);
  auto worker = [&]() {
    for (;;) {
      {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (first_error) {
          break;
        }
      }
      int index = 0;
      {
        std::lock_guard<std::mutex> lock(start_mutex);
        if (shouldStop && shouldStop()) {
          break;
        }
        if (next_index >= config.numGames) {
          break;
        }
        index = next_index++;
      }
      try {
        std::unique_ptr<BridgeGame> game =
            std::make_unique<BridgeGame>(playOneGame(config, index, evaluator));
        slots[static_cast<std::size_t>(index)] = std::move(game);
        const int done = completed.fetch_add(1) + 1;
        if (onGameFinished) {
          onGameFinished(done, *slots[static_cast<std::size_t>(index)]);
        }
      } catch (...) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (!first_error) {
          first_error = std::current_exception();
        }
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

  std::vector<BridgeGame> games;
  games.reserve(static_cast<std::size_t>(completed.load()));
  for (std::unique_ptr<BridgeGame>& slot : slots) {
    if (slot) {
      games.push_back(std::move(*slot));
    }
  }

  if (stats != nullptr) {
    const mcts::BatchEvaluatorStats counters = evaluator->stats();
    stats->numServerThreads = evaluator->numServerThreads();
    stats->maxBatchSize = evaluator->maxBatchSize();
    stats->requests = counters.requests;
    stats->batches = counters.batches;
    stats->rows = counters.rows;
    stats->maxObservedBatch = counters.maxObservedBatch;
    stats->setupSeconds =
        std::chrono::duration<double>(setup_done - started).count();
    stats->totalSeconds =
        std::chrono::duration<double>(Clock::now() - started).count();
  }
  return games;
}

}  // namespace suzero::chess5d
