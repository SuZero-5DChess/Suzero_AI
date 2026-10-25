// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-5DChess contributors
//
// Acceptance tests for the 5D chess integration:
//   Chess5DGameState (IGameState) + OnnxEvaluator (IEvaluator) + PuctSearch.
//
// The tests use the exported very_small smallstep_movekey_v1 model. They cover
// the state contract, the legacy-engine rules wrapper, the ONNX protocol, and
// an end-to-end search/play loop.

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "suzero/5dchess/agent.h"
#include "suzero/5dchess/game_state.h"
#include "suzero/5dchess/onnx_batch_backend.h"
#include "suzero/5dchess/onnx_evaluator.h"
#include "suzero/5dchess/selfplay_bridge.h"
#include "suzero/mcts/selfplay.h"

#ifndef SUZERO_5DCHESS_MODEL_PATH
#error "SUZERO_5DCHESS_MODEL_PATH must be defined by CMake"
#endif

namespace {

using suzero::chess5d::Action;
using suzero::chess5d::AgentConfig;
using suzero::chess5d::BridgeGame;
using suzero::chess5d::BridgeSample;
using suzero::chess5d::Decision;
using suzero::chess5d::Chess5DAgent;
using suzero::chess5d::Chess5DGameState;
using suzero::chess5d::GameConfig;
using suzero::chess5d::MoveKey;
using suzero::chess5d::OnnxEvaluator;
using suzero::chess5d::OnnxEvaluatorConfig;
using suzero::chess5d::RecordSnapshot;
using suzero::chess5d::SelfPlayBridgeConfig;
using suzero::chess5d::SelfPlayStats;
using namespace suzero::mcts;

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

const char* modelPath() { return SUZERO_5DCHESS_MODEL_PATH; }

int findSubmit(const Chess5DGameState& game) {
  for (int i = 0; i < game.actionCount(); ++i) {
    if (game.action(i).is_submit) {
      return i;
    }
  }
  return -1;
}

int findMove(const Chess5DGameState& game, const MoveKey& move) {
  for (int i = 0; i < game.actionCount(); ++i) {
    if (!game.action(i).is_submit && game.action(i).move == move) {
      return i;
    }
  }
  return -1;
}

/// Minimal state used only to check that the evaluator rejects foreign games.
class ForeignState final : public IGameState {
 public:
  std::unique_ptr<IGameState> clone() const override {
    return std::make_unique<ForeignState>(*this);
  }
  int actionCount() const override { return 1; }
  bool isLegal(ActionId action) const override { return action == 0; }
  void applyAction(ActionId) override {}
  PlayerId currentPlayer() const override { return 0; }
  bool isTerminal() const override { return false; }
  double terminalUtility() const override { return 0.0; }
  StateKey stateKey() const override { return "foreign"; }
};

void testInitialState() {
  Chess5DGameState game;

  CHECK(!game.isTerminal());
  CHECK(game.currentPlayer() == 0);
  CHECK(game.actionCount() > 1);
  CHECK(static_cast<int>(game.actions().size()) == game.actionCount());
  CHECK(game.boardCount() >= 1);
  CHECK(game.boardLimit() == 25);
  CHECK_NEAR(game.urgency(), 25.0 - game.boardCount(), 1e-6);
  CHECK(game.moveCount() == 0);
  CHECK(game.terminalReason().empty());

  std::set<MoveKey> unique_moves;
  int submits = 0;
  for (int i = 0; i < game.actionCount(); ++i) {
    const Action& action = game.action(i);
    CHECK(game.isLegal(i));
    if (action.is_submit) {
      ++submits;
      CHECK(i == game.actionCount() - 1);  // SUBMIT is always last.
      continue;
    }
    CHECK(unique_moves.insert(action.move).second);
    CHECK(action.move[0] >= 0 && action.move[0] < 4);
    CHECK(action.move[1] >= 0 && action.move[1] < 4);
    CHECK(action.move[4] >= 0 && action.move[4] < 4);
    CHECK(action.move[5] >= 0 && action.move[5] < 4);
    CHECK(action.move != MoveKey{});
  }
  CHECK(submits <= 1);

  const StateKey key = game.stateKey();
  CHECK(!key.empty());

  auto clone = game.cloneState();
  CHECK(clone->stateKey() == key);
  CHECK(clone->actionCount() == game.actionCount());
  CHECK(clone->actions() == game.actions());
  CHECK(clone->nativeHash() == game.nativeHash());
  CHECK(clone->boardCount() == game.boardCount());
}

void testRecordSnapshotInitialConsistency() {
  Chess5DGameState game;
  const RecordSnapshot initial = game.recordSnapshot();

  // These are the native reference values returned by engine.game for
  // "Very Small - Open". Full show_fen retains unmoved flags while the board
  // cache intentionally strips them.
  CHECK(initial.sizeX == 4);
  CHECK(initial.sizeY == 4);
  CHECK(initial.fen == "[nbrk/3p*/P*3/KRBN:0:1:w]\n");
  CHECK(initial.boards.size() == 1);
  if (initial.boards.size() == 1) {
    CHECK(initial.boards[0].l == 0);
    CHECK(initial.boards[0].t == 1);
    CHECK(!initial.boards[0].c);
    CHECK(initial.boards[0].fen == "nbrk/3p/P3/KRBN");
  }
  CHECK(initial.presentT == 1);
  CHECK(!initial.presentC);
  CHECK(initial.mandatoryTimelines == std::vector<int>{0});
  CHECK(initial.optionalTimelines.empty());
  CHECK(initial.unplayableTimelines.empty());

  // Native coordinates are [x,y,t,l], with (0,0) at the lower-left FEN
  // square. In the reference position the rook at b1 can move to b2.
  const MoveKey rook_b1_b2{1, 0, 1, 0, 1, 1, 1, 0};
  int rook_move = -1;
  const std::set<std::pair<int, int>> white_sources{
      {0, 0}, {1, 0}, {2, 0}, {3, 0}, {0, 1}};
  for (int i = 0; i < game.actionCount(); ++i) {
    const Action& action = game.action(i);
    if (action.is_submit) {
      continue;
    }
    CHECK(action.move[2] == 1);
    CHECK(action.move[3] == 0);
    CHECK(white_sources.count({action.move[0], action.move[1]}) == 1);
    if (action.move == rook_b1_b2) {
      rook_move = i;
    }
  }
  CHECK(rook_move >= 0);
  if (rook_move < 0) {
    return;
  }

  game.applyAction(rook_move);
  const RecordSnapshot after_move = game.recordSnapshot();
  CHECK(after_move.boards.size() ==
        static_cast<std::size_t>(game.boardCount()));
  CHECK(after_move.boards.size() == 2);
  CHECK(after_move.fen.find("[nbrk/3p*/P*3/KRBN:0:1:w]") !=
        std::string::npos);
  for (const auto& board : after_move.boards) {
    CHECK(board.fen.find('*') == std::string::npos);
  }
}

void testCloneAndActionApplication() {
  Chess5DGameState game;
  const std::string original_key = game.stateKey();
  const int original_actions = game.actionCount();

  auto clone = game.cloneState();
  CHECK(clone->action(0) == game.action(0));
  clone->applyAction(0);

  CHECK(clone->moveCount() == 1);
  CHECK(clone->stateKey() != original_key);
  // A small step does not end the turn.
  CHECK(clone->currentPlayer() == 0);
  CHECK(game.stateKey() == original_key);
  CHECK(game.actionCount() == original_actions);
  CHECK(game.moveCount() == 0);

  // A clone of a clone stays independent.
  auto nested = clone->cloneState();
  CHECK(nested->stateKey() == clone->stateKey());
  CHECK(nested->nativeHash() == clone->nativeHash());
}

void testSubmitAdvancesTheTurn() {
  Chess5DGameState game;
  int submit = findSubmit(game);

  // At the start the player owes a move on the initial timeline, so SUBMIT is
  // normally unavailable. Play frontier moves until it becomes legal.
  int plies = 0;
  while (submit < 0 && plies < 20) {
    CHECK(game.actionCount() > 0);
    game.applyAction(0);
    ++plies;
    submit = findSubmit(game);
  }
  CHECK(plies < 20);
  CHECK(submit >= 0);
  CHECK(submit == game.actionCount() - 1);

  const int moves_before = game.moveCount();
  const int boards_before = game.boardCount();
  game.applyAction(submit);

  CHECK(game.moveCount() == moves_before + 1);
  CHECK(game.currentPlayer() == 1);
  CHECK(game.boardCount() >= boards_before);
  CHECK(!game.isTerminal() || game.terminalReason() == "material");
}

void testInitialMaterialTerminal() {
  GameConfig config;
  config.board_limit = 1;
  Chess5DGameState game(config);

  CHECK(game.boardCount() == 1);
  CHECK(game.isTerminal());
  CHECK(game.terminalReason() == "material");
  CHECK(game.actionCount() == 0);
  CHECK(game.moveCount() == 0);
  CHECK_NEAR(game.terminalUtility(), 0.0, 1e-6);

  auto clone = game.cloneState();
  CHECK(clone->isTerminal());
  CHECK(clone->terminalReason() == "material");
  CHECK(clone->actionCount() == 0);
  CHECK(clone->stateKey() == game.stateKey());
  CHECK_NEAR(clone->terminalUtility(), game.terminalUtility(), 1e-12);
}

void testNonSubmitMaterialTerminal() {
  struct Case {
    const char* fen;
    MoveKey move;
    int player;
  };
  const Case cases[] = {
      {"[3k/4/4/KR2:0:1:w]", {1, 0, 1, 0, 1, 1, 1, 0}, 0},
      {"[kr2/4/4/3K:0:1:b]", {1, 3, 1, 0, 1, 2, 1, 0}, 1},
  };

  for (const Case& test : cases) {
    GameConfig config;
    config.variant_pgn =
        std::string("[Board \"Custom\"]\n[Size \"4x4\"]\n") + test.fen +
        "\n";
    config.board_limit = 2;
    Chess5DGameState game(config);

    CHECK(!game.isTerminal());
    CHECK(game.boardCount() == 1);
    CHECK(game.currentPlayer() == test.player);
    const int action = findMove(game, test.move);
    CHECK(action >= 0);
    if (action < 0) {
      continue;
    }
    CHECK(!game.action(action).is_submit);

    game.applyAction(action);
    CHECK(game.moveCount() == 1);
    CHECK(game.boardCount() == 2);
    CHECK(game.currentPlayer() == test.player);
    CHECK(game.isTerminal());
    CHECK(game.terminalReason() == "material");
    CHECK(game.actionCount() == 0);
    CHECK_NEAR(game.terminalUtility(), std::tanh(2.0), 1e-6);

    auto clone = game.cloneState();
    CHECK(clone->isTerminal());
    CHECK(clone->terminalReason() == "material");
    CHECK(clone->actionCount() == 0);
    CHECK(clone->currentPlayer() == game.currentPlayer());
    CHECK(clone->stateKey() == game.stateKey());
    CHECK_NEAR(clone->terminalUtility(), game.terminalUtility(), 1e-12);
  }
}

void testCaptureKingPrecedesMaterialTerminal() {
  GameConfig config;
  config.variant_pgn =
      "[Board \"Custom\"]\n[Size \"4x4\"]\n"
      "[4/4/4/KR1k:0:1:w]\n";
  config.board_limit = 2;
  Chess5DGameState game(config);

  const MoveKey capture{1, 0, 1, 0, 3, 0, 1, 0};
  const int action = findMove(game, capture);
  CHECK(action >= 0);
  if (action < 0) {
    return;
  }

  game.applyAction(action);
  CHECK(game.boardCount() == 2);
  CHECK(game.isTerminal());
  CHECK(game.terminalReason() == "capture_king");
  CHECK_NEAR(game.terminalUtility(), 1.0, 1e-12);
}

void testEncoding() {
  Chess5DGameState game;
  const auto encoded = game.encodePosition();

  CHECK(encoded.num_boards == game.boardCount());
  CHECK(encoded.side_to_move == game.currentPlayer());
  CHECK(encoded.board_side == 4);
  CHECK(encoded.board_squares == 16);
  CHECK(encoded.piece_channels == 12);
  CHECK_NEAR(encoded.urgency, game.urgency(), 1e-6);
  CHECK(encoded.board_planes.size() ==
        static_cast<std::size_t>(encoded.num_boards) * 12 * 16);
  CHECK(encoded.l_coords.size() ==
        static_cast<std::size_t>(encoded.num_boards));
  CHECK(encoded.t_coords.size() ==
        static_cast<std::size_t>(encoded.num_boards));

  int ones = 0;
  for (float value : encoded.board_planes) {
    CHECK(value == 0.0f || value == 1.0f);
    if (value == 1.0f) {
      ++ones;
    }
  }
  CHECK(ones > 0);
  // Fewer pieces than squares per board.
  CHECK(ones <= encoded.num_boards * 16);
}

void testOnnxEvaluator() {
  OnnxEvaluatorConfig config;
  config.model_path = modelPath();
  OnnxEvaluator evaluator(config);
  CHECK(evaluator.config().board_side == 4);

  Chess5DGameState game;
  const Evaluation first = evaluator.evaluate(game);
  const Evaluation second = evaluator.evaluate(game);

  CHECK(first.policy.size() == static_cast<std::size_t>(game.actionCount()));
  CHECK(first.policy.size() == second.policy.size());
  double sum = 0.0;
  for (std::size_t i = 0; i < first.policy.size(); ++i) {
    CHECK(std::isfinite(first.policy[i]));
    CHECK(first.policy[i] >= 0.0f);
    CHECK_NEAR(first.policy[i], second.policy[i], 1e-6);
    sum += first.policy[i];
  }
  CHECK_NEAR(sum, 1.0, 1e-5);
  CHECK(std::isfinite(first.utility));
  CHECK(first.utility >= -1.0 && first.utility <= 1.0);
  CHECK_NEAR(first.utility, second.utility, 1e-6);

  bool threw = false;
  try {
    ForeignState foreign;
    (void)evaluator.evaluate(foreign);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

void testOnnxEvaluatorEnvironmentLifetime() {
  OnnxEvaluatorConfig config;
  config.model_path = modelPath();
  Chess5DGameState game;

  for (int iteration = 0; iteration < 3; ++iteration) {
    auto first = std::make_unique<OnnxEvaluator>(config);
    const Evaluation expected = first->evaluate(game);
    OnnxEvaluator second(config);
    first.reset();

    const Evaluation actual = second.evaluate(game);
    CHECK(actual.policy.size() == expected.policy.size());
    for (std::size_t i = 0;
         i < actual.policy.size() && i < expected.policy.size(); ++i) {
      CHECK_NEAR(actual.policy[i], expected.policy[i], 1e-6);
    }
    CHECK_NEAR(actual.utility, expected.utility, 1e-6);
  }
}

std::vector<std::unique_ptr<Chess5DGameState>> makeVaryingPositions(
    int steps) {
  std::vector<std::unique_ptr<Chess5DGameState>> positions;
  auto current = std::make_unique<Chess5DGameState>();
  for (int step = 0; step < steps; ++step) {
    positions.push_back(current->cloneState());
    if (current->isTerminal() || current->actionCount() <= 0) {
      break;
    }
    int action = 0;
    for (int candidate = 0; candidate < current->actionCount(); ++candidate) {
      if (!current->action(candidate).is_submit) {
        action = candidate;
        break;
      }
    }
    current->applyAction(action);
  }
  return positions;
}

void testBatchedOnnxEvaluatorMatchesSingle() {
  OnnxEvaluatorConfig config;
  config.model_path = modelPath();
  OnnxEvaluator single(config);
  suzero::chess5d::OnnxBatchBackend backend(config, 16);

  const auto positions = makeVaryingPositions(10);
  CHECK(positions.size() >= 3);

  std::vector<std::unique_ptr<Evaluation>> expected;
  expected.reserve(positions.size());
  std::vector<const IGameState*> states;
  states.reserve(positions.size());
  for (const auto& position : positions) {
    states.push_back(position.get());
    expected.push_back(std::make_unique<Evaluation>(single.evaluate(*position)));
  }

  std::vector<Evaluation> actual;
  backend.evaluateBatch(states, actual);
  CHECK(actual.size() == expected.size());

  for (std::size_t i = 0; i < actual.size() && i < expected.size(); ++i) {
    CHECK(actual[i].policy.size() == expected[i]->policy.size());
    if (actual[i].policy.size() != expected[i]->policy.size()) {
      continue;
    }
    for (std::size_t j = 0; j < actual[i].policy.size(); ++j) {
      CHECK_NEAR(actual[i].policy[j], expected[i]->policy[j], 1e-4);
    }
    CHECK_NEAR(actual[i].utility, expected[i]->utility, 1e-4);
  }

  // A batch of one must match the same position evaluated alone as well.
  std::vector<Evaluation> one;
  backend.evaluateBatch({states.front()}, one);
  CHECK(one.size() == 1);
  CHECK(one.front().policy.size() == expected.front()->policy.size());
  for (std::size_t i = 0; i < one.front().policy.size(); ++i) {
    CHECK_NEAR(one.front().policy[i], expected.front()->policy[i], 1e-4);
  }
}

void testBatchedSelfPlaySmoke() {
  OnnxEvaluatorConfig config;
  config.model_path = modelPath();

  BatchEvaluatorConfig batch_config;
  batch_config.maxBatchSize = 4;
  batch_config.numServerThreads = 1;
  auto evaluator =
      suzero::chess5d::makeBatchedOnnxEvaluator(config, batch_config);

  SelfPlayConfig self_play;
  self_play.numGameThreads = 2;
  self_play.maxGames = 2;
  self_play.maxMovesPerGame = 6;
  self_play.searchParams.maxPlayouts = 4;
  self_play.searchParams.maxVisits = 0;
  self_play.searchParams.chosenMoveTemperature = 0.0;
  self_play.seed = 3;

  SelfPlayRunner runner(
      self_play,
      []() { return std::make_shared<Chess5DGameState>(); }, evaluator);

  const std::vector<SelfPlayGame> games = runner.run();
  CHECK(games.size() == 2);
  CHECK(evaluator->stats().rows > 0);

  for (const SelfPlayGame& game : games) {
    CHECK(game.movesPlayed > 0);
    for (const SelfPlaySample& sample : game.samples) {
      CHECK(sample.state != nullptr);
      if (sample.state == nullptr) {
        continue;
      }
      CHECK(sample.policy.size() ==
            static_cast<std::size_t>(sample.state->actionCount()));
      double sum = 0.0;
      for (float probability : sample.policy) {
        CHECK(std::isfinite(probability));
        CHECK(probability >= 0.0f);
        sum += probability;
      }
      CHECK_NEAR(sum, 1.0, 1e-4);
    }
  }
}

void testSelfPlayBridgeSmoke() {
  SelfPlayBridgeConfig config;
  config.evaluator.model_path = modelPath();
  config.numGames = 2;
  config.numGameThreads = 2;
  config.numSimulations = 4;
  config.fastSimulations = 2;
  config.playoutCapRandomization = false;
  config.minBoardLimit = 6;
  config.maxBoardLimit = 8;
  config.maxBatchSize = 4;
  config.seed = 11;

  std::atomic<int> finished{0};
  SelfPlayStats stats;
  const std::vector<BridgeGame> games = runSelfPlay(
      config,
      [&finished](int completed, const BridgeGame&) {
        finished.store(completed);
      },
      &stats);

  CHECK(games.size() == 2);
  CHECK(finished.load() == 2);

  // Batching counters: every request is served exactly once, and the observed
  // batch never exceeds either the configured cap or the number of rows.
  CHECK(stats.numServerThreads >= 1);
  CHECK(stats.maxBatchSize == 4);
  CHECK(stats.requests > 0);
  CHECK(stats.requests == stats.rows);
  CHECK(stats.batches > 0);
  CHECK(stats.batches <= stats.requests);
  CHECK(stats.maxObservedBatch > 0);
  CHECK(stats.maxObservedBatch <= stats.rows);
  CHECK(stats.maxObservedBatch <=
        static_cast<std::uint64_t>(stats.maxBatchSize));
  CHECK(stats.totalSeconds > 0.0);
  CHECK(stats.setupSeconds >= 0.0);
  CHECK(stats.setupSeconds <= stats.totalSeconds);
  for (const BridgeGame& game : games) {
    CHECK(game.index >= 0 && game.index < 2);
    CHECK(game.movesPlayed > 0);
    CHECK(game.boardLimit >= 6 && game.boardLimit <= 8);
    CHECK(game.boardCount >= 0);
    CHECK(game.semimoves >= 0);
    CHECK(game.turns >= 0);
    CHECK(game.finalPlayer == 0 || game.finalPlayer == 1);
    CHECK(std::fabs(game.whiteOutcome) <= 1.0);
    CHECK(!game.terminalReason.empty());
    if (game.terminalReason == "material") {
      CHECK(game.boardCount >= game.boardLimit);
    }
    CHECK(game.moves.empty());
    CHECK(game.finalFen.empty());
    for (const BridgeSample& sample : game.samples) {
      CHECK(!sample.actions.empty());
      CHECK(sample.policy.size() == sample.actions.size());
      CHECK(sample.encoded.num_boards ==
            static_cast<int>(sample.encoded.l_coords.size()));
      CHECK(sample.encoded.num_boards ==
            static_cast<int>(sample.encoded.t_coords.size()));
      CHECK(sample.player == 0 || sample.player == 1);
      CHECK(std::fabs(sample.value) <= 1.0f);
      const double expected_value =
          sample.player == 0 ? game.whiteOutcome : -game.whiteOutcome;
      CHECK_NEAR(sample.value, expected_value, 1e-6);
      double sum = 0.0;
      for (float probability : sample.policy) {
        CHECK(std::isfinite(probability));
        CHECK(probability >= 0.0f);
        sum += probability;
      }
      CHECK_NEAR(sum, 1.0, 1e-4);
    }
  }
}

void testSelfPlayBridgeRecordingAndReplay() {
  SelfPlayBridgeConfig config;
  config.evaluator.model_path = modelPath();
  config.numGames = 1;
  config.numGameThreads = 1;
  config.numSimulations = 2;
  config.fastSimulations = 1;
  config.playoutCapRandomization = true;
  config.fullSearchProbability = 0.0;
  config.minBoardLimit = 20;
  config.maxBoardLimit = 20;
  config.maxBatchSize = 2;
  config.recordGames = true;
  config.seed = 29;

  const std::vector<BridgeGame> games = runSelfPlay(config);
  CHECK(games.size() == 1);
  if (games.size() != 1) {
    return;
  }
  const BridgeGame& recorded = games.front();
  CHECK(!recorded.moves.empty());
  CHECK(recorded.movesPlayed == static_cast<int>(recorded.moves.size()));
  CHECK(!recorded.finalFen.empty());
  CHECK(!recorded.terminalReason.empty());
  if (recorded.terminalReason == "material") {
    CHECK(recorded.boardCount >= recorded.boardLimit);
  }
  // No policy samples are retained when every decision uses fast search.
  CHECK(recorded.samples.empty());

  GameConfig replay_config = config.game;
  replay_config.board_limit = recorded.boardLimit;
  replay_config.material_scale = config.materialScale;
  Chess5DGameState replay(replay_config);

  int submit_count = 0;
  int forced_submit_count = 0;
  int move_count = 0;
  for (const auto& recorded_move : recorded.moves) {
    CHECK(recorded_move.player == replay.currentPlayer());
    if (recorded_move.action.is_submit) {
      ++submit_count;
      if (replay.actionCount() == 1 && replay.action(0).is_submit) {
        ++forced_submit_count;
      }
    } else {
      ++move_count;
    }

    int action_id = -1;
    for (int candidate = 0; candidate < replay.actionCount(); ++candidate) {
      if (replay.action(candidate) == recorded_move.action) {
        action_id = candidate;
        break;
      }
    }
    CHECK(action_id >= 0);
    if (action_id < 0) {
      return;
    }
    replay.applyAction(action_id);
  }

  CHECK(submit_count > 0);
  CHECK(forced_submit_count > 0);
  CHECK(move_count == recorded.semimoves);
  const RecordSnapshot replayed = replay.recordSnapshot();
  CHECK(replayed.fen == recorded.finalFen);
  CHECK(replay.isTerminal());
  CHECK(replay.terminalReason() == recorded.terminalReason);
  CHECK(replay.currentPlayer() == recorded.finalPlayer);
  CHECK(static_cast<int>(replayed.boards.size()) == recorded.boardCount);
}

AgentConfig makeAgentConfig(std::int64_t playouts) {
  AgentConfig config;
  config.model_path = modelPath();
  config.search = suzero::chess5d::DefaultSearchParams();
  config.search.maxPlayouts = playouts;
  config.seed = 7;
  return config;
}

void testAgentSearchAndTreeReuse() {
  AgentConfig config = makeAgentConfig(8);
  Chess5DAgent agent(config);

  CHECK(agent.state().stateKey() == agent.search().rootState().stateKey());

  const Decision decision = agent.selectAction(0.0);
  CHECK(!decision.forced);
  CHECK(decision.actionId >= 0);
  CHECK(decision.actionId < agent.state().actionCount());
  CHECK(decision.action == agent.state().action(decision.actionId));
  CHECK(decision.searchResult.playouts == 8);
  CHECK(decision.searchResult.rootVisits == 9);
  CHECK(decision.searchResult.bestAction != kInvalidAction);
  CHECK(decision.searchResult.actions.size() >= 2);

  const StateKey before = agent.state().stateKey();
  agent.applyAction(decision.actionId);
  CHECK(agent.state().stateKey() != before);
  CHECK(agent.state().moveCount() == 1);
  if (!agent.state().isTerminal()) {
    CHECK(agent.search().rootState().stateKey() == agent.state().stateKey());
  }

  const Decision next = agent.selectAction(0.0);
  CHECK(next.actionId >= 0);
  CHECK(next.actionId < agent.state().actionCount());
}

void testAgentShortGame() {
  AgentConfig config = makeAgentConfig(2);
  Chess5DAgent agent(config);

  int steps = 0;
  while (!agent.state().isTerminal() && steps < 80) {
    const Decision decision = agent.selectAction(0.0);
    if (decision.forced) {
      CHECK(agent.state().actionCount() == 1);
      CHECK(decision.action.is_submit);
    }
    agent.applyAction(decision.actionId);
    ++steps;
    if (!agent.state().isTerminal()) {
      CHECK(agent.search().rootState().stateKey() == agent.state().stateKey());
    }
  }

  CHECK(steps > 0);
  CHECK(agent.state().moveCount() == steps);
  if (agent.state().isTerminal()) {
    CHECK(!agent.state().terminalReason().empty());
    CHECK(std::isfinite(agent.state().terminalUtility()));
  }
}

void testDefaultSearchParameters() {
  AgentConfig config;
  config.model_path = modelPath();
  // DefaultSearchParams: 200 playouts, cpuct 2.0.
  CHECK(config.search.maxPlayouts == 200);
  CHECK_NEAR(config.search.cpuctExploration, 2.0, 1e-12);
  CHECK_NEAR(config.temperature, 1.0, 1e-12);
  CHECK_NEAR(config.temperature_final, 0.1, 1e-12);
  CHECK(config.temperature_threshold == 30);

  Chess5DAgent agent(config);
  const Decision decision = agent.selectAction(0.0);
  CHECK(!decision.forced);
  CHECK(decision.searchResult.playouts == 200);
  CHECK(decision.searchResult.rootVisits == 201);
  CHECK(decision.searchResult.bestAction != kInvalidAction);
}

/// Machine-readable initial-position evaluation used to cross-check the
/// encoding against the 5dc_ml reference encoder/model.
int runProbe(int plies) {
  OnnxEvaluatorConfig config;
  config.model_path = modelPath();
  OnnxEvaluator evaluator(config);
  Chess5DGameState game;
  for (int i = 0; i < plies && !game.isTerminal(); ++i) {
    game.applyAction(0);
  }
  const Evaluation evaluation = evaluator.evaluate(game);

  std::cout << std::setprecision(9);
  std::cout << "count " << game.actionCount() << "\n";
  std::cout << "utility " << evaluation.utility << "\n";
  std::cout << "urgency " << game.urgency() << "\n";
  for (int i = 0; i < game.actionCount(); ++i) {
    const Action& action = game.action(i);
    std::cout << "action " << i;
    if (action.is_submit) {
      std::cout << " submit";
    } else {
      for (const auto coordinate : action.move) {
        std::cout << ' ' << coordinate;
      }
    }
    std::cout << ' ' << evaluation.policy[static_cast<std::size_t>(i)]
              << "\n";
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--probe") {
    const int plies = argc > 2 ? std::atoi(argv[2]) : 0;
    return runProbe(plies);
  }

  try {
    testInitialState();
    testRecordSnapshotInitialConsistency();
    testCloneAndActionApplication();
    testSubmitAdvancesTheTurn();
    testInitialMaterialTerminal();
    testNonSubmitMaterialTerminal();
    testCaptureKingPrecedesMaterialTerminal();
    testEncoding();
    testOnnxEvaluator();
    testOnnxEvaluatorEnvironmentLifetime();
    testBatchedOnnxEvaluatorMatchesSingle();
    testBatchedSelfPlaySmoke();
    testSelfPlayBridgeSmoke();
    testSelfPlayBridgeRecordingAndReplay();
    testAgentSearchAndTreeReuse();
    testAgentShortGame();
    testDefaultSearchParameters();
  } catch (const std::exception& error) {
    std::cerr << "UNCAUGHT EXCEPTION: " << error.what() << "\n";
    return 1;
  }

  if (g_failures != 0) {
    std::cerr << g_failures << " of " << g_checks << " checks failed\n";
    return 1;
  }
  std::cout << "All " << g_checks << " 5D chess checks passed\n";
  return 0;
}
