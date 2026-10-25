// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-5DChess contributors

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "suzero/5dchess/match.h"
#include "suzero/5dchess/match_game.h"
#include "suzero/5dchess/onnx_batch_backend.h"

#ifndef SUZERO_5DCHESS_MODEL_PATH
#error "SUZERO_5DCHESS_MODEL_PATH must be defined by CMake"
#endif

namespace {

using namespace suzero::mcts;
using namespace suzero::chess5d;

int checks = 0;
int failures = 0;

#define CHECK(condition)                                                     \
  do {                                                                       \
    ++checks;                                                                \
    if (!(condition)) {                                                      \
      ++failures;                                                            \
      std::cerr << "FAIL " << __LINE__ << ": " << #condition << '\n';         \
    }                                                                        \
  } while (0)

/// A second concrete adapter: its evaluator must see this object, not the
/// contained identity adapter. Cloning must preserve that boundary too.
class OtherEncodingGame final : public IMatchGame,
                                public IChess5DModelInput {
 public:
  explicit OtherEncodingGame(GameConfig config)
      : inner_(std::make_unique<Chess5DMatchGame>(config, true)) {}
  std::unique_ptr<IGameState> clone() const override {
    return std::unique_ptr<OtherEncodingGame>(new OtherEncodingGame(inner_->clone()));
  }
  int actionCount() const override { return inner_->actionCount(); }
  bool isLegal(ActionId a) const override { return inner_->isLegal(a); }
  void applyAction(ActionId a) override { inner_->applyAction(a); }
  PlayerId currentPlayer() const override { return inner_->currentPlayer(); }
  bool isTerminal() const override { return inner_->isTerminal(); }
  double terminalUtility() const override { return inner_->terminalUtility(); }
  StateKey stateKey() const override { return "other:" + inner_->stateKey(); }
  int moveCount() const override { return inner_->moveCount(); }
  std::string rulesKey() const override { return inner_->rulesKey(); }
  StateKey canonicalStateKey() const override { return inner_->canonicalStateKey(); }
  MatchAction toMatchAction(ActionId a) const override { return inner_->toMatchAction(a); }
  ActionId fromMatchAction(const MatchAction& a) const override { return inner_->fromMatchAction(a); }
  std::string terminalReason() const override { return inner_->terminalReason(); }
  const std::vector<Action>& actions() const override { return inner_->actions(); }
  EncodedPosition encodePosition() const override { return inner_->encodePosition(); }

 private:
  explicit OtherEncodingGame(std::unique_ptr<IGameState> state) {
    auto* typed = dynamic_cast<Chess5DMatchGame*>(state.get());
    if (!typed) throw std::logic_error("match clone lost its type");
    state.release();
    inner_.reset(typed);
  }
  std::unique_ptr<Chess5DMatchGame> inner_;
};

OnnxEvaluatorConfig modelConfig() {
  OnnxEvaluatorConfig config;
  config.model_path = SUZERO_5DCHESS_MODEL_PATH;
  return config;
}

template <typename Game>
class TypedOnnxEvaluator final : public IEvaluator {
 public:
  TypedOnnxEvaluator() : evaluator_(modelConfig()) {}
  Evaluation evaluate(const IGameState& state) override {
    if (!dynamic_cast<const Game*>(&state)) {
      throw std::logic_error("wrong adapter reached a model");
    }
    ++calls;
    return evaluator_.evaluate(state);
  }
  int calls = 0;

 private:
  OnnxEvaluator evaluator_;
};

void testMappingAndSubmit() {
  GameConfig config;
  Chess5DMatchGame a(config);
  config.line_shift += 7;
  OtherEncodingGame b(config);
  CHECK(a.rulesKey() == b.rulesKey());
  CHECK(a.canonicalStateKey() == b.canonicalStateKey());
  CHECK(a.stateKey() != b.stateKey());
  const auto input_a = a.encodePosition();
  const auto input_b = b.encodePosition();
  CHECK(input_a.board_planes == input_b.board_planes);
  CHECK(input_b.l_coords[0] == input_a.l_coords[0] + 7);
  CHECK(a.actionCount() > 1);
  const int move_count = a.actionCount() - (a.actions().back().is_submit ? 1 : 0);
  for (int i = 0; i < a.actionCount(); ++i) {
    const auto logical = a.toMatchAction(i);
    const auto j = b.fromMatchAction(logical);
    CHECK(j == (a.actions()[i].is_submit ? i : move_count - 1 - i));
    CHECK(b.toMatchAction(j) == logical);
    CHECK(a.fromMatchAction(logical) == i);
  }
  CHECK(b.fromMatchAction("bad-action") == kInvalidAction);
  const auto original = a.canonicalStateKey();
  auto clone = a.clone();
  CHECK(dynamic_cast<IMatchGame*>(clone.get()) != nullptr);
  CHECK(dynamic_cast<IChess5DModelInput*>(clone.get()) != nullptr);
  const auto move = a.toMatchAction(0);
  a.applyAction(0);
  b.applyAction(b.fromMatchAction(move));
  CHECK(a.currentPlayer() == 0 && b.currentPlayer() == 0);
  CHECK(a.canonicalStateKey() == b.canonicalStateKey());
  CHECK(dynamic_cast<IMatchGame&>(*clone).canonicalStateKey() == original);
  const int submit = a.actionCount() - 1;
  CHECK(a.actions()[submit].is_submit);
  const auto logical_submit = a.toMatchAction(submit);
  a.applyAction(submit);
  b.applyAction(b.fromMatchAction(logical_submit));
  CHECK(a.currentPlayer() == 1 && b.currentPlayer() == 1);
  CHECK(a.canonicalStateKey() == b.canonicalStateKey());
  CHECK(a.moveCount() == 2 && b.moveCount() == 2);

  config.board_limit += 1;
  Chess5DMatchGame incompatible(config);
  CHECK(a.rulesKey() != incompatible.rulesKey());
}

void testOnnxInputInterface() {
  GameConfig config;
  Chess5DGameState original(config);
  Chess5DMatchGame identity(config);
  OtherEncodingGame reverse(config);
  OnnxEvaluator evaluator(modelConfig());
  const auto raw = evaluator.evaluate(original);
  const auto normal = evaluator.evaluate(identity);
  const auto reversed = evaluator.evaluate(reverse);
  CHECK(std::abs(raw.utility - normal.utility) < 1e-6);
  CHECK(raw.policy == normal.policy);
  CHECK(std::abs(normal.utility - reversed.utility) < 1e-5);
  for (int i = 0; i < identity.actionCount(); ++i) {
    const int j = reverse.fromMatchAction(identity.toMatchAction(i));
    CHECK(std::abs(normal.policy[i] - reversed.policy[j]) < 1e-5);
  }
  OnnxBatchBackend backend(modelConfig(), 2);
  std::vector<Evaluation> batch;
  backend.evaluateBatch({&identity, &reverse}, batch);
  CHECK(batch.size() == 2);
  for (int i = 0; i < identity.actionCount(); ++i) {
    CHECK(std::abs(batch[0].policy[i] - normal.policy[i]) < 1e-5);
    CHECK(std::abs(batch[1].policy[i] - reversed.policy[i]) < 1e-5);
  }
}

void testHeterogeneousTerminalSeries() {
  GameConfig config;
  config.board_limit = 2;
  auto eval_a = std::make_shared<TypedOnnxEvaluator<Chess5DMatchGame>>();
  auto eval_b = std::make_shared<TypedOnnxEvaluator<OtherEncodingGame>>();
  std::array<MatchParticipant, 2> bots;
  bots[0].name = "normal";
  bots[0].player = 0;
  bots[0].stateFactory = [config] { return std::make_shared<Chess5DMatchGame>(config); };
  bots[0].evaluator = eval_a;
  bots[0].searchParams.maxPlayouts = 3;
  bots[0].searchParams.maxDepth = 8;
  config.line_shift += 2;
  bots[1].name = "other";
  bots[1].player = 1;
  bots[1].stateFactory = [config] { return std::make_shared<OtherEncodingGame>(config); };
  bots[1].evaluator = eval_b;
  bots[1].searchParams.maxPlayouts = 5;
  bots[1].searchParams.maxDepth = 10;
  MatchSeriesConfig series;
  series.game.maxMoves = 8;
  const auto result = runMatchSeries(series, bots);
  CHECK(result.games.size() == 2);
  CHECK(eval_a->calls > 0 && eval_b->calls > 0);
  for (std::size_t i = 0; i < result.games.size(); ++i) {
    const auto& game = result.games[i];
    CHECK(game.terminal && !game.truncated);
    CHECK(game.terminalReason == "material");
    CHECK(game.finalUtility.has_value());
    CHECK(game.participantUtilities[0].has_value());
    CHECK(game.participantUtilities[1].has_value());
    CHECK(*game.participantUtilities[0] == -*game.participantUtilities[1]);
    CHECK(game.moves.size() == 1);
    CHECK(game.moves[0].participantIndex == static_cast<int>(i));
    CHECK(game.moves[0].localActions[0] != game.moves[0].localActions[1]);
    auto replay = bots[0].stateFactory();
    for (const auto& move : game.moves) {
      replay->applyAction(replay->fromMatchAction(move.action));
    }
    CHECK(replay->isTerminal());
    CHECK(replay->terminalUtility() == *game.finalUtility);
  }
  CHECK(result.totals[0].truncated == 0);
  CHECK(result.totals[0].wins + result.totals[0].losses + result.totals[0].draws == 2);

  GameConfig truncation_config;
  bots[0].stateFactory = [truncation_config] {
    return std::make_shared<Chess5DMatchGame>(truncation_config);
  };
  truncation_config.line_shift += 2;
  bots[1].stateFactory = [truncation_config] {
    return std::make_shared<OtherEncodingGame>(truncation_config);
  };
  series.game.maxMoves = 1;
  const auto truncated = runMatchSeries(series, bots);
  CHECK(truncated.totals[0].truncated == 2);
  CHECK(truncated.totals[0].draws == 0);
  for (const auto& game : truncated.games) {
    CHECK(game.truncated && !game.terminal);
    CHECK(!game.finalUtility.has_value());
  }
}

void testOnnxBridge() {
  std::array<OnnxMatchParticipant, 2> bots;
  for (auto& bot : bots) {
    bot.evaluator = modelConfig();
    bot.game.board_limit = 2;
    bot.search.maxPlayouts = 2;
    bot.search.maxDepth = 8;
  }
  bots[1].reverseMoveOrder = true;
  MatchSeriesConfig config;
  config.game.maxMoves = 4;
  int callbacks = 0;
  const auto result = runOnnxMatchSeries(config, bots,
      [&callbacks](int index, const MatchResult& game) {
        CHECK(index == callbacks++);
        CHECK(game.terminal);
      });
  CHECK(callbacks == 2);
  CHECK(result.games.size() == 2);
}

}  // namespace

int main() {
  try {
    testMappingAndSubmit();
    testOnnxInputInterface();
    testHeterogeneousTerminalSeries();
    testOnnxBridge();
  } catch (const std::exception& error) {
    std::cerr << "Unexpected exception: " << error.what() << '\n';
    return 1;
  }
  std::cout << checks << " checks, " << failures << " failures\n";
  return failures == 0 ? 0 : 1;
}
