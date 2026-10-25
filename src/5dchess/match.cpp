// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-02 contributors

#include "suzero/5dchess/match.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>

#include "suzero/5dchess/match_game.h"
#include "suzero/5dchess/onnx_batch_backend.h"

namespace suzero::chess5d {

mcts::MatchSeriesResult runOnnxMatchSeries(
    const mcts::MatchSeriesConfig& config,
    const std::array<OnnxMatchParticipant, 2>& participants,
    const mcts::MatchGameCallback& onGameFinished) {
  if (config.pairs <= 0 ||
      config.pairs > std::numeric_limits<int>::max() / 2 ||
      config.game.maxMoves < 0) {
    throw std::invalid_argument("ONNX match requires positive pairs and non-negative maxMoves");
  }

  if (config.numGameThreads <= 0) {
    throw std::invalid_argument("ONNX match requires positive numGameThreads");
  }

  std::array<mcts::MatchParticipant, 2> bots;
  std::array<std::shared_ptr<Chess5DMatchGame>, 2> initial;
  for (std::size_t i = 0; i < bots.size(); ++i) {
    const auto& source = participants[i];
    if (source.game.board_side != source.evaluator.board_side ||
        source.game.board_squares != source.evaluator.board_squares ||
        source.game.piece_channels != source.evaluator.piece_channels) {
      throw std::invalid_argument("ONNX match game/model geometry mismatch");
    }
    initial[i] = std::make_shared<Chess5DMatchGame>(
        source.game, source.reverseMoveOrder);
    bots[i].name = source.name;
    bots[i].player = static_cast<mcts::PlayerId>(i);
    bots[i].stateFactory = [game = source.game,
                            reverse = source.reverseMoveOrder]() {
      return std::make_shared<Chess5DMatchGame>(game, reverse);
    };
    bots[i].searchParams = source.search;
  }
  // Reject incompatible rules before loading potentially large models.
  if (initial[0]->rulesKey() != initial[1]->rulesKey() ||
      initial[0]->canonicalStateKey() != initial[1]->canonicalStateKey()) {
    throw std::invalid_argument("ONNX match participants have incompatible games");
  }
  for (std::size_t i = 0; i < bots.size(); ++i) {
    if (config.numGameThreads == 1) {
      bots[i].evaluator = std::make_shared<OnnxEvaluator>(participants[i].evaluator);
    } else {
      // Keep model queues/sessions separate; never batch different weights or
      // encodings together. Parallel games feed one server per participant.
      mcts::BatchEvaluatorConfig batch;
      batch.maxBatchSize = std::min({config.numGameThreads, config.pairs * 2, 128});
      bots[i].evaluator = makeBatchedOnnxEvaluator(participants[i].evaluator, batch);
    }
  }
  return mcts::runMatchSeries(config, bots, onGameFinished);
}

}  // namespace suzero::chess5d
