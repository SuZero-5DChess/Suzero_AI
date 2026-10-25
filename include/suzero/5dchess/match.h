// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-5DChess contributors

#pragma once

#include <array>
#include <string>

#include "suzero/5dchess/game_state.h"
#include "suzero/5dchess/onnx_evaluator.h"
#include "suzero/mcts/match.h"

namespace suzero::chess5d {

/// Each bot owns its input encoding, ONNX session and search configuration.
/// Only the player assignments (white=0, black=1) are exchanged in a pair.
struct OnnxMatchParticipant {
  std::string name;
  GameConfig game;
  OnnxEvaluatorConfig evaluator;
  mcts::SearchParams search;
  bool reverseMoveOrder = false;
};

/// Runs a color-balanced series with config.numGameThreads concurrent games,
/// without touching self-play or training state. Parallel games use a separate
/// batching evaluator/session per model. Both models must speak
/// smallstep_movekey_v1; arbitrary protocols can use MatchRunner directly.
mcts::MatchSeriesResult runOnnxMatchSeries(
    const mcts::MatchSeriesConfig& config,
    const std::array<OnnxMatchParticipant, 2>& participants,
    const mcts::MatchGameCallback& onGameFinished = {});

}  // namespace suzero::chess5d
