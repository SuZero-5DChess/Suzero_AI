// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-02 contributors
//
// Private implementation unit shared by OnnxEvaluator (one position per call)
// and OnnxBatchBackend (many positions per device call). Keeping the protocol,
// metadata validation and tensor layout in one place prevents the two paths
// from drifting apart.

#pragma once

#include <memory>
#include <vector>

#include "suzero/5dchess/game_state.h"
#include "suzero/5dchess/onnx_evaluator.h"
#include "suzero/mcts/evaluator.h"

namespace suzero::chess5d::internal {

class OnnxModel {
 public:
  explicit OnnxModel(OnnxEvaluatorConfig config);
  ~OnnxModel();

  OnnxModel(const OnnxModel&) = delete;
  OnnxModel& operator=(const OnnxModel&) = delete;

  const OnnxEvaluatorConfig& config() const;

  /// Evaluates one position with a batch of one.
  mcts::Evaluation evaluate(const IChess5DModelInput& input);

  /// Evaluates positions with one device call.
  ///
  /// Positions may have different board counts and candidate counts; they are
  /// padded to the batch maxima and masked through `used_board_counts` and by
  /// ignoring the padded action logits. Positions whose only candidate is
  /// SUBMIT do not enter the device call and receive a forced policy.
  std::vector<mcts::Evaluation> evaluateBatch(
      const std::vector<const IChess5DModelInput*>& model_inputs);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace suzero::chess5d::internal
