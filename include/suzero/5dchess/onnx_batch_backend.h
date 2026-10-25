// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-02 contributors
//
// IBatchBackend implementation for the smallstep_movekey_v1 ONNX model.
//
// The backend accepts a heterogeneous batch: each position may have a
// different number of boards and candidate actions. Positions are padded to
// the batch maxima; `used_board_counts` masks padded boards inside the model,
// and padded action logits are ignored when each result is assembled.

#pragma once

#include <memory>
#include <vector>

#include "suzero/5dchess/onnx_evaluator.h"
#include "suzero/mcts/batch_evaluator.h"
#include "suzero/mcts/evaluator.h"
#include "suzero/mcts/game_state.h"

namespace suzero::chess5d {

/// One ONNX session devoted to a single server thread.
///
/// `BatchedEvaluator` creates one of these per server thread, so this class
/// does not need to be thread-safe with itself.
class OnnxBatchBackend final : public mcts::IBatchBackend {
 public:
  /// `max_batch_size` must be positive; it is the memory/concurrency cap this
  /// backend advertises to BatchedEvaluator.
  OnnxBatchBackend(OnnxEvaluatorConfig config, int max_batch_size);
  ~OnnxBatchBackend() override;

  OnnxBatchBackend(const OnnxBatchBackend&) = delete;
  OnnxBatchBackend& operator=(const OnnxBatchBackend&) = delete;

  int maxBatchSize() const override;

  /// Every state must be a Chess5DGameState. Results are in input order.
  void evaluateBatch(const std::vector<const mcts::IGameState*>& states,
                     std::vector<mcts::Evaluation>& results) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Convenience constructor for the intended pairing:
///
///   auto evaluator = makeBatchedOnnxEvaluator(model_config, batch_config);
///
/// It creates one ONNX session per configured server thread. Each session is
/// independent, so `numServerThreads > 1` costs one copy of the model per
/// thread.
std::shared_ptr<mcts::BatchedEvaluator> makeBatchedOnnxEvaluator(
    OnnxEvaluatorConfig config,
    mcts::BatchEvaluatorConfig batch_config = {});

}  // namespace suzero::chess5d
