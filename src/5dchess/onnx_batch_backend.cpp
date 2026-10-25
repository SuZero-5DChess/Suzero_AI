// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-02 contributors

#include "suzero/5dchess/onnx_batch_backend.h"

#include <stdexcept>
#include <utility>

#include "onnx_model.h"

namespace suzero::chess5d {

struct OnnxBatchBackend::Impl {
  static int requirePositiveMax(int max_batch_size) {
    if (max_batch_size <= 0) {
      throw std::invalid_argument(
          "OnnxBatchBackend: max_batch_size must be positive");
    }
    return max_batch_size;
  }

  Impl(OnnxEvaluatorConfig config, int max_batch_size)
      : config(std::move(config)),
        max_batch_size(requirePositiveMax(max_batch_size)),
        model(this->config) {}

  OnnxEvaluatorConfig config;
  int max_batch_size;
  internal::OnnxModel model;
};

OnnxBatchBackend::OnnxBatchBackend(OnnxEvaluatorConfig config,
                                   int max_batch_size)
    : impl_(std::make_unique<Impl>(std::move(config), max_batch_size)) {}

OnnxBatchBackend::~OnnxBatchBackend() = default;

int OnnxBatchBackend::maxBatchSize() const {
  return impl_->max_batch_size;
}

void OnnxBatchBackend::evaluateBatch(
    const std::vector<const mcts::IGameState*>& states,
    std::vector<mcts::Evaluation>& results) {
  if (states.size() > static_cast<std::size_t>(impl_->max_batch_size)) {
    throw std::invalid_argument(
        "OnnxBatchBackend: batch exceeds maxBatchSize");
  }

  std::vector<const IChess5DModelInput*> inputs;
  inputs.reserve(states.size());
  for (const mcts::IGameState* state : states) {
    const auto* input = dynamic_cast<const IChess5DModelInput*>(state);
    if (input == nullptr) {
      throw std::invalid_argument(
          "OnnxBatchBackend requires IChess5DModelInput");
    }
    inputs.push_back(input);
  }

  results = impl_->model.evaluateBatch(inputs);
}

std::shared_ptr<mcts::BatchedEvaluator> makeBatchedOnnxEvaluator(
    OnnxEvaluatorConfig config, mcts::BatchEvaluatorConfig batch_config) {
  const int max_batch_size = batch_config.maxBatchSize;
  return std::make_shared<mcts::BatchedEvaluator>(
      batch_config,
      [config = std::move(config), max_batch_size]() {
        return std::make_unique<OnnxBatchBackend>(config, max_batch_size);
      });
}

}  // namespace suzero::chess5d
