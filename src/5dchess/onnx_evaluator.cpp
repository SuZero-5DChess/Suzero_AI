// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-02 contributors

#include "suzero/5dchess/onnx_evaluator.h"

#include <stdexcept>
#include <utility>

#include "onnx_model.h"

namespace suzero::chess5d {

struct OnnxEvaluator::Impl {
  explicit Impl(OnnxEvaluatorConfig config)
      : model(std::move(config)) {}

  internal::OnnxModel model;
};

OnnxEvaluator::OnnxEvaluator(OnnxEvaluatorConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

OnnxEvaluator::~OnnxEvaluator() = default;

const OnnxEvaluatorConfig& OnnxEvaluator::config() const {
  return impl_->model.config();
}

mcts::Evaluation OnnxEvaluator::evaluate(const mcts::IGameState& state) {
  const auto* input = dynamic_cast<const IChess5DModelInput*>(&state);
  if (input == nullptr) {
    throw std::invalid_argument(
        "OnnxEvaluator requires IChess5DModelInput");
  }
  return impl_->model.evaluate(*input);
}

}  // namespace suzero::chess5d
