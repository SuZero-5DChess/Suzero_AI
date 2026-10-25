// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-5DChess contributors
//
// IEvaluator implementation backed by the 5dc_ml smallstep_movekey_v1 ONNX
// model. The ONNX Runtime C++ API is hidden behind a pimpl so the public header
// stays dependency-free.

#pragma once

#include <memory>
#include <string>

#include "suzero/5dchess/game_state.h"
#include "suzero/mcts/evaluator.h"

namespace suzero::chess5d {

/// Configuration of the ONNX policy/value evaluator.
///
/// Defaults describe the exported `very_small` (4x4, 12 channels) model. The
/// sidecar metadata written by the exporter is validated against these values
/// and against the session's own input/output signature.
struct OnnxEvaluatorConfig {
  /// Path to the .onnx model. Required.
  std::string model_path;

  /// Path to the exporter's JSON sidecar. Empty means `model_path + ".json"`.
  std::string metadata_path;

  /// Execution provider: "cpu" or "cuda".
  ///
  /// "cuda" appends the CUDA execution provider and fails loudly when it is
  /// unavailable (CPU-only ONNX Runtime build, missing cuDNN/CUDA libraries,
  /// or no visible GPU). It never silently falls back to the CPU provider.
  std::string provider = "cpu";

  /// CUDA device ordinal used when `provider == "cuda"`.
  int cuda_device_id = 0;

  /// Intra-op threads for the ONNX Runtime session.
  int intra_op_threads = 1;

  // Board geometry expected by the model.
  int board_side = 4;
  int board_squares = 16;
  int piece_channels = 12;
};

/// Policy/value evaluator for IGameState objects that also implement
/// IChess5DModelInput.
///
/// Protocol (`smallstep_movekey_v1`):
///   inputs : board_planes      float32 [B, N, C, S]
///            l_coords          int64   [B, N]
///            t_coords          int64   [B, N]
///            used_board_counts int64   [B]
///            urgency           float32 [B]
///            side_to_move      int64   [B]   (0 = white, 1 = black)
///            actions           int64   [B, q, 8]  non-submit MoveKeys
///   outputs: value              float32 [B, 1]  side-to-move utility
///            submit_logit       float32 [B, 1]
///            other_logits       float32 [B, q]
///
/// The returned policy has one entry per candidate action, in
/// `IChess5DModelInput::actions()` order. The protocol requires non-submit
/// moves first and an optional SUBMIT action last.
///
/// A position whose only candidate is SUBMIT (q == 0) is not sent to the
/// network: it returns a degenerate policy of size 1 and utility 0, matching
/// the protocol's requirement that inference requests contain q > 0.
///
/// Evaluation is synchronous. Concurrent calls on one instance are serialized.
class OnnxEvaluator final : public mcts::IEvaluator {
 public:
  explicit OnnxEvaluator(OnnxEvaluatorConfig config);
  ~OnnxEvaluator() override;

  OnnxEvaluator(const OnnxEvaluator&) = delete;
  OnnxEvaluator& operator=(const OnnxEvaluator&) = delete;

  mcts::Evaluation evaluate(const mcts::IGameState& state) override;

  const OnnxEvaluatorConfig& config() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace suzero::chess5d
