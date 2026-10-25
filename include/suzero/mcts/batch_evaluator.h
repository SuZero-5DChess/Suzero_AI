// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-5DChess contributors
//
// This file is part of SuZero AI, a 5D chess AlphaZero project. The search
// core is derived from KataGo (https://github.com/lightvector/KataGo).
// See THIRD_PARTY_NOTICES.md.
//
// In-process batching evaluator, following KataGo's NNEvaluator design:
// many game/search threads submit one position query at a time into a shared
// queue; a small number of server threads drain the queue into a batch and run
// one device call for the whole batch.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "suzero/mcts/evaluator.h"
#include "suzero/mcts/game_state.h"

namespace suzero::mcts {

/// Backend that evaluates a whole batch in one device call.
///
/// The batching evaluator owns one backend per server thread. A backend is
/// therefore never called concurrently with itself, so implementations do not
/// need internal locking. They must still be safe to *read* the requested
/// states: the calling game thread is blocked inside the request until the
/// result is delivered.
class IBatchBackend {
 public:
  virtual ~IBatchBackend() = default;

  /// Maximum number of rows accepted by one call. Must be positive.
  ///
  /// `BatchedEvaluator` clamps its configured batch size to this value, so a
  /// backend can advertise a tighter memory limit than the caller requested.
  virtual int maxBatchSize() const = 0;

  /// Evaluates `states` and writes exactly `states.size()` results in the same
  /// order. `results` is cleared before the call and may retain capacity.
  ///
  /// Throwing propagates the exception to every client in the batch.
  virtual void evaluateBatch(const std::vector<const IGameState*>& states,
                             std::vector<Evaluation>& results) = 0;
};

/// Tuning for BatchedEvaluator.
struct BatchEvaluatorConfig {
  /// Upper bound on rows per device call.
  int maxBatchSize = 128;

  /// Number of independent backend instances / server threads. Each instance
  /// is created by the factory supplied to the constructor.
  int numServerThreads = 1;

  /// Maximum queued requests before `evaluate()` blocks (backpressure).
  /// <= 0 derives `maxBatchSize * 4 * numServerThreads`, mirroring KataGo's
  /// query-queue sizing.
  int maxQueuedRequests = 0;
};

/// Snapshot of batching statistics.
struct BatchEvaluatorStats {
  std::uint64_t requests = 0;
  std::uint64_t batches = 0;
  std::uint64_t rows = 0;
  std::uint64_t maxObservedBatch = 0;
};

/// Thread-safe IEvaluator that batches concurrent calls.
///
/// Semantics, matching KataGo's server loop:
///  * `evaluate()` enqueues one request and blocks until its result arrives.
///  * A server thread blocks until the queue is non-empty, then immediately
///    takes up to `maxBatchSize()` entries without waiting for the batch to
///    fill. There is no artificial batching delay; parallelism across games is
///    what fills batches.
///  * The queue is bounded, so a slow backend applies backpressure to callers
///    instead of growing without limit.
///  * An exception thrown by the backend is delivered to every caller whose
///    request was in that batch.
///
/// Unlike most IEvaluator implementations this class is safe to call from many
/// threads at once. Destroying it while evaluations are in flight is allowed:
/// the destructor closes the queue, lets servers finish requests that are
/// already queued or in progress, waits for active callers to leave, and then
/// joins the server threads. New calls made after the queue is closed throw
/// std::runtime_error.
class BatchedEvaluator final : public IEvaluator {
 public:
  using BackendFactory = std::function<std::unique_ptr<IBatchBackend>()>;

  /// Creates `config.numServerThreads` backends and starts their threads.
  /// `factory` must return a non-null backend each time it is invoked.
  BatchedEvaluator(BatchEvaluatorConfig config, BackendFactory factory);
  ~BatchedEvaluator() override;

  BatchedEvaluator(const BatchedEvaluator&) = delete;
  BatchedEvaluator& operator=(const BatchedEvaluator&) = delete;
  BatchedEvaluator(BatchedEvaluator&&) = delete;
  BatchedEvaluator& operator=(BatchedEvaluator&&) = delete;

  /// Enqueues `state`, blocks for the result, and returns it.
  ///
  /// `state` must remain alive and unmodified until this call returns. That is
  /// automatic when the caller is the search thread that owns the state.
  Evaluation evaluate(const IGameState& state) override;

  /// Effective batch cap: `config.maxBatchSize` clamped to every backend's
  /// advertised maximum.
  int maxBatchSize() const;

  int numServerThreads() const;

  BatchEvaluatorStats stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace suzero::mcts
