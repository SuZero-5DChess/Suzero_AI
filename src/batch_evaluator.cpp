// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-5DChess contributors

#include "suzero/mcts/batch_evaluator.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace suzero::mcts {

struct BatchedEvaluator::Impl {
  struct Request {
    const IGameState* state = nullptr;
    Evaluation result;
    std::exception_ptr error;
    bool done = false;
    std::mutex mutex;
    std::condition_variable ready;
  };

  Impl(BatchEvaluatorConfig config, const BackendFactory& factory)
      : max_batch_size(config.maxBatchSize),
        num_server_threads(config.numServerThreads),
        queue_capacity(config.maxQueuedRequests) {
    if (max_batch_size <= 0) {
      throw std::invalid_argument(
          "BatchedEvaluator: maxBatchSize must be positive");
    }
    if (num_server_threads <= 0) {
      throw std::invalid_argument(
          "BatchedEvaluator: numServerThreads must be positive");
    }
    if (queue_capacity < 0) {
      throw std::invalid_argument(
          "BatchedEvaluator: maxQueuedRequests must be non-negative");
    }

    // Create every backend before spawning any thread. Besides surfacing
    // construction errors synchronously, this lets us clamp the effective
    // batch size to what the smallest backend accepts.
    backends.reserve(static_cast<std::size_t>(num_server_threads));
    for (int i = 0; i < num_server_threads; ++i) {
      std::unique_ptr<IBatchBackend> backend = factory();
      if (backend == nullptr) {
        throw std::invalid_argument(
            "BatchedEvaluator: backend factory returned null");
      }
      const int backend_cap = backend->maxBatchSize();
      if (backend_cap <= 0) {
        throw std::invalid_argument(
            "BatchedEvaluator: backend maxBatchSize must be positive");
      }
      max_batch_size = std::min(max_batch_size, backend_cap);
      backends.push_back(std::move(backend));
    }

    if (queue_capacity <= 0) {
      queue_capacity = max_batch_size * 4 * num_server_threads;
    }

    threads.reserve(static_cast<std::size_t>(num_server_threads));
    for (int i = 0; i < num_server_threads; ++i) {
      IBatchBackend* backend = backends[static_cast<std::size_t>(i)].get();
      threads.emplace_back([this, backend]() { serve(backend); });
    }
  }

  ~Impl() {
    {
      std::unique_lock<std::mutex> lock(mutex);
      closed = true;
      not_full.notify_all();
      not_empty.notify_all();
      // Requests already queued are still served. Wait until every caller has
      // left an in-flight evaluate() before the Impl (and its queue) dies.
      clients_done.wait(lock, [this] { return active_clients == 0; });
    }
    for (std::thread& thread : threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

  void finishClient() {
    std::lock_guard<std::mutex> lock(mutex);
    --active_clients;
    if (active_clients == 0) {
      clients_done.notify_all();
    }
  }

  void serve(IBatchBackend* backend) {
    std::vector<Request*> batch;
    std::vector<const IGameState*> states;
    std::vector<Evaluation> results;

    for (;;) {
      {
        std::unique_lock<std::mutex> lock(mutex);
        not_empty.wait(lock, [this] { return closed || !queue.empty(); });
        if (queue.empty()) {
          // Closed and fully drained.
          break;
        }
        // KataGo semantics: take what is available now, up to the cap, and
        // never wait for the batch to fill.
        const std::size_t take = std::min(
            queue.size(), static_cast<std::size_t>(max_batch_size));
        batch.assign(queue.begin(),
                     queue.begin() + static_cast<std::ptrdiff_t>(take));
        queue.erase(queue.begin(),
                    queue.begin() + static_cast<std::ptrdiff_t>(take));
        not_full.notify_all();
      }

      states.clear();
      states.reserve(batch.size());
      for (Request* request : batch) {
        states.push_back(request->state);
      }

      results.clear();
      std::exception_ptr error;
      try {
        backend->evaluateBatch(states, results);
        if (results.size() != states.size()) {
          throw std::logic_error(
              "BatchedEvaluator: backend returned a wrong-sized batch");
        }
      } catch (...) {
        error = std::current_exception();
        results.clear();
      }

      for (std::size_t i = 0; i < batch.size(); ++i) {
        Request* request = batch[i];
        {
          std::lock_guard<std::mutex> guard(request->mutex);
          if (error) {
            request->error = error;
          } else {
            request->result = std::move(results[i]);
          }
          request->done = true;
          // Notify while locked so the waiter cannot destroy this stack request.
          request->ready.notify_one();
        }
      }

      batches.fetch_add(1, std::memory_order_relaxed);
      rows.fetch_add(batch.size(), std::memory_order_relaxed);
      updateMaxObservedBatch(batch.size());
    }
  }

  void updateMaxObservedBatch(std::uint64_t observed) {
    std::uint64_t current = max_observed_batch.load(std::memory_order_relaxed);
    while (observed > current &&
           !max_observed_batch.compare_exchange_weak(
               current, observed, std::memory_order_relaxed)) {
    }
  }

  // Declared first so the constructor's initialization order matches.
  int max_batch_size;
  const int num_server_threads;
  int queue_capacity;

  std::vector<std::unique_ptr<IBatchBackend>> backends;
  std::vector<std::thread> threads;

  std::mutex mutex;
  std::condition_variable not_empty;
  std::condition_variable not_full;
  std::condition_variable clients_done;
  std::deque<Request*> queue;
  bool closed = false;
  int active_clients = 0;

  std::atomic<std::uint64_t> requests{0};
  std::atomic<std::uint64_t> batches{0};
  std::atomic<std::uint64_t> rows{0};
  std::atomic<std::uint64_t> max_observed_batch{0};
};

BatchedEvaluator::BatchedEvaluator(BatchEvaluatorConfig config,
                                   BackendFactory factory)
    : impl_(std::make_unique<Impl>(std::move(config), factory)) {}

BatchedEvaluator::~BatchedEvaluator() = default;

Evaluation BatchedEvaluator::evaluate(const IGameState& state) {
  Impl::Request request;
  request.state = &state;

  bool enqueued = false;
  {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (impl_->closed) {
      throw std::runtime_error("BatchedEvaluator: evaluator is closed");
    }
    ++impl_->active_clients;
    impl_->not_full.wait(lock, [this] {
      return impl_->closed ||
             impl_->queue.size() <
                 static_cast<std::size_t>(impl_->queue_capacity);
    });
    if (!impl_->closed) {
      impl_->queue.push_back(&request);
      impl_->requests.fetch_add(1, std::memory_order_relaxed);
      enqueued = true;
      impl_->not_empty.notify_one();
    }
  }

  if (!enqueued) {
    impl_->finishClient();
    throw std::runtime_error("BatchedEvaluator: evaluator is closed");
  }

  std::exception_ptr error;
  Evaluation result;
  {
    std::unique_lock<std::mutex> lock(request.mutex);
    request.ready.wait(lock, [&request] { return request.done; });
    error = request.error;
    if (!error) {
      result = std::move(request.result);
    }
  }

  impl_->finishClient();

  if (error) {
    std::rethrow_exception(error);
  }
  return result;
}

int BatchedEvaluator::maxBatchSize() const { return impl_->max_batch_size; }

int BatchedEvaluator::numServerThreads() const {
  return impl_->num_server_threads;
}

BatchEvaluatorStats BatchedEvaluator::stats() const {
  BatchEvaluatorStats snapshot;
  snapshot.requests = impl_->requests.load(std::memory_order_relaxed);
  snapshot.batches = impl_->batches.load(std::memory_order_relaxed);
  snapshot.rows = impl_->rows.load(std::memory_order_relaxed);
  snapshot.maxObservedBatch =
      impl_->max_observed_batch.load(std::memory_order_relaxed);
  return snapshot;
}

}  // namespace suzero::mcts
