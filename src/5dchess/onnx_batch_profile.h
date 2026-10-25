// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-02 contributors

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>

namespace suzero::chess5d::internal {

/// Optional, per-session diagnostics for OnnxModel::evaluateBatch().
///
/// Enabled only when SUZERO_ONNX_PROFILE_PATH is non-empty. When disabled,
/// stamp() does not read the clock. Samples are accumulated in memory by the
/// batch backend thread and one JSON line is appended when the owning
/// OnnxModel::Impl is destroyed. No file is touched on the inference path.
///
/// Phase durations are wall-clock nanoseconds measured on the calling thread.
/// run_ns covers Ort::Session::Run, including ORT's synchronous input/output
/// transfers and execution-provider overhead; it is not pure GPU/kernel time.
/// Each OnnxModel is owned by one batch backend, but separate server threads
/// can overlap in wall time, so durations summed across JSON lines are not an
/// elapsed-time total and must not be treated as device utilization.
class OnnxBatchProfile {
 public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  struct Call {
    // prepare_ns spans `started` to `lock_begin` (or to `finished` for a
    // forced-only call). encode/allocate/pack are nested subintervals of that
    // span and must not be added to prepare_ns again when reporting totals.
    TimePoint started{};
    TimePoint encode_begin{};
    TimePoint encode_end{};
    TimePoint allocate_begin{};
    TimePoint allocate_end{};
    TimePoint pack_begin{};
    TimePoint pack_end{};
    TimePoint lock_begin{};
    TimePoint run_begin{};
    TimePoint run_end{};
    TimePoint finished{};

    std::uint64_t backend_rows = 0;
    std::uint64_t inference_rows = 0;
    std::uint64_t forced_rows = 0;
    std::uint64_t used_boards = 0;
    std::uint64_t board_slots = 0;
    std::uint64_t used_actions = 0;
    std::uint64_t action_slots = 0;
    std::uint64_t max_boards = 0;
    std::uint64_t max_q = 0;

    /// True only after a successful Run and decode. Forced-only calls leave
    /// this false: they count as backend calls/rows, but never as ORT runs.
    bool ran = false;
  };

  OnnxBatchProfile() {
    const char* path = std::getenv("SUZERO_ONNX_PROFILE_PATH");
    if (path != nullptr && path[0] != '\0') {
      path_ = path;
      enabled_ = true;
    }
  }

  ~OnnxBatchProfile() {
    if (!armed_ || !enabled_) {
      return;
    }
    try {
      writeLine();
    } catch (...) {
      std::fprintf(stderr,
                   "OnnxBatchProfile: failed to write profile '%s'\n",
                   path_.c_str());
    }
  }

  OnnxBatchProfile(const OnnxBatchProfile&) = delete;
  OnnxBatchProfile& operator=(const OnnxBatchProfile&) = delete;

  /// Enables output only after the owning OnnxModel finished construction.
  /// A failed construction therefore does not emit an all-zero record.
  void arm() noexcept { armed_ = enabled_; }

  bool enabled() const noexcept { return enabled_; }

  /// Returns a timestamp only when profiling is enabled, keeping disabled
  /// inference paths free of steady_clock reads.
  TimePoint stamp() const noexcept {
    return enabled_ ? Clock::now() : TimePoint{};
  }

  /// Commits one completed call. Callers invoke this only after all work for
  /// the call succeeded; exceptions therefore do not produce partial success
  /// records. Forced-only calls (ran == false) record prepare time from
  /// `started` to `finished` and leave all Run/decode fields at zero.
  void record(const Call& call) noexcept {
    if (!enabled_) {
      return;
    }

    Stats& stats = stats_;
    ++stats.backend_calls;
    stats.backend_rows += call.backend_rows;
    stats.inference_rows += call.inference_rows;
    stats.forced_rows += call.forced_rows;
    stats.used_boards += call.used_boards;
    stats.board_slots += call.board_slots;
    stats.used_actions += call.used_actions;
    stats.action_slots += call.action_slots;

    if (call.ran) {
      ++stats.ort_runs;
      stats.prepare_ns += elapsedNs(call.started, call.lock_begin);
      stats.lock_wait_ns += elapsedNs(call.lock_begin, call.run_begin);
      stats.run_ns += elapsedNs(call.run_begin, call.run_end);
      stats.decode_ns += elapsedNs(call.run_end, call.finished);
      stats.allocate_ns +=
          elapsedNs(call.allocate_begin, call.allocate_end);
      stats.pack_ns += elapsedNs(call.pack_begin, call.pack_end);
      // Keyed by rows actually sent to ORT. Forced-only calls do not create a
      // zero-sized histogram entry.
      try {
        ++stats.batch_histogram[call.inference_rows];
      } catch (...) {
        std::fprintf(stderr,
                     "OnnxBatchProfile: failed to update batch histogram\n");
      }
    } else {
      stats.prepare_ns += elapsedNs(call.started, call.finished);
    }
    // Nested in prepare_ns, not an additional total. Recorded for both normal
    // and forced-only calls because both run the action/encode loop.
    stats.encode_ns += elapsedNs(call.encode_begin, call.encode_end);

    stats.max_batch = std::max(stats.max_batch, call.backend_rows);
    if (call.ran) {
      stats.max_boards = std::max(stats.max_boards, call.max_boards);
      stats.max_q = std::max(stats.max_q, call.max_q);
    }
  }

 private:
  struct Stats {
    std::uint64_t backend_calls = 0;
    std::uint64_t backend_rows = 0;
    std::uint64_t inference_rows = 0;
    std::uint64_t forced_rows = 0;
    std::uint64_t ort_runs = 0;
    std::uint64_t used_boards = 0;
    std::uint64_t board_slots = 0;
    std::uint64_t used_actions = 0;
    std::uint64_t action_slots = 0;
    std::uint64_t max_batch = 0;
    std::uint64_t max_boards = 0;
    std::uint64_t max_q = 0;
    std::uint64_t prepare_ns = 0;
    std::uint64_t encode_ns = 0;
    std::uint64_t allocate_ns = 0;
    std::uint64_t pack_ns = 0;
    std::uint64_t lock_wait_ns = 0;
    std::uint64_t run_ns = 0;
    std::uint64_t decode_ns = 0;
    std::map<std::uint64_t, std::uint64_t> batch_histogram;
  };

  static std::uint64_t elapsedNs(TimePoint begin, TimePoint end) noexcept {
    if (begin == TimePoint{} || end == TimePoint{} || end < begin) {
      return 0;
    }
    const auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
            .count();
    return duration > 0 ? static_cast<std::uint64_t>(duration) : 0;
  }

  static std::mutex& writeMutex() {
    static std::mutex mutex;
    return mutex;
  }

  void writeLine() {
    const Stats& stats = stats_;
    std::ostringstream line;
    line << "{\"backend_calls\":" << stats.backend_calls
         << ",\"backend_rows\":" << stats.backend_rows
         << ",\"inference_rows\":" << stats.inference_rows
         << ",\"forced_rows\":" << stats.forced_rows
         << ",\"ort_runs\":" << stats.ort_runs
         << ",\"used_boards\":" << stats.used_boards
         << ",\"board_slots\":" << stats.board_slots
         << ",\"used_actions\":" << stats.used_actions
         << ",\"action_slots\":" << stats.action_slots
         << ",\"max_batch\":" << stats.max_batch
         << ",\"max_boards\":" << stats.max_boards
         << ",\"max_q\":" << stats.max_q
         << ",\"prepare_ns\":" << stats.prepare_ns
         << ",\"encode_ns\":" << stats.encode_ns
         << ",\"allocate_ns\":" << stats.allocate_ns
         << ",\"pack_ns\":" << stats.pack_ns
         << ",\"lock_wait_ns\":" << stats.lock_wait_ns
         << ",\"run_ns\":" << stats.run_ns
         << ",\"decode_ns\":" << stats.decode_ns
         << ",\"batch_histogram\":{";
    bool first = true;
    for (const auto& entry : stats.batch_histogram) {
      if (!first) {
        line << ',';
      }
      first = false;
      line << '"' << entry.first << "\":" << entry.second;
    }
    line << "}}\n";

    std::lock_guard<std::mutex> lock(writeMutex());
    std::ofstream output(path_, std::ios::app);
    if (!output) {
      std::fprintf(stderr,
                   "OnnxBatchProfile: cannot open profile '%s'\n",
                   path_.c_str());
      return;
    }
    const std::string text = line.str();
    output << text;
    output.flush();
    if (!output) {
      std::fprintf(stderr,
                   "OnnxBatchProfile: failed to append profile '%s'\n",
                   path_.c_str());
    }
  }

  std::string path_;
  Stats stats_;
  bool enabled_ = false;
  bool armed_ = false;
};

}  // namespace suzero::chess5d::internal
