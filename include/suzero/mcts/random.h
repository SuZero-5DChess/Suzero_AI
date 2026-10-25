// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-5DChess contributors
//
// This file is part of SuZero AI, a 5D chess AlphaZero project. The search
// core is derived from KataGo (https://github.com/lightvector/KataGo).
// See THIRD_PARTY_NOTICES.md.

#pragma once

#include <cstdint>

namespace suzero::mcts {

/// Small deterministic RNG used for Dirichlet noise and temperature sampling.
///
/// Implements PCG32 and a Marsaglia-Tsang gamma sampler. It is self-contained
/// so that results are reproducible across standard libraries and platforms.
class Rng {
 public:
  explicit Rng(std::uint64_t seed = 0x853c49e6748fea9bULL,
               std::uint64_t stream = 0xda3e39cb94b95bdbULL);

  void seed(std::uint64_t seed, std::uint64_t stream = 0xda3e39cb94b95bdbULL);

  std::uint32_t nextUint32();
  std::uint64_t nextUint64();

  /// Uniform in [0, 1).
  double nextDouble();

  /// Standard normal, via the Marsaglia polar method.
  double nextGaussian();

  /// Gamma(shape, 1) draw. `shape` must be > 0.
  double nextGamma(double shape);

 private:
  std::uint64_t state_;
  std::uint64_t inc_;

  bool has_spare_gaussian_ = false;
  double spare_gaussian_ = 0.0;
};

}  // namespace suzero::mcts
