// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-02 contributors
//
// This file is part of a game-agnostic PUCT/MCTS library derived from KataGo
// (https://github.com/lightvector/KataGo). See THIRD_PARTY_NOTICES.md.

#include "suzero/mcts/random.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace suzero::mcts {

namespace {

constexpr std::uint64_t kMultiplier = 6364136223846793005ULL;

}  // namespace

Rng::Rng(std::uint64_t seed, std::uint64_t stream) {
  this->seed(seed, stream);
}

void Rng::seed(std::uint64_t seed, std::uint64_t stream) {
  state_ = 0U;
  // Stream must be odd for PCG32.
  inc_ = (stream << 1U) | 1U;
  nextUint32();
  state_ += seed;
  nextUint32();
  has_spare_gaussian_ = false;
  spare_gaussian_ = 0.0;
}

std::uint32_t Rng::nextUint32() {
  const std::uint64_t old_state = state_;
  state_ = old_state * kMultiplier + inc_;
  const std::uint32_t xor_shifted =
      static_cast<std::uint32_t>(((old_state >> 18U) ^ old_state) >> 27U);
  const std::uint32_t rot = static_cast<std::uint32_t>(old_state >> 59U);
  return (xor_shifted >> rot) | (xor_shifted << ((~rot + 1U) & 31U));
}

std::uint64_t Rng::nextUint64() {
  const std::uint64_t hi = nextUint32();
  const std::uint64_t lo = nextUint32();
  return (hi << 32U) | lo;
}

double Rng::nextDouble() {
  // 53 random bits, uniform in [0, 1).
  return static_cast<double>(nextUint64() >> 11U) * (1.0 / 9007199254740992.0);
}

double Rng::nextGaussian() {
  if (has_spare_gaussian_) {
    has_spare_gaussian_ = false;
    return spare_gaussian_;
  }

  double u = 0.0;
  double v = 0.0;
  double s = 0.0;
  do {
    u = 2.0 * nextDouble() - 1.0;
    v = 2.0 * nextDouble() - 1.0;
    s = u * u + v * v;
  } while (s >= 1.0 || s == 0.0);

  const double scale = std::sqrt(-2.0 * std::log(s) / s);
  spare_gaussian_ = v * scale;
  has_spare_gaussian_ = true;
  return u * scale;
}

double Rng::nextGamma(double shape) {
  if (!(shape > 0.0) || !std::isfinite(shape)) {
    throw std::invalid_argument("Rng::nextGamma: shape must be finite and > 0");
  }

  // Boost a small shape into the range handled by Marsaglia-Tsang.
  if (shape < 1.0) {
    const double u = nextDouble();
    // u can be 0; log(0) is -inf which would produce 0, which is a valid
    // limiting sample but is unhelpful. Use the open interval instead.
    return nextGamma(shape + 1.0) *
           std::pow(u > 0.0 ? u : std::numeric_limits<double>::min(),
                    1.0 / shape);
  }

  const double d = shape - 1.0 / 3.0;
  const double c = 1.0 / std::sqrt(9.0 * d);
  for (;;) {
    double x = 0.0;
    double v = 0.0;
    do {
      x = nextGaussian();
      v = 1.0 + c * x;
    } while (v <= 0.0);

    v = v * v * v;
    const double u = nextDouble();
    if (u < 1.0 - 0.0331 * x * x * x * x) {
      return d * v;
    }
    if (u > 0.0 &&
        std::log(u) < 0.5 * x * x + d * (1.0 - v + std::log(v))) {
      return d * v;
    }
  }
}

}  // namespace suzero::mcts
