// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-02 contributors
//
// Internal numeric helpers shared by the search implementation and its tests.
// This header is not part of the public include tree.

#pragma once

#include <cmath>

namespace suzero::mcts::internal {

inline constexpr double kPi = 3.14159265358979323846;

/// CDF of Student's t distribution with three degrees of freedom.
///
/// KataGo's value weighting uses a Student-t distribution with
/// VALUE_WEIGHT_DEGREES_OF_FREEDOM = 3.0 rather than a normal distribution.
inline double studentTCdf3(double z) {
  const double root3 = std::sqrt(3.0);
  return 0.5 +
         (std::atan(z / root3) + root3 * z / (3.0 + z * z)) / kPi;
}

}  // namespace suzero::mcts::internal
