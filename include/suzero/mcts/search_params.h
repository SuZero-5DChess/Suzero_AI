// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-02 contributors
//
// This file is part of a game-agnostic PUCT/MCTS library derived from KataGo
// (https://github.com/lightvector/KataGo). See THIRD_PARTY_NOTICES.md.

#pragma once

#include <cstdint>

namespace suzero::mcts {

/// Tunable parameters of the PUCT search.
///
/// The names and semantics follow KataGo's SearchParams where the concept is
/// game-independent. Parameters that only make sense for Go (score utility,
/// ownership, ko, symmetries, human-SL models, pass handling, time controls,
/// visit caps, ...) are intentionally absent. See docs/katago-mapping.md.
struct SearchParams {
  // ---------------------------------------------------------------- limits --
  // A value <= 0 means "no limit" for each limit.

  /// Stop when the root has at least this many visits. Tree reuse counts the
  /// visits accumulated before this call.
  std::int64_t maxVisits = 0;

  /// Stop after this many playouts *in the current run* (tree reuse does not
  /// count). The default keeps an accidental `run()` bounded.
  std::int64_t maxPlayouts = 800;

  /// Wall-clock limit for the current run, in seconds.
  double maxTimeSeconds = 0.0;

  /// Maximum number of edges traversed in one playout. <= 0 means unlimited.
  /// Graph search already stops repeated nodes on a path, but this remains a
  /// hard safety bound.
  int maxDepth = 0;

  // ------------------------------------------------------------ graph search --

  /// Probability of skipping an edge-visit catch-up and descending into an
  /// already heavily visited transposition anyway. 0 always catches up, which
  /// is KataGo's default. See docs/katago-mapping.md.
  double graphSearchCatchUpLeakProb = 0.0;

  /// Whether each playout tracks the nodes on its path to stop cycles.
  ///
  /// Set to false only for games where a playout can never return to a state
  /// it already visited (strictly progressing games). This removes the per-edge
  /// path-set hash/lookup. On a cyclic game it can recurse until `maxDepth` or
  /// the call stack is exhausted, so keep `maxDepth` set if there is any doubt.
  bool graphSearchCycleCheckEnabled = true;

  // ------------------------------------------------------------ exploration --

  /// Constant part of the PUCT exploration coefficient.
  double cpuctExploration = 1.0;

  /// Log-scaling part of the PUCT exploration coefficient.
  double cpuctExplorationLog = 0.0;

  /// Scale at which the log term starts to matter.
  double cpuctExplorationBase = 500.0;

  /// Prior standard deviation of utility, used by the utility-stdev factor.
  /// Must be > 0.
  double cpuctUtilityStdevPrior = 0.25;

  /// Pseudo-weight of the prior in the utility-stdev estimate.
  double cpuctUtilityStdevPriorWeight = 1.0;

  /// 0 disables utility-stdev scaling; 1 scales exploration directly by the
  /// estimated utility standard deviation relative to the prior.
  double cpuctUtilityStdevScale = 0.0;

  /// Scale of the utility range. Used for the loss value FPU is blended
  /// towards. For utilities in [-1, 1] this should usually be 1.
  double utilityRadius = 1.0;

  // ------------------------------------------------------- first play urgency --

  /// Maximum FPU reduction for unvisited children.
  double fpuReductionMax = 0.2;

  /// Blend of the FPU value towards `-utilityRadius`.
  double fpuLossProp = 0.0;

  /// Root-specific overrides.
  double rootFpuReductionMax = 0.2;
  double rootFpuLossProp = 0.0;

  /// If true, blend the parent utility used by FPU towards the node's own
  /// evaluator value based on the policy mass already visited.
  bool fpuParentWeightByVisitedPolicy = false;

  /// Exponent applied to the visited policy mass in that blend.
  double fpuParentWeightByVisitedPolicyPow = 1.0;

  /// Alternative blend: 0 = parent average, 1 = node's evaluator value.
  /// Only used when `fpuParentWeightByVisitedPolicy` is false.
  double fpuParentWeight = 0.0;

  // -------------------------------------------------------------- root prior --

  /// Replace a fraction of the root prior with Dirichlet noise. The noise is
  /// generated once per root node (not once per call to run()).
  bool rootNoiseEnabled = false;

  /// Total concentration (sum of Dirichlet alphas). KataGo uses
  /// 0.03 * board area; for other games choose a value on the order of the
  /// typical number of legal actions.
  double rootDirichletNoiseTotalConcentration = 10.83;

  /// Mixture weight of the noise, in [0, 1].
  double rootDirichletNoiseWeight = 0.25;

  /// Temperature applied to the root prior before noise. 1 = unchanged.
  double rootPolicyTemperature = 1.0;

  /// Temperature used at the start of the game. The effective temperature is
  /// interpolated from this value towards `rootPolicyTemperature` using
  /// `chosenMoveTemperatureHalflife` and IGameState::moveCount().
  /// 1 = unchanged (default, matching KataGo).
  double rootPolicyTemperatureEarly = 1.0;

  // ---------------------------------------------------------- move selection --

  /// Sampling temperature for `chooseAction`. <= 1e-4 means argmax.
  double chosenMoveTemperature = 0.0;

  /// Temperature used at the start of the game by `chooseAction()` (the
  /// overload without an explicit temperature). Interpolated towards
  /// `chosenMoveTemperature` with `chosenMoveTemperatureHalflife`.
  double chosenMoveTemperatureEarly = 0.0;

  /// Number of moves after which the early temperatures have moved halfway
  /// towards their final values. Must be > 0.
  ///
  /// KataGo also scales this by the board size (19/sqrt(area)); this library
  /// has no board concept, so the halflife is expressed directly in moves.
  double chosenMoveTemperatureHalflife = 19.0;

  /// Temperatures only dampen moves whose pre-temperature probability is at
  /// most this value. 1 means every move is affected.
  double chosenMoveTemperatureOnlyBelowProb = 1.0;

  /// Subtract this many units of child weight from every move before sampling.
  /// Capped at maxWeight / 64.
  double chosenMoveSubtract = 0.0;

  /// Drop moves below this much child weight before sampling.
  /// Capped at maxWeight / 64.
  double chosenMovePrune = 1.0;

  // ------------------------------------------------------- value aggregation --

  /// Exponent used to downweight children whose value is below the weighted
  /// average of their siblings (KataGo's value weighting). 0 disables it.
  double valueWeightExponent = 0.5;
};

}  // namespace suzero::mcts
