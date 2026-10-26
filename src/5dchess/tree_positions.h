// SPDX-License-Identifier: MIT
#pragma once

#include "suzero/5dchess/selfplay_bridge.h"
#include "suzero/mcts/random.h"
#include "suzero/mcts/search_node.h"

namespace suzero::chess5d::internal {

void validateTreePositionConfig(const SelfPlayBridgeConfig& config);

/// Read-only extraction, called after choosing the move and before root advance.
/// State is replayed through clones; no evaluator or search RNG is touched.
/// The root's played-action branch is excluded in its entirety.
std::vector<BridgeSample> extractTreePositions(
    const SelfPlayBridgeConfig& config, const mcts::IGameState& rootState,
    const mcts::SearchNode& root, mcts::ActionId playedAction, mcts::Rng& rng);

}  // namespace suzero::chess5d::internal
