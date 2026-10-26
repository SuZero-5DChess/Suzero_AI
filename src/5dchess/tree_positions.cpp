// SPDX-License-Identifier: MIT
// Tree-position extraction follows KataGo's recordTreePositionsRec, adapted
// for graph search and 5D chess small steps (players need not alternate).
#include "tree_positions.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace suzero::chess5d::internal {

void validateTreePositionConfig(const SelfPlayBridgeConfig& config) {
  if (config.recordTreeMinVisits < 1 || config.recordTreeMaxPerTurn < 1 ||
      config.recordTreeMaxDepth < 1 || config.recordTreeMaxDepth > 64) {
    throw std::invalid_argument(
        "runSelfPlay: tree min visits and max per turn must be positive; "
        "tree max depth must be in [1, 64]");
  }
  if (!std::isfinite(config.recordTreeSampleProbability) ||
      config.recordTreeSampleProbability < 0.0 ||
      config.recordTreeSampleProbability > 1.0) {
    throw std::invalid_argument(
        "runSelfPlay: recordTreeSampleProbability must be finite and in [0, 1]");
  }
}

namespace {

class Extractor {
 public:
  Extractor(const SelfPlayBridgeConfig& config, mcts::ActionId playedAction,
            mcts::Rng& rng)
      : config_(config), played_action_(playedAction), rng_(rng) {}

  std::vector<BridgeSample> run(const mcts::IGameState& state,
                                const mcts::SearchNode& root) {
    walk(state, root, 0, {true, true});
    return std::move(samples_);
  }

 private:
  bool full() const {
    return samples_.size() >=
           static_cast<std::size_t>(config_.recordTreeMaxPerTurn);
  }

  void walk(const mcts::IGameState& state, const mcts::SearchNode& node,
            int depth, std::array<bool, 2> alwaysBest) {
    if (full() || remaining_nodes_ == 0) {
      return;
    }
    --remaining_nodes_;
    const auto player = node.player();
    if (player < 0 || player > 1 || !node.isExpanded() || node.isTerminal() ||
        state.isTerminal() || state.currentPlayer() != player ||
        state.stateKey() != node.stateKey() ||
        !path_.insert(node.stateKey()).second) {
      return;
    }

    std::int64_t total = 0;
    std::int64_t bestVisits = -1;
    mcts::ActionId bestAction = mcts::kInvalidAction;
    bool valid = true;
    for (int i = 0; i < static_cast<int>(node.priors().size()); ++i) {
      const auto visits = node.edgeVisits(i);
      if (visits < 0 || total > std::numeric_limits<std::int64_t>::max() - visits) {
        valid = false;
        break;
      }
      total += visits;
      const auto action = node.priors()[static_cast<std::size_t>(i)].action;
      if (visits > bestVisits || (visits == bestVisits && action < bestAction)) {
        bestVisits = visits;
        bestAction = action;
      }
    }
    // Incoming visits alone are not enough: an expanded leaf has no policy.
    // Also require a full threshold of outgoing work, all local to this node.
    if (!valid || total < config_.recordTreeMinVisits) {
      path_.erase(node.stateKey());
      return;
    }

    const auto* input = dynamic_cast<const IChess5DModelInput*>(&state);
    if (depth > 0 && alwaysBest[static_cast<std::size_t>(player)] && input &&
        !sampled_.count(node.stateKey())) {
      const auto& actions = input->actions();
      const double utility = node.stats().utilityAvg();
      bool usable = static_cast<int>(actions.size()) == state.actionCount() &&
                    std::isfinite(utility) && node.stats().weightSum() > 0.0 &&
                    std::any_of(actions.begin(), actions.end(),
                                [](const Action& action) { return !action.is_submit; });
      std::vector<float> policy(actions.size(), 0.0f);
      for (int i = 0; usable && i < static_cast<int>(node.priors().size()); ++i) {
        const auto action = node.priors()[static_cast<std::size_t>(i)].action;
        if (action < 0 || action >= static_cast<mcts::ActionId>(actions.size()) ||
            !state.isLegal(action)) {
          usable = false;
          break;
        }
        policy[static_cast<std::size_t>(action)] =
            static_cast<float>(static_cast<double>(node.edgeVisits(i)) /
                               static_cast<double>(total));
      }
      // One probability trial per eligible state, even with multiple DAG paths.
      if (usable) {
        sampled_.insert(node.stateKey());
      }
      if (usable && rng_.nextDouble() < config_.recordTreeSampleProbability) {
        BridgeSample sample;
        sample.encoded = input->encodePosition();
        sample.actions = actions;
        sample.policy = std::move(policy);
        sample.player = player;
        sample.moveCount = state.moveCount();
        // NodeStats already uses THIS player's perspective, not root/White's.
        sample.value = static_cast<float>(std::clamp(utility, -1.0, 1.0));
        sample.source = 1;
        sample.searchVisits = node.stats().visits();
        samples_.push_back(std::move(sample));
      }
    }

    if (depth < config_.recordTreeMaxDepth && !full()) {
      for (int i = 0; i < static_cast<int>(node.priors().size()); ++i) {
        if (full() || remaining_nodes_ == 0) {
          break;
        }
        const auto action = node.priors()[static_cast<std::size_t>(i)].action;
        if ((depth == 0 && action == played_action_) ||
            node.edgeVisits(i) < config_.recordTreeMinVisits) {
          continue;
        }
        const auto* child = node.childAt(i);
        if (!child || path_.count(child->stateKey()) || !state.isLegal(action)) {
          continue;
        }
        auto nextBest = alwaysBest;
        if (action != bestAction) {
          nextBest[static_cast<std::size_t>(player)] = false;
        }
        if (!nextBest[0] && !nextBest[1]) {
          continue;
        }
        auto next = state.clone();
        if (!next) {
          throw std::runtime_error("extractTreePositions: clone returned null");
        }
        next->applyAction(action);
        walk(*next, *child, depth + 1, nextBest);
      }
    }
    path_.erase(node.stateKey());
  }

  const SelfPlayBridgeConfig& config_;
  mcts::ActionId played_action_;
  mcts::Rng& rng_;
  int remaining_nodes_ = 4096;
  std::unordered_set<mcts::StateKey> path_;
  std::unordered_set<mcts::StateKey> sampled_;
  std::vector<BridgeSample> samples_;
};

}  // namespace

std::vector<BridgeSample> extractTreePositions(
    const SelfPlayBridgeConfig& config, const mcts::IGameState& rootState,
    const mcts::SearchNode& root, mcts::ActionId playedAction, mcts::Rng& rng) {
  if (!config.recordTreePositions || config.recordTreeSampleProbability == 0.0) {
    return {};
  }
  validateTreePositionConfig(config);
  return Extractor(config, playedAction, rng).run(rootState, root);
}

}  // namespace suzero::chess5d::internal
