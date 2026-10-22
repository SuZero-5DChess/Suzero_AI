// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-02 contributors
//
// This file is part of a game-agnostic PUCT/MCTS library derived from KataGo
// (https://github.com/lightvector/KataGo). See THIRD_PARTY_NOTICES.md.

#pragma once

#include <cstdint>
#include <vector>

#include "suzero/mcts/types.h"

namespace suzero::mcts {

class PuctSearch;

/// A legal action and its prior probability at a node.
struct Prior {
  ActionId action = kInvalidAction;
  float probability = 0.0f;
};

/// Aggregated statistics of one search node.
///
/// Utilities are stored as weighted sums in the node player's perspective, so
/// that the perspective used by backup is explicit and stable.
class NodeStats {
 public:
  std::int64_t visits() const { return visits_; }

  /// Sum of the weights of all values that contributed to this node.
  double weightSum() const { return weight_sum_; }

  /// Weighted average utility from this node's player perspective.
  double utilityAvg() const {
    return weight_sum_ > 0.0 ? utility_sum_ / weight_sum_ : 0.0;
  }

  /// Weighted average of the squared utility, used by FPU / exploration
  /// scaling.
  double utilitySqAvg() const {
    return weight_sum_ > 0.0 ? utility_sq_sum_ / weight_sum_ : 0.0;
  }

  /// KataGo's edge-visit-adjusted child weight:
  ///   weightSum * edgeVisits / max(childVisits, 1)
  double childWeight(std::int64_t edgeVisits) const;

 private:
  friend class PuctSearch;

  std::int64_t visits_ = 0;
  double weight_sum_ = 0.0;
  double utility_sum_ = 0.0;
  double utility_sq_sum_ = 0.0;
};

/// One node of the search graph.
///
/// A node stores its prior list once (at expansion), its statistics, and the
/// edges that have been expanded so far. It does not store a game state; the
/// search reconstructs state by applying actions while descending. Nodes are
/// owned by PuctSearch's node table; edges merely reference them, so several
/// parents may point at the same node.
class SearchNode {
 public:
  SearchNode(StateKey stateKey, PlayerId player);

  SearchNode(const SearchNode&) = delete;
  SearchNode& operator=(const SearchNode&) = delete;
  SearchNode(SearchNode&&) = delete;
  SearchNode& operator=(SearchNode&&) = delete;

  const StateKey& stateKey() const { return state_key_; }
  PlayerId player() const { return player_; }
  bool isExpanded() const { return expanded_; }
  bool isTerminal() const { return terminal_; }

  /// The evaluator's value at this node, in this node's player perspective.
  double evaluatorUtility() const { return evaluator_utility_; }

  const NodeStats& stats() const { return stats_; }

  /// Legal, positively-weighted actions, in ascending action id order.
  const std::vector<Prior>& priors() const { return priors_; }

  /// Number of expanded children.
  int childCount() const { return static_cast<int>(edges_.size()); }

  /// Edge visit count for the child at `priorIndex`, or 0 if not expanded.
  std::int64_t edgeVisits(int priorIndex) const;

  /// Immutable child at `priorIndex`, or nullptr if not expanded.
  const SearchNode* childAt(int priorIndex) const;

  /// Immutable child for an action, or nullptr if not expanded.
  const SearchNode* childForAction(ActionId action) const;

 private:
  friend class PuctSearch;

  struct Edge {
    Prior prior;
    SearchNode* child = nullptr;  // Non-owning; owned by the node table.
    std::int64_t visits = 0;
  };

  StateKey state_key_;
  PlayerId player_;
  bool expanded_ = false;
  bool terminal_ = false;
  double evaluator_utility_ = 0.0;
  NodeStats stats_;

  // Parallel arrays: priors_[i] is expanded iff edge_index_by_prior_[i] >= 0.
  std::vector<Prior> priors_;
  std::vector<int> edge_index_by_prior_;
  std::vector<Edge> edges_;
};

}  // namespace suzero::mcts
