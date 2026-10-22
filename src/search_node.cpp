// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-02 contributors
//
// This file is part of a game-agnostic PUCT/MCTS library derived from KataGo
// (https://github.com/lightvector/KataGo). See THIRD_PARTY_NOTICES.md.

#include "suzero/mcts/search_node.h"

#include <algorithm>

namespace suzero::mcts {

double NodeStats::childWeight(std::int64_t edgeVisits) const {
  const std::int64_t child_visits = std::max<std::int64_t>(visits_, 1);
  return weight_sum_ * static_cast<double>(edgeVisits) /
         static_cast<double>(child_visits);
}

SearchNode::SearchNode(PlayerId player) : player_(player) {}

std::int64_t SearchNode::edgeVisits(int priorIndex) const {
  if (priorIndex < 0 ||
      priorIndex >= static_cast<int>(edge_index_by_prior_.size())) {
    return 0;
  }
  const int edge_index = edge_index_by_prior_[priorIndex];
  if (edge_index < 0) {
    return 0;
  }
  return edges_[edge_index].visits;
}

const SearchNode* SearchNode::childAt(int priorIndex) const {
  if (priorIndex < 0 ||
      priorIndex >= static_cast<int>(edge_index_by_prior_.size())) {
    return nullptr;
  }
  const int edge_index = edge_index_by_prior_[priorIndex];
  if (edge_index < 0) {
    return nullptr;
  }
  return edges_[edge_index].child.get();
}

const SearchNode* SearchNode::childForAction(ActionId action) const {
  for (std::size_t i = 0; i < priors_.size(); ++i) {
    if (priors_[i].action == action) {
      return childAt(static_cast<int>(i));
    }
  }
  return nullptr;
}

}  // namespace suzero::mcts
