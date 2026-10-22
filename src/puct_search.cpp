// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-02 contributors
//
// This file is part of a game-agnostic PUCT/MCTS library derived from KataGo
// (https://github.com/lightvector/KataGo). See THIRD_PARTY_NOTICES.md.

#include "suzero/mcts/puct_search.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace suzero::mcts {

namespace {

constexpr double kTotalChildWeightPuctOffset = 0.01;
constexpr double kValueWeightingVarianceFloor = 1e-8;

double normalCdf(double z) {
  return 0.5 * std::erfc(-z / std::sqrt(2.0));
}

double elapsedSeconds(std::chrono::steady_clock::time_point start) {
  const auto now = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(now - start).count();
}

}  // namespace

PuctSearch::PuctSearch(std::shared_ptr<IGameState> rootState,
                       std::shared_ptr<IEvaluator> evaluator,
                       SearchParams params, std::uint64_t seed)
    : root_state_(std::move(rootState)),
      evaluator_(std::move(evaluator)),
      params_(params),
      rng_(seed) {
  if (!root_state_) {
    throw std::invalid_argument("PuctSearch: root state must not be null");
  }
  if (!evaluator_) {
    throw std::invalid_argument("PuctSearch: evaluator must not be null");
  }
  action_count_ = root_state_->actionCount();
  if (action_count_ <= 0) {
    throw std::invalid_argument(
        "PuctSearch: IGameState::actionCount() must be positive");
  }
  validateParams();
  root_ = std::make_unique<SearchNode>(root_state_->currentPlayer());
}

PuctSearch::~PuctSearch() = default;

void PuctSearch::validateParams() const {
  auto require = [](bool condition, const char* message) {
    if (!condition) {
      throw std::invalid_argument(std::string("PuctSearch: ") + message);
    }
  };

  require(params_.maxVisits >= 0, "maxVisits must be >= 0");
  require(params_.maxPlayouts >= 0, "maxPlayouts must be >= 0");
  require(params_.maxTimeSeconds >= 0.0, "maxTimeSeconds must be >= 0");
  require(params_.maxDepth >= 0, "maxDepth must be >= 0");

  require(params_.cpuctExploration >= 0.0, "cpuctExploration must be >= 0");
  require(params_.cpuctExplorationLog >= 0.0,
          "cpuctExplorationLog must be >= 0");
  require(params_.cpuctExplorationBase > 0.0,
          "cpuctExplorationBase must be > 0");
  require(params_.cpuctUtilityStdevPrior > 0.0,
          "cpuctUtilityStdevPrior must be > 0");
  require(params_.cpuctUtilityStdevPriorWeight > 0.0,
          "cpuctUtilityStdevPriorWeight must be > 0");
  require(params_.cpuctUtilityStdevScale >= 0.0,
          "cpuctUtilityStdevScale must be >= 0");
  require(params_.utilityRadius > 0.0, "utilityRadius must be > 0");

  require(params_.fpuReductionMax >= 0.0, "fpuReductionMax must be >= 0");
  require(params_.fpuLossProp >= 0.0 && params_.fpuLossProp <= 1.0,
          "fpuLossProp must be in [0, 1]");
  require(params_.rootFpuReductionMax >= 0.0,
          "rootFpuReductionMax must be >= 0");
  require(params_.rootFpuLossProp >= 0.0 && params_.rootFpuLossProp <= 1.0,
          "rootFpuLossProp must be in [0, 1]");
  require(params_.fpuParentWeightByVisitedPolicyPow > 0.0,
          "fpuParentWeightByVisitedPolicyPow must be > 0");
  require(params_.fpuParentWeight >= 0.0 && params_.fpuParentWeight <= 1.0,
          "fpuParentWeight must be in [0, 1]");

  require(params_.rootDirichletNoiseWeight >= 0.0 &&
              params_.rootDirichletNoiseWeight <= 1.0,
          "rootDirichletNoiseWeight must be in [0, 1]");
  if (params_.rootNoiseEnabled) {
    require(params_.rootDirichletNoiseTotalConcentration > 0.0,
            "rootDirichletNoiseTotalConcentration must be > 0 when noise is "
            "enabled");
  }
  require(params_.rootPolicyTemperature > 0.0,
          "rootPolicyTemperature must be > 0");

  require(params_.chosenMoveTemperature >= 0.0,
          "chosenMoveTemperature must be >= 0");
  require(params_.chosenMoveTemperatureOnlyBelowProb > 0.0 &&
              params_.chosenMoveTemperatureOnlyBelowProb <= 1.0,
          "chosenMoveTemperatureOnlyBelowProb must be in (0, 1]");
  require(params_.chosenMoveSubtract >= 0.0,
          "chosenMoveSubtract must be >= 0");
  require(params_.chosenMovePrune >= 0.0, "chosenMovePrune must be >= 0");
  require(params_.valueWeightExponent >= 0.0,
          "valueWeightExponent must be >= 0");
}

void PuctSearch::resetRoot(std::shared_ptr<IGameState> rootState) {
  if (!rootState) {
    throw std::invalid_argument("PuctSearch::resetRoot: null state");
  }
  if (rootState->actionCount() != action_count_) {
    throw std::invalid_argument(
        "PuctSearch::resetRoot: actionCount changed between states");
  }
  root_state_ = std::move(rootState);
  root_ = std::make_unique<SearchNode>(root_state_->currentPlayer());
  root_priors_.clear();
  root_prepared_ = false;
}

void PuctSearch::ensureRootPrepared() {
  if (!root_->isExpanded()) {
    std::unique_ptr<IGameState> state = root_state_->clone();
    expandNode(*root_, *state);
  }
  if (!root_prepared_) {
    root_priors_ = root_->priors();
    applyRootPolicyTemperature(root_priors_);
    if (params_.rootNoiseEnabled) {
      addRootDirichletNoise(root_priors_);
    }
    root_prepared_ = true;
  }
}

void PuctSearch::expandNode(SearchNode& node, const IGameState& state) {
  node.expanded_ = true;

  if (state.isTerminal()) {
    node.terminal_ = true;
    node.evaluator_utility_ = state.terminalUtility();
    if (!std::isfinite(node.evaluator_utility_)) {
      throw std::runtime_error(
          "PuctSearch: terminalUtility() returned a non-finite value");
    }
  } else {
    Evaluation evaluation = evaluator_->evaluate(state);
    if (!std::isfinite(evaluation.utility)) {
      throw std::runtime_error(
          "PuctSearch: evaluator returned a non-finite utility");
    }
    node.evaluator_utility_ = evaluation.utility;
    buildPriors(node, state, evaluation.policy);
  }

  // The evaluator's value counts as one visit's worth of weight, matching
  // KataGo's "own evaluation" term.
  node.stats_.visits_ += 1;
  recomputeNodeStats(node, &node == root_.get());
}

void PuctSearch::buildPriors(SearchNode& node, const IGameState& state,
                             const std::vector<float>& policy) {
  if (static_cast<int>(policy.size()) != action_count_) {
    throw std::runtime_error(
        "PuctSearch: evaluator policy size " +
        std::to_string(policy.size()) + " does not match actionCount() " +
        std::to_string(action_count_));
  }

  node.priors_.clear();
  double sum = 0.0;
  for (int action_index = 0; action_index < action_count_; ++action_index) {
    const float probability = policy[action_index];
    // NaN fails the > 0 test. Negative and zero entries are unavailable.
    if (!(probability > 0.0f)) {
      continue;
    }
    const ActionId action = static_cast<ActionId>(action_index);
    if (!state.isLegal(action)) {
      continue;
    }
    node.priors_.push_back(Prior{action, probability});
    sum += static_cast<double>(probability);
  }

  if (sum > 0.0) {
    for (Prior& prior : node.priors_) {
      prior.probability =
          static_cast<float>(static_cast<double>(prior.probability) / sum);
    }
  }
  node.edge_index_by_prior_.assign(node.priors_.size(), -1);
}

void PuctSearch::applyRootPolicyTemperature(std::vector<Prior>& priors) const {
  const double temperature = params_.rootPolicyTemperature;
  if (temperature == 1.0 || priors.empty()) {
    return;
  }

  double max_probability = 0.0;
  for (const Prior& prior : priors) {
    max_probability = std::max(max_probability,
                               static_cast<double>(prior.probability));
  }
  if (!(max_probability > 0.0)) {
    return;
  }

  if (temperature <= 1e-4) {
    // Zero-temperature limit: one-hot on the largest prior.
    std::size_t best = 0;
    for (std::size_t i = 1; i < priors.size(); ++i) {
      if (priors[i].probability > priors[best].probability) {
        best = i;
      }
    }
    for (std::size_t i = 0; i < priors.size(); ++i) {
      priors[i].probability = (i == best) ? 1.0f : 0.0f;
    }
    return;
  }

  const double log_max = std::log(max_probability);
  double sum = 0.0;
  for (Prior& prior : priors) {
    const double log_p = std::log(static_cast<double>(prior.probability));
    const double scaled =
        std::exp((log_p - log_max) / temperature);
    prior.probability = static_cast<float>(scaled);
    sum += scaled;
  }
  if (sum > 0.0) {
    for (Prior& prior : priors) {
      prior.probability =
          static_cast<float>(static_cast<double>(prior.probability) / sum);
    }
  }
}

void PuctSearch::addRootDirichletNoise(std::vector<Prior>& priors) {
  const double weight = params_.rootDirichletNoiseWeight;
  if (weight <= 0.0 || priors.empty()) {
    return;
  }

  const int legal_count = static_cast<int>(priors.size());
  if (legal_count <= 0) {
    return;
  }

  // KataGo's policy-shaped alpha distribution: half uniform, half shaped by
  // the log of the prior. The total concentration is distributed according to
  // these proportions.
  std::vector<double> alphas(priors.size(), 0.0);
  double log_policy_sum = 0.0;
  for (std::size_t i = 0; i < priors.size(); ++i) {
    const double p = std::max(0.01, static_cast<double>(priors[i].probability));
    alphas[i] = std::log(p + 1e-20);
    log_policy_sum += alphas[i];
  }
  const double log_policy_mean = log_policy_sum / legal_count;

  double alpha_prop_sum = 0.0;
  for (std::size_t i = 0; i < priors.size(); ++i) {
    alphas[i] = std::max(0.0, alphas[i] - log_policy_mean);
    alpha_prop_sum += alphas[i];
  }

  const double uniform = 1.0 / legal_count;
  if (alpha_prop_sum <= 0.0) {
    for (double& alpha : alphas) {
      alpha = uniform;
    }
  } else {
    for (double& alpha : alphas) {
      alpha = 0.5 * (alpha / alpha_prop_sum + uniform);
    }
  }

  std::vector<double> draws(priors.size(), 0.0);
  double draw_sum = 0.0;
  for (std::size_t i = 0; i < priors.size(); ++i) {
    const double shape =
        alphas[i] * params_.rootDirichletNoiseTotalConcentration;
    draws[i] = rng_.nextGamma(shape);
    draw_sum += draws[i];
  }
  if (!(draw_sum > 0.0)) {
    return;
  }

  for (std::size_t i = 0; i < priors.size(); ++i) {
    const double noise = draws[i] / draw_sum;
    const double original = static_cast<double>(priors[i].probability);
    priors[i].probability =
        static_cast<float>((1.0 - weight) * original + weight * noise);
  }
}

const std::vector<Prior>& PuctSearch::activePriors(
    const SearchNode& node) const {
  if (&node == root_.get() && root_prepared_) {
    return root_priors_;
  }
  return node.priors_;
}

double PuctSearch::computeExploreScaling(
    double total_child_weight, double utility_stdev_factor) const {
  const double base = params_.cpuctExplorationBase;
  const double cpuct =
      params_.cpuctExploration +
      params_.cpuctExplorationLog *
          std::log((total_child_weight + base) / base);
  return cpuct * std::sqrt(total_child_weight + kTotalChildWeightPuctOffset) *
         utility_stdev_factor;
}

double PuctSearch::computeFpuValue(const SearchNode& node, bool is_root,
                                   double policy_mass_visited,
                                   double& utility_stdev_factor) const {
  const NodeStats& stats = node.stats();
  const double parent_utility = stats.utilityAvg();

  const double prior = params_.cpuctUtilityStdevPrior;
  double parent_utility_stdev = prior;
  if (stats.visits() > 0 && stats.weightSum() > 1.0) {
    double utility_sq_avg = stats.utilitySqAvg();
    const double utility_sq = parent_utility * parent_utility;
    utility_sq_avg = std::max(utility_sq_avg, utility_sq);
    const double prior_variance = prior * prior;
    const double prior_weight = params_.cpuctUtilityStdevPriorWeight;
    parent_utility_stdev = std::sqrt(std::max(
        0.0,
        ((utility_sq + prior_variance) * prior_weight +
         utility_sq_avg * stats.weightSum()) /
                (prior_weight + stats.weightSum() - 1.0) -
            utility_sq));
  }

  utility_stdev_factor =
      1.0 + params_.cpuctUtilityStdevScale *
                (parent_utility_stdev / prior - 1.0);

  double parent_utility_for_fpu = parent_utility;
  if (params_.fpuParentWeightByVisitedPolicy) {
    const double mass =
        std::min(1.0, std::pow(std::max(0.0, policy_mass_visited),
                               params_.fpuParentWeightByVisitedPolicyPow));
    parent_utility_for_fpu =
        mass * parent_utility + (1.0 - mass) * node.evaluatorUtility();
  } else if (params_.fpuParentWeight > 0.0) {
    parent_utility_for_fpu =
        params_.fpuParentWeight * node.evaluatorUtility() +
        (1.0 - params_.fpuParentWeight) * parent_utility;
  }

  const double reduction_max =
      is_root ? params_.rootFpuReductionMax : params_.fpuReductionMax;
  const double loss_prop =
      is_root ? params_.rootFpuLossProp : params_.fpuLossProp;
  const double reduction =
      reduction_max * std::sqrt(std::max(0.0, policy_mass_visited));
  double fpu_value = parent_utility_for_fpu - reduction;
  const double loss_value = -params_.utilityRadius;
  fpu_value += (loss_value - fpu_value) * loss_prop;
  return fpu_value;
}

int PuctSearch::selectPrior(SearchNode& node, bool is_root) {
  const std::vector<Prior>& priors = activePriors(node);
  if (priors.empty()) {
    return -1;
  }

  double total_child_weight = 0.0;
  double policy_mass_visited = 0.0;
  for (std::size_t i = 0; i < priors.size(); ++i) {
    const int edge_index = node.edge_index_by_prior_[i];
    if (edge_index < 0) {
      continue;
    }
    const SearchNode::Edge& edge = node.edges_[edge_index];
    total_child_weight +=
        edge.child->stats().childWeight(edge.visits);
    policy_mass_visited += priors[i].probability;
  }

  double utility_stdev_factor = 1.0;
  const double fpu_value = computeFpuValue(
      node, is_root, policy_mass_visited, utility_stdev_factor);
  const double explore_scaling =
      computeExploreScaling(total_child_weight, utility_stdev_factor);

  double best_value = kIllegalSelectionValue;
  int best_prior_index = -1;

  for (std::size_t i = 0; i < priors.size(); ++i) {
    const int edge_index = node.edge_index_by_prior_[i];
    const double probability = static_cast<double>(priors[i].probability);

    if (edge_index < 0) {
      if (probability <= 0.0) {
        continue;
      }
      // Unvisited action: FPU value plus its exploration term.
      const double value = fpu_value + explore_scaling * probability;
      if (value > best_value) {
        best_value = value;
        best_prior_index = static_cast<int>(i);
      }
      continue;
    }

    const SearchNode::Edge& edge = node.edges_[edge_index];
    const NodeStats& child_stats = edge.child->stats();
    const std::int64_t child_visits = child_stats.visits();
    const double child_weight = child_stats.childWeight(edge.visits);

    double child_value = fpu_value;
    if (child_visits > 0 && child_weight > 0.0) {
      const double utility = child_stats.utilityAvg();
      child_value =
          (edge.child->player() == node.player()) ? utility : -utility;
    }
    // PUCT exploration term. With edge-visit weighting, a freshly expanded
    // child has zero weight and receives the full FPU exploration bonus.
    const double value =
        child_value + explore_scaling * probability / (1.0 + child_weight);
    if (value > best_value) {
      best_value = value;
      best_prior_index = static_cast<int>(i);
    }
  }

  return best_prior_index;
}

void PuctSearch::playout(SearchNode& node, IGameState& state, int depth) {
  if (!node.expanded_) {
    expandNode(node, state);
    return;
  }

  if (node.terminal_) {
    addOwnEvaluationAsLeaf(node);
    return;
  }

  if (params_.maxDepth > 0 && depth >= params_.maxDepth) {
    addOwnEvaluationAsLeaf(node);
    return;
  }

  const bool is_root = (&node == root_.get());
  const int prior_index = selectPrior(node, is_root);
  if (prior_index < 0) {
    // Dead end with no searchable action: treat the node's own evaluation as
    // the leaf value for this playout.
    addOwnEvaluationAsLeaf(node);
    return;
  }

  const std::vector<Prior>& priors = activePriors(node);
  const ActionId action = priors[prior_index].action;
  int edge_index = node.edge_index_by_prior_[prior_index];

  if (edge_index < 0) {
    state.applyAction(action);
    auto child = std::make_unique<SearchNode>(state.currentPlayer());
    node.edges_.push_back(
        SearchNode::Edge{priors[prior_index], std::move(child), 0});
    node.edge_index_by_prior_[prior_index] =
        static_cast<int>(node.edges_.size()) - 1;
    edge_index = static_cast<int>(node.edges_.size()) - 1;

    SearchNode& child_node = *node.edges_[edge_index].child;
    playout(child_node, state, depth + 1);
    node.edges_[edge_index].visits += 1;
  } else {
    SearchNode::Edge& edge = node.edges_[edge_index];
    state.applyAction(action);
    playout(*edge.child, state, depth + 1);
    edge.visits += 1;
  }

  node.stats_.visits_ += 1;
  recomputeNodeStats(node, is_root);
}

void PuctSearch::addOwnEvaluationAsLeaf(SearchNode& node) {
  const double utility = node.evaluator_utility_;
  node.stats_.visits_ += 1;
  node.stats_.weight_sum_ += 1.0;
  node.stats_.utility_sum_ += utility;
  node.stats_.utility_sq_sum_ += utility * utility;
}

void PuctSearch::recomputeNodeStats(SearchNode& node, bool is_root) {
  struct ChildValueInternal {
    double weight = 0.0;
    double original_weight = 0.0;
    double utility = 0.0;
    std::int64_t visits = 0;
  };

  std::vector<ChildValueInternal> children;
  children.reserve(node.edges_.size());

  double original_total_weight = 0.0;
  for (const SearchNode::Edge& edge : node.edges_) {
    const NodeStats& child_stats = edge.child->stats();
    if (child_stats.visits() <= 0 || !(child_stats.weightSum() > 0.0) ||
        edge.visits <= 0) {
      continue;
    }
    const double weight = child_stats.childWeight(edge.visits);
    const double utility =
        (edge.child->player() == node.player())
            ? child_stats.utilityAvg()
            : -child_stats.utilityAvg();
    children.push_back(
        ChildValueInternal{weight, weight, utility, child_stats.visits()});
    original_total_weight += weight;
  }

  double total_child_weight = original_total_weight;

  if (!children.empty() && original_total_weight > 0.0) {
    // Root-only subtract/prune used together with Dirichlet noise, mirroring
    // KataGo's chosenMoveSubtract / chosenMovePrune.
    double amount_to_subtract = 0.0;
    double amount_to_prune = 0.0;
    if (is_root && params_.rootNoiseEnabled) {
      double max_child_weight = 0.0;
      for (const ChildValueInternal& child : children) {
        max_child_weight = std::max(max_child_weight, child.weight);
      }
      amount_to_subtract =
          std::min(params_.chosenMoveSubtract, max_child_weight / 64.0);
      amount_to_prune =
          std::min(params_.chosenMovePrune, max_child_weight / 64.0);
    }

    // KataGo computes the "simple value" and per-child standard deviations
    // from pre-subtraction weights.
    double simple_value = 0.0;
    for (const ChildValueInternal& child : children) {
      simple_value += child.utility * child.original_weight;
    }
    simple_value /= original_total_weight;

    if (amount_to_subtract > 0.0 || amount_to_prune > 0.0) {
      total_child_weight = 0.0;
      for (ChildValueInternal& child : children) {
        if (child.weight < amount_to_prune) {
          child.weight = 0.0;
        } else {
          child.weight = std::max(0.0, child.weight - amount_to_subtract);
        }
        total_child_weight += child.weight;
      }
    }

    if (params_.valueWeightExponent != 0.0 && total_child_weight > 0.0) {
      double unnormalized_total = 0.0;
      for (ChildValueInternal& child : children) {
        if (child.visits <= 0 || child.weight <= 0.0) {
          continue;
        }
        const double precision =
            1.5 * std::sqrt(std::max(child.original_weight, 1e-30));
        const double stdev =
            std::sqrt(kValueWeightingVarianceFloor + 1.0 / precision);
        const double p =
            normalCdf((child.utility - simple_value) / stdev) + 0.0001;
        child.weight *= std::pow(p, params_.valueWeightExponent);
        unnormalized_total += child.weight;
      }
      if (unnormalized_total > 0.0) {
        const double factor = total_child_weight / unnormalized_total;
        for (ChildValueInternal& child : children) {
          child.weight *= factor;
        }
      } else {
        total_child_weight = 0.0;
      }
    }
  } else {
    total_child_weight = 0.0;
  }

  // Own evaluation always contributes with weight 1 (KataGo's uncertainty
  // weighting is intentionally not ported).
  const double own_utility = node.evaluator_utility_;
  double utility_sum = own_utility;
  double utility_sq_sum = own_utility * own_utility;
  double weight_sum = 1.0;

  for (const ChildValueInternal& child : children) {
    if (child.weight <= 0.0) {
      continue;
    }
    utility_sum += child.weight * child.utility;
    utility_sq_sum += child.weight * child.utility * child.utility;
    weight_sum += child.weight;
  }

  node.stats_.weight_sum_ = weight_sum;
  node.stats_.utility_sum_ = utility_sum;
  node.stats_.utility_sq_sum_ = utility_sq_sum;
}

bool PuctSearch::advanceRoot(ActionId action) {
  for (std::size_t i = 0; i < root_->priors_.size(); ++i) {
    if (root_->priors_[i].action != action) {
      continue;
    }
    const int edge_index = root_->edge_index_by_prior_[i];
    if (edge_index < 0) {
      return false;
    }

    std::unique_ptr<IGameState> new_state = root_state_->clone();
    new_state->applyAction(action);

    std::unique_ptr<SearchNode> new_root =
        std::move(root_->edges_[edge_index].child);
    if (!new_root) {
      return false;
    }

    root_state_ = std::move(new_state);
    root_ = std::move(new_root);
    root_priors_.clear();
    root_prepared_ = false;
    return true;
  }
  return false;
}

bool PuctSearch::limitsReached(
    std::int64_t playouts,
    std::chrono::steady_clock::time_point start) const {
  if (params_.maxPlayouts > 0 && playouts >= params_.maxPlayouts) {
    return true;
  }
  if (params_.maxVisits > 0 &&
      root_->stats().visits() >= params_.maxVisits) {
    return true;
  }
  if (params_.maxTimeSeconds > 0.0 &&
      elapsedSeconds(start) >= params_.maxTimeSeconds) {
    return true;
  }
  return false;
}

SearchResult PuctSearch::runImpl(
    IStopCriterion* criterion,
    const std::function<bool(const SearchProgress&)>* callback) {
  ensureRootPrepared();

  if (root_->isTerminal() || root_->priors().empty()) {
    return buildResult(0, true);
  }

  const auto start = std::chrono::steady_clock::now();
  std::int64_t playouts = 0;
  bool completed = true;

  for (;;) {
    if (limitsReached(playouts, start)) {
      break;
    }

    const SearchProgress progress{root_->stats().visits(), playouts,
                                  elapsedSeconds(start)};
    if ((criterion != nullptr && criterion->shouldStop(progress)) ||
        (callback != nullptr && (*callback)(progress))) {
      completed = false;
      break;
    }

    std::unique_ptr<IGameState> state = root_state_->clone();
    playout(*root_, *state, 0);
    ++playouts;
  }

  return buildResult(playouts, completed);
}

SearchResult PuctSearch::run() {
  return runImpl(nullptr, nullptr);
}

SearchResult PuctSearch::run(IStopCriterion* criterion) {
  return runImpl(criterion, nullptr);
}

SearchResult PuctSearch::run(
    const std::function<bool(const SearchProgress&)>& criterion) {
  return runImpl(nullptr, &criterion);
}

SearchResult PuctSearch::buildResult(std::int64_t playouts,
                                     bool completed) const {
  SearchResult result;
  result.rootVisits = root_->stats().visits();
  result.playouts = playouts;
  result.completed = completed;
  result.rootUtility = root_->stats().utilityAvg();

  const std::vector<Prior>& priors =
      root_prepared_ ? root_priors_ : root_->priors();

  std::int64_t total_visits = 0;
  ActionId best_action = kInvalidAction;
  std::int64_t best_visits = -1;
  float best_prior = -1.0f;

  for (std::size_t i = 0; i < priors.size(); ++i) {
    ActionStats stats;
    stats.action = priors[i].action;
    stats.prior = priors[i].probability;

    const auto& node_priors = root_->priors();
    std::int64_t edge_visits = 0;
    if (i < node_priors.size() &&
        node_priors[i].action == stats.action) {
      const int edge_index = root_->edge_index_by_prior_[i];
      if (edge_index >= 0) {
        const SearchNode::Edge& edge = root_->edges_[edge_index];
        edge_visits = edge.visits;
        const NodeStats& child_stats = edge.child->stats();
        if (child_stats.visits() > 0 && child_stats.weightSum() > 0.0) {
          const double utility = child_stats.utilityAvg();
          stats.qValue = (edge.child->player() == root_->player())
                             ? utility
                             : -utility;
        } else {
          stats.qValue = std::numeric_limits<double>::quiet_NaN();
        }
      } else {
        stats.qValue = std::numeric_limits<double>::quiet_NaN();
      }
    } else {
      stats.qValue = std::numeric_limits<double>::quiet_NaN();
    }

    stats.visits = edge_visits;
    total_visits += edge_visits;
    result.actions.push_back(stats);

    if (stats.visits > best_visits ||
        (stats.visits == best_visits && stats.prior > best_prior)) {
      best_visits = stats.visits;
      best_prior = stats.prior;
      best_action = stats.action;
    }
  }

  if (total_visits > 0) {
    for (ActionStats& stats : result.actions) {
      stats.visitShare =
          static_cast<double>(stats.visits) /
          static_cast<double>(total_visits);
    }
  }

  result.bestAction = best_action;
  return result;
}

std::vector<double> SearchResult::denseVisitDistribution(
    int actionCount) const {
  std::vector<double> distribution(
      actionCount > 0 ? static_cast<std::size_t>(actionCount) : 0, 0.0);
  std::int64_t total = 0;
  for (const ActionStats& stats : actions) {
    total += stats.visits;
  }
  if (total <= 0) {
    return distribution;
  }
  for (const ActionStats& stats : actions) {
    if (stats.action >= 0 &&
        stats.action < static_cast<ActionId>(distribution.size())) {
      distribution[static_cast<std::size_t>(stats.action)] +=
          static_cast<double>(stats.visits) / static_cast<double>(total);
    }
  }
  return distribution;
}

ActionId PuctSearch::sampleWithTemperature(
    const std::vector<ActionId>& actions,
    const std::vector<double>& values, double temperature,
    double only_below_prob) {
  if (actions.empty() || actions.size() != values.size()) {
    return kInvalidAction;
  }

  double max_value = 0.0;
  double sum_value = 0.0;
  std::size_t best_index = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    const double value = std::max(0.0, values[i]);
    sum_value += value;
    if (values[i] > max_value) {
      max_value = values[i];
      best_index = i;
    }
  }
  if (!(max_value > 0.0) || !(sum_value > 0.0)) {
    return actions[0];
  }

  if (temperature <= 1.0e-4 && only_below_prob >= 1.0) {
    return actions[best_index];
  }

  const double log_max = std::log(max_value);
  const double log_sum = std::log(sum_value);
  const double log_only_below = std::log(std::max(1e-50, only_below_prob));
  const double threshold =
      std::min(0.0, log_only_below + log_sum - log_max);

  std::vector<double> processed(values.size(), 0.0);
  double total = 0.0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (values[i] <= 0.0) {
      continue;
    }
    const double log_relative = std::log(values[i]) - log_max;
    double new_log_relative = log_relative;
    if (log_relative <= threshold) {
      new_log_relative =
          (log_relative - threshold) / temperature + threshold;
    }
    processed[i] = std::exp(new_log_relative);
    total += processed[i];
  }
  if (!(total > 0.0)) {
    return actions[best_index];
  }

  const double target = rng_.nextDouble() * total;
  double cumulative = 0.0;
  for (std::size_t i = 0; i < processed.size(); ++i) {
    cumulative += processed[i];
    if (target < cumulative) {
      return actions[i];
    }
  }
  return actions.back();
}

ActionId PuctSearch::chooseAction(double temperature) {
  return chooseAction(temperature,
                      params_.chosenMoveTemperatureOnlyBelowProb);
}

ActionId PuctSearch::chooseAction(double temperature, double onlyBelowProb) {
  ensureRootPrepared();

  // Prefer expanded children, weighted by KataGo's child weight.
  std::vector<ActionId> actions;
  std::vector<double> values;
  double max_value = 0.0;

  for (std::size_t i = 0; i < root_priors_.size(); ++i) {
    const int edge_index = root_->edge_index_by_prior_[i];
    if (edge_index < 0) {
      continue;
    }
    if (root_priors_[i].probability <= 0.0f) {
      continue;
    }
    const SearchNode::Edge& edge = root_->edges_[edge_index];
    const double weight = edge.child->stats().childWeight(edge.visits);
    actions.push_back(root_priors_[i].action);
    values.push_back(weight);
    max_value = std::max(max_value, weight);
  }

  // No expanded child: fall back to the root prior (KataGo's direct policy
  // fallback for an under-searched root).
  if (actions.empty()) {
    for (const Prior& prior : root_priors_) {
      if (prior.probability <= 0.0f) {
        continue;
      }
      actions.push_back(prior.action);
      values.push_back(prior.probability);
      max_value = std::max(max_value, static_cast<double>(prior.probability));
    }
  }

  if (actions.empty()) {
    return kInvalidAction;
  }

  // KataGo's chosenMoveSubtract / chosenMovePrune, capped relative to the
  // largest weight.
  const double amount_to_subtract =
      std::min(params_.chosenMoveSubtract, max_value / 64.0);
  const double amount_to_prune =
      std::min(params_.chosenMovePrune, max_value / 64.0);
  for (double& value : values) {
    if (value < amount_to_prune) {
      value = 0.0;
    } else {
      value = std::max(0.0, value - amount_to_subtract);
    }
  }

  return sampleWithTemperature(actions, values, temperature, onlyBelowProb);
}

ActionId PuctSearch::chooseAction() {
  return chooseAction(params_.chosenMoveTemperature);
}

}  // namespace suzero::mcts
