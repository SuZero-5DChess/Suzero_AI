// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-5DChess contributors
//
// This file is part of SuZero AI, a 5D chess AlphaZero project. The search
// core is derived from KataGo (https://github.com/lightvector/KataGo).
// See THIRD_PARTY_NOTICES.md.

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "suzero/mcts/evaluator.h"
#include "suzero/mcts/game_state.h"
#include "suzero/mcts/random.h"
#include "suzero/mcts/search_node.h"
#include "suzero/mcts/search_params.h"
#include "suzero/mcts/types.h"

namespace suzero::mcts {

/// Progress information passed to external stop criteria.
struct SearchProgress {
  std::int64_t rootVisits = 0;
  std::int64_t playouts = 0;
  double elapsedSeconds = 0.0;
};

/// Abstract cooperative stop criterion, checked once per playout.
class IStopCriterion {
 public:
  virtual ~IStopCriterion() = default;
  virtual bool shouldStop(const SearchProgress& progress) = 0;
};

/// Per-action statistics returned at the root.
struct ActionStats {
  ActionId action = kInvalidAction;

  /// Effective root prior after policy temperature and optional noise.
  float prior = 0.0f;

  /// Number of edge visits accumulated at the root for this action.
  std::int64_t visits = 0;

  /// Average child value from the root player's perspective. NaN when the
  /// action has no expanded child.
  double qValue = 0.0;

  /// Fraction of root edge visits (0 when the root has no visits).
  double visitShare = 0.0;
};

/// Result of a search run.
struct SearchResult {
  /// Every legal, positively-weighted root action, in ascending action order.
  std::vector<ActionStats> actions;

  /// Action with the largest final play-selection weight (KataGo's
  /// getChosenMoveLoc semantics, without LCB). kInvalidAction when the root is
  /// terminal or has no legal action.
  ActionId bestAction = kInvalidAction;

  /// Root value from the root player's perspective.
  double rootUtility = 0.0;

  /// Total visits at the root (includes visits from tree reuse).
  std::int64_t rootVisits = 0;

  /// Playouts performed by this run.
  std::int64_t playouts = 0;

  /// False only when an IStopCriterion stopped the run before a configured
  /// limit was reached.
  bool completed = true;

  /// Dense visit distribution of size `actionCount`, normalized to sum 1.
  /// Returns all zeros when there are no visits.
  std::vector<double> denseVisitDistribution(int actionCount) const;
};

/// PUCT graph search following KataGo's search semantics.
///
/// This implementation is single-threaded and performs graph search
/// (transposition-aware) rather than pure tree search: states with the same
/// IGameState::stateKey() share one node, edges reference their child instead
/// of owning it, and the root is deliberately kept out of the node table so it
/// can never be reached through a cycle. It keeps the algorithmic core of
/// KataGo's search (PUCT with child weight, FPU, virtual-loss-free playouts,
/// edge-visit catch-up, Student-t weighted value aggregation, root Dirichlet
/// noise, retrospective root play-selection weighting, temperature sampling)
/// but deliberately omits evaluation caching, multithreading, Go's score
/// utilities, and symmetry/pass handling. See docs/katago-mapping.md.
///
/// All methods are single-threaded and must not be called concurrently on the
/// same instance.
class PuctSearch {
 public:
  /// Takes ownership of shared state/evaluator references and builds a tree
  /// with one root node. `params` and `seed` determine all later randomness.
  PuctSearch(std::shared_ptr<IGameState> rootState,
             std::shared_ptr<IEvaluator> evaluator,
             SearchParams params = SearchParams(),
             std::uint64_t seed = kDefaultSeed);

  ~PuctSearch();

  PuctSearch(const PuctSearch&) = delete;
  PuctSearch& operator=(const PuctSearch&) = delete;
  PuctSearch(PuctSearch&&) = delete;
  PuctSearch& operator=(PuctSearch&&) = delete;

  /// Runs playouts until a configured limit is reached or the criterion says
  /// stop. Calling it again continues from the current tree.
  SearchResult run();
  SearchResult run(IStopCriterion* criterion);
  SearchResult run(const std::function<bool(const SearchProgress&)>& criterion);

  /// Overrides the root Dirichlet-noise parameters of `params()`.
  ///
  /// Takes effect the next time a root prior is prepared (a fresh root or one
  /// promoted by `advanceRoot`); an already prepared root keeps its priors.
  /// Self-play uses this to apply noise only to full-search moves.
  void setRootNoise(bool enabled, double totalConcentration, double weight);

  /// Samples a move from root child weights using the given temperature.
  /// Falls back to the root prior when no child has been expanded.
  ActionId chooseAction(double temperature);
  ActionId chooseAction(double temperature, double onlyBelowProb);

  /// Samples using `params().chosenMoveTemperature`.
  ActionId chooseAction();

  /// Reuses the subtree of `action` as the new root, if that action has an
  /// expanded child. The root game state is advanced accordingly.
  /// The subtree is copied and the node table is rebuilt (single-threaded
  /// equivalent of KataGo's copy-and-sweep), so every SearchNode pointer
  /// obtained before this call is invalidated.
  /// Returns false (and changes nothing) otherwise.
  bool advanceRoot(ActionId action);

  /// Replaces the root state and clears the graph.
  void resetRoot(std::shared_ptr<IGameState> rootState);

  const IGameState& rootState() const { return *root_state_; }
  const SearchNode& rootNode() const { return *root_; }
  const SearchParams& params() const { return params_; }
  int actionCount() const { return action_count_; }

  /// Number of live nodes, including the root. Transpositions reduce this
  /// below what a pure tree would contain.
  std::int64_t nodeCount() const {
    return static_cast<std::int64_t>(node_table_.size()) + 1;
  }

  /// RNG used by noise and temperature sampling. Exposed for reproducibility
  /// and tests.
  Rng& rng() { return rng_; }

 private:
  struct ChildValue {
    double weight = 0.0;
    double utility = 0.0;
    std::int64_t visits = 0;
  };

  struct NodeLookupResult {
    SearchNode* node = nullptr;
    bool inserted = false;
  };

  using PathSet = std::unordered_set<const SearchNode*>;

  void validateParams() const;
  void ensureRootPrepared();
  void expandNode(SearchNode& node, const IGameState& state);
  void buildPriors(SearchNode& node, const IGameState& state,
                   const std::vector<float>& policy);
  NodeLookupResult allocateOrFindNode(const StateKey& key, PlayerId player);
  void playout(SearchNode& node, IGameState& state, int depth,
               PathSet* path);
  bool maybeCatchUpEdgeVisits(SearchNode& node, int edge_index, bool is_root);
  int selectPrior(SearchNode& node, bool is_root);
  double computeFpuValue(const SearchNode& node, bool is_root,
                         double policy_mass_visited,
                         double& utility_stdev_factor) const;
  double computeExploreScaling(double total_child_weight,
                               double utility_stdev_factor) const;
  void recomputeNodeStats(SearchNode& node, bool is_root);
  void addOwnEvaluationAsLeaf(SearchNode& node);
  void applyRootPolicyTemperature(std::vector<Prior>& priors) const;
  void addRootDirichletNoise(std::vector<Prior>& priors);
  const std::vector<Prior>& activePriors(const SearchNode& node) const;
  bool computeRootPlaySelectionValues(std::vector<ActionId>& actions,
                                      std::vector<double>& values) const;
  double interpolateEarly(double halflife, double early_value,
                          double value) const;
  ActionId sampleWithTemperature(const std::vector<ActionId>& actions,
                                 const std::vector<double>& values,
                                 double temperature,
                                 double only_below_prob);
  SearchResult buildResult(std::int64_t playouts, bool completed) const;
  bool limitsReached(std::int64_t playouts,
                     std::chrono::steady_clock::time_point start) const;
  SearchResult runImpl(IStopCriterion* criterion,
                       const std::function<bool(const SearchProgress&)>*
                           callback);
  void clearGraph();
  std::unique_ptr<SearchNode> copySubtree(
      const SearchNode& subtree_root,
      std::unordered_map<StateKey, std::unique_ptr<SearchNode>>& table) const;

  std::shared_ptr<IGameState> root_state_;
  std::shared_ptr<IEvaluator> evaluator_;
  SearchParams params_;
  int action_count_ = 0;

  // The root is owned directly and is deliberately not in node_table_, so no
  // transposition can ever point back to it and create a cycle through it.
  std::unique_ptr<SearchNode> root_;
  std::vector<Prior> root_priors_;
  bool root_prepared_ = false;

  // Owns every non-root node. Edges only hold raw pointers.
  std::unordered_map<StateKey, std::unique_ptr<SearchNode>> node_table_;

  Rng rng_;
};

}  // namespace suzero::mcts
