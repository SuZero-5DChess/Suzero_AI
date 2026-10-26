// SPDX-License-Identifier: MIT
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

#include "5dchess/tree_positions.h"
#include "suzero/mcts/puct_search.h"

namespace {
using namespace suzero::chess5d;
using namespace suzero::mcts;
int failures = 0;
int checks = 0;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; \
  std::cerr << __LINE__ << ": " << #x << "\n"; } } while (false)

struct Position {
  PlayerId player;
  std::vector<int> children;
  bool submitOnly = false;
  double utility = 0.3;
};
using Graph = std::vector<Position>;

// Deliberately supports repeated players, transpositions, and cycles. Its model
// input makes helper tests independent of ONNX and the concrete native engine.
class State final : public IGameState, public IChess5DModelInput {
 public:
  inline static int cloneCalls = 0;
  State(std::shared_ptr<Graph> graph, int id = 0) : graph_(std::move(graph)), id_(id) {
    refresh();
  }
  std::unique_ptr<IGameState> clone() const override {
    ++cloneCalls;
    return std::make_unique<State>(*this);
  }
  int actionCount() const override { return static_cast<int>(actions_.size()); }
  bool isLegal(ActionId action) const override {
    return action >= 0 && action < actionCount();
  }
  void applyAction(ActionId action) override {
    id_ = graph_->at(id_).children.at(action);
    refresh();
  }
  PlayerId currentPlayer() const override { return graph_->at(id_).player; }
  bool isTerminal() const override { return graph_->at(id_).children.empty(); }
  double terminalUtility() const override { return graph_->at(id_).utility; }
  StateKey stateKey() const override { return std::to_string(id_); }
  int moveCount() const override { return id_; }
  const std::vector<Action>& actions() const override { return actions_; }
  EncodedPosition encodePosition() const override {
    EncodedPosition result;
    result.side_to_move = currentPlayer();
    result.urgency = static_cast<float>(id_);
    return result;
  }
  double utility() const { return graph_->at(id_).utility; }
 private:
  void refresh() {
    actions_.assign(graph_->at(id_).children.size(), Action{});
    for (int i = 0; i < actionCount(); ++i) {
      actions_[i].is_submit = graph_->at(id_).submitOnly;
      actions_[i].move[0] = id_;
      actions_[i].move[1] = i;
    }
  }
  std::shared_ptr<Graph> graph_;
  int id_;
  std::vector<Action> actions_;
};

class Evaluator final : public IEvaluator {
 public:
  int calls = 0;
  Evaluation evaluate(const IGameState& state) override {
    ++calls;
    return {std::vector<float>(state.actionCount(), 1.0f),
            dynamic_cast<const State&>(state).utility()};
  }
};

SelfPlayBridgeConfig config() {
  SelfPlayBridgeConfig c;
  c.recordTreePositions = true;
  c.recordTreeMinVisits = 1;
  c.recordTreeMaxDepth = 5;
  c.recordTreeMaxPerTurn = 100;
  c.recordTreeSampleProbability = 1.0;
  return c;
}
SearchParams params(int playouts = 3000) {
  SearchParams p;
  p.maxPlayouts = playouts;
  p.maxDepth = 16;
  p.cpuctExploration = 20.0;
  p.rootDirichletNoiseWeight = 0.0;
  return p;
}
int bestAction(const SearchNode& node) {
  int best = -1;
  std::int64_t count = -1;
  for (int i = 0; i < static_cast<int>(node.priors().size()); ++i) {
    const int action = node.priors()[i].action;
    if (node.edgeVisits(i) > count ||
        (node.edgeVisits(i) == count && action < best)) {
      best = action;
      count = node.edgeVisits(i);
    }
  }
  return best;
}
std::int64_t outgoing(const SearchNode& node) {
  std::int64_t result = 0;
  for (int i = 0; i < static_cast<int>(node.priors().size()); ++i) {
    result += node.edgeVisits(i);
  }
  return result;
}
const SearchNode* findNode(const SearchNode& node, int id, std::set<StateKey>& seen) {
  if (!seen.insert(node.stateKey()).second) return nullptr;
  if (node.stateKey() == std::to_string(id)) return &node;
  for (int i = 0; i < static_cast<int>(node.priors().size()); ++i) {
    const auto* child = node.childAt(i);
    if (!child) continue;
    if (const auto* found = findNode(*child, id, seen)) return found;
  }
  return nullptr;
}

void testConfig() {
  SelfPlayBridgeConfig defaults;
  CHECK(defaults.recordTreePositions);
  CHECK(defaults.recordTreeMinVisits == 64);
  CHECK(defaults.recordTreeMaxDepth == 5);
  CHECK(defaults.recordTreeMaxPerTurn == 4);
  CHECK(defaults.recordTreeSampleProbability == 0.25);
  internal::validateTreePositionConfig(defaults);
  for (int field = 0; field < 8; ++field) {
    auto c = defaults;
    switch (field) {
      case 0: c.recordTreeMinVisits = 0; break;
      case 1: c.recordTreeMaxPerTurn = 0; break;
      case 2: c.recordTreeMaxDepth = 0; break;
      case 3: c.recordTreeMaxDepth = 65; break;
      case 4: c.recordTreeSampleProbability = -0.01; break;
      case 5: c.recordTreeSampleProbability = 1.01; break;
      case 6: c.recordTreeSampleProbability = std::numeric_limits<double>::infinity(); break;
      case 7: c.recordTreeSampleProbability = std::numeric_limits<double>::quiet_NaN(); break;
    }
    bool rejected = false;
    try { internal::validateTreePositionConfig(c); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
  }
}

void testSmallStepsAndDag() {
  auto graph = std::make_shared<Graph>(Graph{
      {0, {1, 2}}, {0, {3, 4}}, {0, {5, 6}},
      {1, {7, 8}}, {1, {7, 8}}, {1, {7, 8}}, {1, {7, 8}},
      {1, {9, 10}}, {0, {9, 10}}, {0, {}}, {0, {}}});
  auto state = std::make_shared<State>(graph);
  auto evaluator = std::make_shared<Evaluator>();
  PuctSearch search(state, evaluator, params(), 19);
  search.run();
  auto c = config();
  const int played = bestAction(search.rootNode());
  const int excluded = played == 0 ? 1 : 2;
  const int allowed = played == 0 ? 2 : 1;
  const int calls = evaluator->calls;
  const auto rootVisits = search.rootNode().stats().visits();
  const auto key = state->stateKey();
  Rng rng(42);
  const auto rows = internal::extractTreePositions(c, *state, search.rootNode(), played, rng);
  CHECK(!rows.empty());
  CHECK(evaluator->calls == calls);
  CHECK(search.rootNode().stats().visits() == rootVisits);
  CHECK(state->stateKey() == key);
  std::set<int> ids;
  for (const auto& row : rows) {
    const int id = row.moveCount;
    CHECK(ids.insert(id).second);  // Shared DAG children appear once.
    CHECK(id != 0 && id != excluded && id != allowed);
    CHECK(id != 8 && id < 9);  // White made a non-best move: no White rows.
    CHECK(row.player == 1 && row.encoded.side_to_move == row.player);
    CHECK(row.source == 1);
    CHECK(static_cast<int>(row.encoded.urgency) == id);
    CHECK(row.actions == State(graph, id).actions());
    CHECK(std::abs(std::accumulate(row.policy.begin(), row.policy.end(), 0.0) - 1.0) < 1e-6);
    std::set<StateKey> seen;
    const auto* node = findNode(search.rootNode(), id, seen);
    CHECK(node != nullptr);
    if (!node) continue;
    CHECK(row.searchVisits == node->stats().visits());
    CHECK(std::abs(row.value - node->stats().utilityAvg()) < 1e-6);
    for (int i = 0; i < static_cast<int>(node->priors().size()); ++i) {
      CHECK(std::abs(row.policy[node->priors()[i].action] -
            static_cast<double>(node->edgeVisits(i)) / outgoing(*node)) < 1e-6);
    }
    // Only grandchildren of the unplayed branch, or the shared black child.
    CHECK(id == (allowed == 1 ? 3 : 5) ||
          id == (allowed == 1 ? 4 : 6) || id == 7);
  }
  c.recordTreeMaxDepth = 1;
  CHECK(internal::extractTreePositions(c, *state, search.rootNode(), played, rng).empty());
  c.recordTreeMaxDepth = 2;
  for (const auto& row : internal::extractTreePositions(c, *state, search.rootNode(), played, rng)) {
    CHECK(row.moveCount >= 3 && row.moveCount <= 6);
  }
  c.recordTreeMaxDepth = 5;
  c.recordTreeMaxPerTurn = 1;
  CHECK(internal::extractTreePositions(c, *state, search.rootNode(), played, rng).size() == 1);
  c.recordTreeMinVisits = 1000000;
  CHECK(internal::extractTreePositions(c, *state, search.rootNode(), played, rng).empty());
  c = config();
  c.recordTreePositions = false;
  CHECK(internal::extractTreePositions(c, *state, search.rootNode(), played, rng).empty());
  c.recordTreePositions = true;
  c.recordTreeSampleProbability = 0.0;
  CHECK(internal::extractTreePositions(c, *state, search.rootNode(), played, rng).empty());
  c.recordTreeSampleProbability = 0.5;
  Rng first(123), second(123);
  const auto a = internal::extractTreePositions(c, *state, search.rootNode(), played, first);
  const auto b = internal::extractTreePositions(c, *state, search.rootNode(), played, second);
  CHECK(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) CHECK(a[i].moveCount == b[i].moveCount);
  // Extraction has not advanced PUCT's RNG.
  PuctSearch control(std::make_shared<State>(graph), evaluator, params(), 19);
  control.run();
  CHECK(search.rng().nextUint64() == control.rng().nextUint64());
}

void testBestMoveTieAndSamePlayer() {
  auto graph = std::make_shared<Graph>(Graph{
      {0, {1, 2}, false, 0.0}, {0, {3}, false, 0.0},
      {0, {4}, false, 0.0}, {1, {5}, false, 0.0},
      {1, {5}, false, 0.0}, {0, {}, false, 0.0}});
  auto state = std::make_shared<State>(graph);
  auto evaluator = std::make_shared<Evaluator>();
  PuctSearch search(state, evaluator, params(100), 4);
  search.run();
  CHECK(search.rootNode().edgeVisits(0) == search.rootNode().edgeVisits(1));
  auto c = config();
  c.recordTreeMaxDepth = 1;
  Rng rng(4);
  const auto bestBranch = internal::extractTreePositions(c, *state, search.rootNode(), 1, rng);
  CHECK(bestBranch.size() == 1);
  if (!bestBranch.empty()) CHECK(bestBranch.front().moveCount == 1);
  const auto nonbestBranch = internal::extractTreePositions(c, *state, search.rootNode(), 0, rng);
  CHECK(nonbestBranch.empty());
}

void testLeafAndSubmitOnly() {
  // Both root moves switch players, so both children would be eligible if
  // they had actual decision policies. One is terminal, one is forced SUBMIT.
  auto graph = std::make_shared<Graph>(Graph{
      {0, {1, 2}}, {1, {}}, {1, {3}, true}, {0, {}}});
  auto state = std::make_shared<State>(graph);
  auto evaluator = std::make_shared<Evaluator>();
  PuctSearch search(state, evaluator, params(), 5);
  search.run();
  Rng rng(1);
  auto c = config();
  CHECK(internal::extractTreePositions(c, *state, search.rootNode(), 0, rng).empty());
  CHECK(internal::extractTreePositions(c, *state, search.rootNode(), 1, rng).empty());
  // Just-expanded nonterminal leaves have incoming visits but zero outgoing
  // visits. A prior-only target must never be manufactured for them.
  graph = std::make_shared<Graph>(Graph{{0, {1}}, {1, {2}}, {0, {}}});
  state = std::make_shared<State>(graph);
  PuctSearch leafSearch(state, evaluator, params(1), 5);
  leafSearch.run();
  const auto* child = leafSearch.rootNode().childAt(0);
  CHECK(child && child->isExpanded() && !child->isTerminal());
  CHECK(child && outgoing(*child) == 0);
  CHECK(internal::extractTreePositions(c, *state, leafSearch.rootNode(), kInvalidAction, rng).empty());
}

void testLocalVisitQualityGuards() {
  auto graph = std::make_shared<Graph>(Graph{
      {0, {1}}, {1, {2}}, {0, {}}});
  auto state = std::make_shared<State>(graph);
  auto evaluator = std::make_shared<Evaluator>();
  PuctSearch search(state, evaluator, params(100), 12);
  search.run();
  CHECK(search.rootNode().edgeVisits(0) == 100);
  const auto* child = search.rootNode().childAt(0);
  CHECK(child && outgoing(*child) == 99);
  auto c = config();
  c.recordTreeMinVisits = 100;
  Rng rng(3);
  CHECK(internal::extractTreePositions(c, *state, search.rootNode(), kInvalidAction, rng).empty());
  c.recordTreeMinVisits = 99;
  CHECK(internal::extractTreePositions(c, *state, search.rootNode(), kInvalidAction, rng).size() == 1);

  // The same child gets 100 shared visits, but each incoming edge only gets
  // 50. Its global stats must not make either 50-visit branch eligible at 51.
  graph = std::make_shared<Graph>(Graph{
      {0, {1, 1}, false, 0.0}, {1, {2}, false, 0.0}, {0, {}, false, 0.0}});
  state = std::make_shared<State>(graph);
  auto p = params(100);
  p.graphSearchCatchUpLeakProb = 1.0;
  PuctSearch sharedSearch(state, evaluator, p, 12);
  sharedSearch.run();
  CHECK(sharedSearch.rootNode().edgeVisits(0) == 50);
  CHECK(sharedSearch.rootNode().edgeVisits(1) == 50);
  child = sharedSearch.rootNode().childAt(0);
  CHECK(child == sharedSearch.rootNode().childAt(1));
  CHECK(child && outgoing(*child) >= 51);
  c.recordTreeMinVisits = 51;
  CHECK(internal::extractTreePositions(c, *state, sharedSearch.rootNode(), kInvalidAction, rng).empty());
}

void testCyclesAndBudget() {
  auto graph = std::make_shared<Graph>(Graph{
      {0, {1}}, {1, {2, 3}}, {1, {1, 3}}, {0, {}}});
  auto state = std::make_shared<State>(graph);
  auto evaluator = std::make_shared<Evaluator>();
  PuctSearch search(state, evaluator, params(), 11);
  search.run();
  auto c = config();
  c.recordTreeMaxDepth = 64;
  Rng rng(9);
  auto rows = internal::extractTreePositions(c, *state, search.rootNode(), kInvalidAction, rng);
  CHECK(rows.size() <= 2);
  std::set<int> ids;
  for (const auto& row : rows) CHECK(ids.insert(row.moveCount).second);

  // Layered DAG has exponentially many paths but few nodes. Repeated White
  // non-best moves leave Black's eligibility intact, so flag pruning alone
  // cannot bound this traversal. Count replay clones to verify the hard cap.
  graph = std::make_shared<Graph>();
  for (int i = 0; i < 20; ++i) graph->push_back({0, {i + 1, i + 1}});
  graph->push_back({1, {21}});
  graph->push_back({0, {}});
  state = std::make_shared<State>(graph);
  auto p = params(10000);
  p.maxDepth = 32;
  p.graphSearchCatchUpLeakProb = 1.0;
  PuctSearch dagSearch(state, evaluator, p, 11);
  dagSearch.run();
  c.recordTreeSampleProbability = 1.0;
  State::cloneCalls = 0;
  rows = internal::extractTreePositions(c, *state, dagSearch.rootNode(), kInvalidAction, rng);
  CHECK(rows.size() <= 20);
  CHECK(State::cloneCalls == 4095);  // 4096 visited nodes, including the root.
}

void testNativeReplay() {
  auto state = std::make_shared<Chess5DGameState>();
  class Uniform final : public IEvaluator {
   public:
    int calls = 0;
    Evaluation evaluate(const IGameState& state) override {
      ++calls;
      return {std::vector<float>(state.actionCount(), 1.0f), 0.125};
    }
  };
  auto evaluator = std::make_shared<Uniform>();
  PuctSearch search(state, evaluator, params(500), 17);
  search.run();
  const auto key = state->stateKey();
  const int calls = evaluator->calls;
  auto c = config();
  c.recordTreeMaxPerTurn = 4;
  Rng rng(1);
  const int played = search.chooseAction(1.0);
  auto rows = internal::extractTreePositions(c, *state, search.rootNode(), played, rng);
  CHECK(!rows.empty());
  CHECK(rows.size() <= 4);
  CHECK(evaluator->calls == calls);
  CHECK(state->stateKey() == key);
  for (const auto& row : rows) {
    CHECK(row.encoded.num_boards >= 1);
    CHECK(row.player == row.encoded.side_to_move);
    CHECK(row.policy.size() == row.actions.size());
    CHECK(std::isfinite(row.value) && row.value >= -1.0f && row.value <= 1.0f);
  }
}
void testSelfPlayLabelsAndIsolation() {
  auto c = config();
  c.evaluator.model_path = SUZERO_5DCHESS_MODEL_PATH;
  c.recordGames = true;
  c.numGames = 1;
  c.numGameThreads = 1;
  c.numSimulations = 64;
  c.fastSimulations = 4;
  c.playoutCapRandomization = false;
  c.minBoardLimit = c.maxBoardLimit = 6;
  c.recordTreeMaxPerTurn = 4;
  c.maxBatchSize = 4;
  c.seed = 37;
  c.recordTreePositions = false;
  SelfPlayStats baselineStats, treeStats;
  const auto baseline = runSelfPlay(c, {}, &baselineStats);
  c.recordTreePositions = true;
  const auto withTree = runSelfPlay(c, {}, &treeStats);
  CHECK(baseline.size() == 1 && withTree.size() == 1);
  if (baseline.empty() || withTree.empty()) return;
  const auto& a = baseline.front();
  const auto& b = withTree.front();
  CHECK(a.treeSamples == 0);
  CHECK(b.treeSamples > 0);
  CHECK(a.moves.size() == b.moves.size());
  for (std::size_t i = 0; i < std::min(a.moves.size(), b.moves.size()); ++i) {
    CHECK(a.moves[i].action == b.moves[i].action);
    CHECK(a.moves[i].player == b.moves[i].player);
  }
  CHECK(a.finalFen == b.finalFen);
  CHECK(a.whiteOutcome == b.whiteOutcome);
  CHECK(a.terminalReason == b.terminalReason);
  CHECK(baselineStats.requests == treeStats.requests);  // No new NN calls.
  std::size_t mainIndex = 0;
  int trees = 0;
  bool independentValue = false;
  for (const auto& row : b.samples) {
    const double outcome = row.player == 0 ? b.whiteOutcome : -b.whiteOutcome;
    if (row.source == 1) {
      ++trees;
      CHECK(row.searchVisits > 0);
      CHECK(std::isfinite(row.value) && std::abs(row.value) <= 1.0f);
      if (std::abs(row.value - outcome) > 1e-5) independentValue = true;
    } else {
      CHECK(row.source == 0);
      CHECK(std::abs(row.value - outcome) < 1e-6);
      CHECK(mainIndex < a.samples.size());
      if (mainIndex >= a.samples.size()) continue;
      const auto& original = a.samples[mainIndex++];
      CHECK(row.actions == original.actions);
      CHECK(row.policy == original.policy);
      CHECK(row.player == original.player);
      CHECK(row.moveCount == original.moveCount);
      CHECK(row.value == original.value);
      CHECK(row.encoded.board_planes == original.encoded.board_planes);
    }
  }
  CHECK(mainIndex == a.samples.size());
  CHECK(trees == b.treeSamples);
  CHECK(independentValue);  // Terminal relabelling must not overwrite tree values.
  c.playoutCapRandomization = true;
  c.fullSearchProbability = 0.0;
  const auto fastOnly = runSelfPlay(c);
  CHECK(fastOnly.front().treeSamples == 0);
  CHECK(fastOnly.front().samples.empty());
}

}  // namespace

int main() {
  try {
    testConfig();
    testSmallStepsAndDag();
    testBestMoveTieAndSamePlayer();
    testLeafAndSubmitOnly();
    testLocalVisitQualityGuards();
    testCyclesAndBudget();
    testNativeReplay();
    testSelfPlayLabelsAndIsolation();
  } catch (const std::exception& e) {
    std::cerr << "Unexpected exception: " << e.what() << "\n";
    return 1;
  }
  std::cout << checks << " checks, " << failures << " failures\n";
  return failures == 0 ? 0 : 1;
}
