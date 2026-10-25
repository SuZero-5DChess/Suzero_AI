// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-02 contributors
//
// Minimal games used only by the unit tests. They are *not* engine
// integrations and are not part of the library target.

#pragma once

#include <memory>
#include <string>
#include <utility>

#include "suzero/mcts/evaluator.h"
#include "suzero/mcts/game_state.h"

namespace suzero::mcts::test {

/// Alternating game: move a token by 1 or 2; land exactly on 5 to win.
class SubtractionGame final : public IGameState {
 public:
  static constexpr int kTarget = 5;
  static constexpr int kActionCount = 2;

  explicit SubtractionGame(int position = 0, PlayerId player = 0,
                           int move_count = 0)
      : position_(position), player_(player), move_count_(move_count) {}

  std::unique_ptr<IGameState> clone() const override {
    return std::make_unique<SubtractionGame>(*this);
  }

  int actionCount() const override { return kActionCount; }

  bool isLegal(ActionId action) const override {
    if (isTerminal() || action < 0 || action >= kActionCount) {
      return false;
    }
    return position_ + (action + 1) <= kTarget;
  }

  void applyAction(ActionId action) override {
    position_ += action + 1;
    player_ = 1 - player_;
  }

  PlayerId currentPlayer() const override { return player_; }
  bool isTerminal() const override { return position_ == kTarget; }
  int moveCount() const override { return move_count_; }

  /// The side to move after the winning move has, by definition, lost.
  double terminalUtility() const override { return -1.0; }

  StateKey stateKey() const override {
    return "sub:" + std::to_string(position_) + ":" +
           std::to_string(player_);
  }

  int position() const { return position_; }

 private:
  int position_;
  PlayerId player_;
  int move_count_;
};

/// Every action ends the game. Only action 2 wins for the mover.
class OneMoveWinGame final : public IGameState {
 public:
  static constexpr int kActionCount = 3;

  explicit OneMoveWinGame(bool done = false, PlayerId player = 0,
                          PlayerId winner = 0)
      : done_(done), player_(player), winner_(winner) {}

  std::unique_ptr<IGameState> clone() const override {
    return std::make_unique<OneMoveWinGame>(*this);
  }

  int actionCount() const override { return kActionCount; }

  bool isLegal(ActionId action) const override {
    return !done_ && action >= 0 && action < kActionCount;
  }

  void applyAction(ActionId action) override {
    winner_ = (action == 2) ? player_ : (1 - player_);
    done_ = true;
    player_ = 1 - player_;
  }

  PlayerId currentPlayer() const override { return player_; }
  bool isTerminal() const override { return done_; }

  double terminalUtility() const override {
    return winner_ == player_ ? 1.0 : -1.0;
  }

  StateKey stateKey() const override {
    // winner_ is only meaningful once the game is over.
    return "omw:" + std::to_string(done_ ? 1 : 0) + ":" +
           std::to_string(player_) + ":" +
           std::to_string(done_ ? winner_ : -1);
  }

 private:
  bool done_;
  PlayerId player_;
  PlayerId winner_;
};

/// Non-alternating game: the same player makes both moves. The unique winning
/// line is first action 0, then action 1. This exercises the rule that values
/// are negated only when the player to move actually changes.
class SamePlayerGame final : public IGameState {
 public:
  static constexpr int kActionCount = 2;

  explicit SamePlayerGame(int step = 0, ActionId first = 0, bool done = false,
                          bool won = false)
      : step_(step), first_(first), done_(done), won_(won) {}

  std::unique_ptr<IGameState> clone() const override {
    return std::make_unique<SamePlayerGame>(*this);
  }

  int actionCount() const override { return kActionCount; }

  bool isLegal(ActionId action) const override {
    return !done_ && action >= 0 && action < kActionCount;
  }

  void applyAction(ActionId action) override {
    if (step_ == 0) {
      first_ = action;
      step_ = 1;
    } else {
      won_ = (first_ == 0 && action == 1);
      done_ = true;
    }
  }

  PlayerId currentPlayer() const override { return 0; }
  bool isTerminal() const override { return done_; }
  double terminalUtility() const override { return won_ ? 1.0 : -1.0; }

  StateKey stateKey() const override {
    return "same:" + std::to_string(step_) + ":" +
           std::to_string(first_) + ":" + std::to_string(done_ ? 1 : 0) +
           ":" + std::to_string(done_ && won_ ? 1 : 0);
  }

 private:
  int step_;
  ActionId first_;
  bool done_;
  bool won_;
};

/// Two moves that commute: playing action 0 then 1 reaches exactly the same
/// state as playing 1 then 0. Used to verify that graph search merges the
/// transposition into a single node.
class CommutingGame final : public IGameState {
 public:
  static constexpr int kActionCount = 2;
  static constexpr int kDoneMask = 3;

  explicit CommutingGame(int mask = 0, PlayerId player = 0)
      : mask_(mask), player_(player) {}

  std::unique_ptr<IGameState> clone() const override {
    return std::make_unique<CommutingGame>(*this);
  }

  int actionCount() const override { return kActionCount; }

  bool isLegal(ActionId action) const override {
    if (isTerminal() || action < 0 || action >= kActionCount) {
      return false;
    }
    return (mask_ & (1 << action)) == 0;
  }

  void applyAction(ActionId action) override {
    mask_ |= 1 << action;
    player_ = 1 - player_;
  }

  PlayerId currentPlayer() const override { return player_; }
  bool isTerminal() const override { return mask_ == kDoneMask; }
  double terminalUtility() const override { return 1.0; }

  StateKey stateKey() const override {
    return "commute:" + std::to_string(mask_) + ":" +
           std::to_string(player_);
  }

 private:
  int mask_;
  PlayerId player_;
};

/// Action 0 cycles between two non-terminal positions, action 1 ends the
/// game. With graph search this creates a cycle-closing edge that the playout
/// path check must stop.
class CycleGame final : public IGameState {
 public:
  static constexpr int kActionCount = 2;

  explicit CycleGame(int phase = 0, bool done = false, PlayerId player = 0)
      : phase_(phase), done_(done), player_(player) {}

  std::unique_ptr<IGameState> clone() const override {
    return std::make_unique<CycleGame>(*this);
  }

  int actionCount() const override { return kActionCount; }

  bool isLegal(ActionId action) const override {
    return !done_ && action >= 0 && action < kActionCount;
  }

  void applyAction(ActionId action) override {
    if (action == 0) {
      phase_ = 1 - phase_;
      player_ = 1 - player_;
    } else {
      done_ = true;
      player_ = 1 - player_;
    }
  }

  PlayerId currentPlayer() const override { return player_; }
  bool isTerminal() const override { return done_; }
  double terminalUtility() const override { return 1.0; }

  StateKey stateKey() const override {
    return "cycle:" + std::to_string(phase_) + ":" +
           std::to_string(done_ ? 1 : 0) + ":" + std::to_string(player_);
  }

 private:
  int phase_;
  bool done_;
  PlayerId player_;
};

/// Uniform policy over legal actions, neutral value.
class UniformEvaluator final : public IEvaluator {
 public:
  Evaluation evaluate(const IGameState& state) override {
    Evaluation evaluation;
    evaluation.policy.assign(state.actionCount(), -1.0f);
    for (int action = 0; action < state.actionCount(); ++action) {
      if (state.isLegal(static_cast<ActionId>(action))) {
        evaluation.policy[static_cast<std::size_t>(action)] = 1.0f;
      }
    }
    evaluation.utility = 0.0;
    return evaluation;
  }
};

/// Uniform value, deliberately skewed prior for OneMoveWinGame.
class SkewedWinEvaluator final : public IEvaluator {
 public:
  Evaluation evaluate(const IGameState& state) override {
    Evaluation evaluation;
    evaluation.policy.assign(state.actionCount(), -1.0f);
    (void)state;
    const float priors[] = {0.6f, 0.25f, 0.15f};
    for (int action = 0; action < state.actionCount(); ++action) {
      evaluation.policy[static_cast<std::size_t>(action)] = priors[action];
    }
    evaluation.utility = 0.0;
    return evaluation;
  }
};

/// Gives action 0 a zero prior and every other legal action an equal positive
/// prior. Used to check that legal zero-probability moves stay visible.
class ZeroActionPriorEvaluator final : public IEvaluator {
 public:
  Evaluation evaluate(const IGameState& state) override {
    Evaluation evaluation;
    evaluation.policy.assign(state.actionCount(), -1.0f);
    for (int action = 0; action < state.actionCount(); ++action) {
      if (!state.isLegal(static_cast<ActionId>(action))) {
        continue;
      }
      evaluation.policy[static_cast<std::size_t>(action)] =
          action == 0 ? 0.0f : 1.0f;
    }
    evaluation.utility = 0.0;
    return evaluation;
  }
};

/// Gives every legal action a zero prior.
class AllZeroPriorEvaluator final : public IEvaluator {
 public:
  Evaluation evaluate(const IGameState& state) override {
    Evaluation evaluation;
    evaluation.policy.assign(state.actionCount(), -1.0f);
    for (int action = 0; action < state.actionCount(); ++action) {
      if (state.isLegal(static_cast<ActionId>(action))) {
        evaluation.policy[static_cast<std::size_t>(action)] = 0.0f;
      }
    }
    evaluation.utility = 0.0;
    return evaluation;
  }
};

/// Returns a policy of the wrong length; used to test validation.
class BrokenEvaluator final : public IEvaluator {
 public:
  Evaluation evaluate(const IGameState& state) override {
    Evaluation evaluation;
    evaluation.policy.assign(state.actionCount() + 1, 1.0f);
    return evaluation;
  }
};

}  // namespace suzero::mcts::test
