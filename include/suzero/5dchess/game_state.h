// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-02 contributors
//
// 5D chess ("Very Small - Open") integration for the game-agnostic PUCT core.
//
// This header deliberately hides the upstream engine types behind a pimpl so
// that consumers only need C++17 and this include directory. The implementation
// compiles against the pinned 5dchess_engine and produces complete small-step
// actions: either one MoveKey (a piece move) or SUBMIT (end the current turn).

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "suzero/mcts/game_state.h"
#include "suzero/mcts/types.h"

namespace suzero::chess5d {

/// Raw small-step action:
///   (from_x, from_y, from_t, from_l, to_x, to_y, to_t, to_l)
///
/// Coordinates are 0-based board coordinates and native timeline indices.
using MoveKey = std::array<std::int32_t, 8>;

inline constexpr int kMoveKeyFields = 8;

/// One candidate action in a concrete state.
///
/// SUBMIT does not carry a MoveKey. It is always the last candidate when the
/// current turn can be submitted, matching the 5dc_ml runner protocol.
struct Action {
  bool is_submit = false;
  MoveKey move{};

  friend bool operator==(const Action& lhs, const Action& rhs) {
    return lhs.is_submit == rhs.is_submit && lhs.move == rhs.move;
  }
  friend bool operator!=(const Action& lhs, const Action& rhs) {
    return !(lhs == rhs);
  }
};

/// Static configuration of a 5dc "Very Small - Open" game.
///
/// Defaults mirror the current 5dc_ml very_small profile and C++ runner. The
/// board limit is the material-scoring trigger: once the number of live boards
/// reaches it on SUBMIT, the game ends with a tanh-scaled material score.
struct GameConfig {
  std::string variant_pgn = "[Board \"Very Small - Open\"]\n[Mode \"5D\"]\n";
  int board_side = 4;
  int board_squares = 16;
  int piece_channels = 12;
  int line_shift = 5;
  int board_limit = 25;
  double material_scale = 2.0;
};

/// Neural-network input for one position (without the action tensor).
struct EncodedPosition {
  int num_boards = 0;
  int side_to_move = 0;
  int board_side = 4;
  int board_squares = 16;
  int piece_channels = 12;
  float urgency = 0.0f;

  /// Layout: [num_boards][piece_channels][board_squares], row-major planes.
  std::vector<float> board_planes;

  /// Per-board native coordinates (timeline index shifted by line_shift).
  std::vector<std::int64_t> l_coords;
  std::vector<std::int64_t> t_coords;
};

/// Model-facing input implemented by any 5D-chess game adapter that can use the
/// current ONNX protocol. The smallstep_movekey_v1 protocol requires all
/// non-SUBMIT actions first and at most one SUBMIT action, in the last slot.
class IChess5DModelInput {
 public:
  virtual ~IChess5DModelInput() = default;

  virtual const std::vector<Action>& actions() const = 0;
  virtual EncodedPosition encodePosition() const = 0;
};

/// One native board in an archival snapshot. Unlike RecordSnapshot::fen, this
/// board-local FEN omits unmoved-piece flags, matching engine.game's
/// get_current_boards() API and the .5dck replay cache.
struct RecordBoardSnapshot {
  int l = 0;
  int t = 0;
  bool c = false;
  std::string fen;
};

/// Read-only native position data needed to build a version-1 .5dck record.
/// It is intentionally independent of the neural-network encoding so it keeps
/// every historical board and branch represented by the engine state.
struct RecordSnapshot {
  int sizeX = 0;
  int sizeY = 0;
  std::string fen;
  std::vector<RecordBoardSnapshot> boards;

  int presentT = 0;
  bool presentC = false;
  std::vector<int> mandatoryTimelines;
  std::vector<int> optionalTimelines;
  std::vector<int> unplayableTimelines;
};

/// IGameState implementation backed by the 5dchess_engine.
///
/// Rules mode is capture-king, matching the current 5dc_ml training default:
/// capturing a royal piece ends the game immediately; royal exposure is legal.
/// A turn is a sequence of small steps and ends with SUBMIT. Because a player
/// may make several small steps in a row, the PUCT core flips value signs only
/// when `currentPlayer()` actually changes.
///
/// The candidate action list is state-dependent. It is a deterministic function
/// of the position, so equal `stateKey()` values enumerate identical actions in
/// identical order (the contract added to IGameState for dynamic action spaces).
class Chess5DGameState final : public mcts::IGameState,
                                  public IChess5DModelInput {
 public:
  explicit Chess5DGameState(GameConfig config = {});
  ~Chess5DGameState() override;

  Chess5DGameState(const Chess5DGameState&) = delete;
  Chess5DGameState& operator=(const Chess5DGameState&) = delete;

  std::unique_ptr<mcts::IGameState> clone() const override;

  /// Covariant clone for callers that need the concrete type.
  std::unique_ptr<Chess5DGameState> cloneState() const;

  int actionCount() const override;
  bool isLegal(mcts::ActionId action) const override;
  void applyAction(mcts::ActionId action) override;

  mcts::PlayerId currentPlayer() const override;
  bool isTerminal() const override;
  double terminalUtility() const override;
  mcts::StateKey stateKey() const override;
  int moveCount() const override;

  // ---------------------------------------------------------- integration --

  const GameConfig& config() const;

  /// Candidate actions of the current position, in stable order.
  const std::vector<Action>& actions() const override;

  /// Throws std::out_of_range when `action` is not a candidate index.
  const Action& action(mcts::ActionId action) const;

  /// Number of live boards.
  int boardCount() const;

  /// Material-scoring board limit from the configuration.
  int boardLimit() const;

  /// Neural-network urgency input: max(0, board_limit - board_count).
  float urgency() const;

  /// Encode the current position for the ONNX evaluator.
  EncodedPosition encodePosition() const override;

  /// Capture the complete native position for .5dck archival. Board-local FENs
  /// omit unmoved flags; `fen` is the engine's full show_fen() serialization
  /// and retains them. This includes unsubmitted moves and branch history.
  RecordSnapshot recordSnapshot() const;

  /// Non-empty only for terminal states:
  /// "capture_king", "material", or "no_legal_action".
  const std::string& terminalReason() const;

  /// Native hash of the engine position (candidate bucket; stateKey() is the
  /// exact identity and is what the search compares).
  std::uint64_t nativeHash() const;

 private:
  struct Impl;

  explicit Chess5DGameState(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace suzero::chess5d
