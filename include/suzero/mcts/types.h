// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-02 contributors
//
// This file is part of a game-agnostic PUCT/MCTS library derived from KataGo
// (https://github.com/lightvector/KataGo). See THIRD_PARTY_NOTICES.md.

#pragma once

#include <cstdint>

namespace suzero::mcts {

/// Dense action index used by the game abstraction and by policies.
///
/// The meaning of an action is entirely up to IGameState. The search engine
/// never interprets it beyond storing it and passing it back.
using ActionId = std::int32_t;

/// Stable player identifier. Only the equality relation matters to the engine.
///
/// Two-player zero-sum games normally use 0 and 1. The engine does not assume
/// players alternate: the perspective of a child value is decided by comparing
/// the child's player with the parent's player.
using PlayerId = std::int32_t;

inline constexpr ActionId kInvalidAction = -1;

/// Selection value used to mark actions that must never be chosen.
inline constexpr double kIllegalSelectionValue = -1e50;

/// Default seed for deterministic search runs.
inline constexpr std::uint64_t kDefaultSeed = 0x853c49e6748fea9bULL;

}  // namespace suzero::mcts
