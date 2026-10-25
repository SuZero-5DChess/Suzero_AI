// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-5DChess contributors

#include "suzero/5dchess/game_state.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "action.h"
#include "bitboard.h"
#include "board.h"
#include "hash.h"
#include "piece.h"
#include "state.h"
#include "turn.h"
#include "vec4.h"

namespace suzero::chess5d {

namespace {

constexpr piece_t kPromotionPiece = QUEEN_W;

bool isRoyalPiece(piece_t piece) {
  const piece_t named = piece_name(piece);
  return named == KING_W || named == KING_B || named == KING_UW ||
         named == KING_UB || named == COMMON_KING_W ||
         named == COMMON_KING_B || named == ROYAL_QUEEN_W ||
         named == ROYAL_QUEEN_B;
}

int pieceValue(piece_t piece) {
  switch (to_white(piece_name(piece))) {
    case KING_W:
      return 0;
    case QUEEN_W:
      return 9;
    case ROOK_W:
      return 5;
    case BISHOP_W:
      return 3;
    case KNIGHT_W:
      return 3;
    case PAWN_W:
      return 1;
    case UNICORN_W:
      return 3;
    case DRAGON_W:
      return 5;
    case BRAWN_W:
      return 1;
    case PRINCESS_W:
      return 7;
    case ROYAL_QUEEN_W:
      return 9;
    case COMMON_KING_W:
      return 3;
    default:
      return 0;
  }
}

int pieceChannel(piece_t piece) {
  switch (piece_name(piece)) {
    case KING_W:
      return 0;
    case QUEEN_W:
      return 1;
    case ROOK_W:
      return 2;
    case BISHOP_W:
      return 3;
    case KNIGHT_W:
      return 4;
    case PAWN_W:
      return 5;
    case KING_B:
      return 6;
    case QUEEN_B:
      return 7;
    case ROOK_B:
      return 8;
    case BISHOP_B:
      return 9;
    case KNIGHT_B:
      return 10;
    case PAWN_B:
      return 11;
    default:
      return -1;
  }
}

MoveKey moveKeyFromVecs(const vec4& from, const vec4& to) {
  return MoveKey{from.x(), from.y(), from.t(), from.l(),
                 to.x(),   to.y(),   to.t(),   to.l()};
}

vec4 vecFromMoveKey(const MoveKey& key, int offset) {
  return vec4(key[offset + 0], key[offset + 1], key[offset + 2],
              key[offset + 3]);
}

/// Exact, history-free identity of an engine position. This mirrors the
/// serializer used by 5dc_ml: all board bitboards (including unmoved flags),
/// board coordinates, odd/even layout, present turn, player and promotion
/// rules. PGN metadata and comments are intentionally absent.
std::string engineIdentity(const state& s) {
  std::ostringstream out;
  const auto initial_lines = s.get_initial_lines_range();
  const auto [width, height] = s.get_board_size();
  const auto [present, player] = s.get_present().get();
  out << width << ':' << height << ':'
      << (initial_lines.first != initial_lines.second) << ':' << present << ':'
      << player << ':' << static_cast<int>(s.get_promotion_options());

  const auto [first_line, last_line] = s.get_lines_range();
  for (int line = first_line; line <= last_line; ++line) {
    const turn end = s.get_timeline_end(line);
    for (turn at = s.get_timeline_start(line); at <= end;
         at = at.next_turn()) {
      const board* b = s.get_board_ptr(line, at.t(), at.c());
      if (b == nullptr) {
        continue;
      }
      out << ";B:" << line << ':' << at.t() << ':' << at.c();
      const bitboard_t fields[] = {
          b->white(),  b->black(),   b->royal(),  b->lking(),
          b->lknight(), b->lpawn(),  b->lrawn(),  b->lrook(),
          b->lbishop(), b->lunicorn(), b->ldragon(), b->umove()};
      for (bitboard_t field : fields) {
        out << ':' << field;
      }
    }
  }
  return out.str();
}

}  // namespace

struct Chess5DGameState::Impl {
  explicit Impl(GameConfig cfg)
      : config(std::move(cfg)),
        position(state::from_pgn<true>(config.variant_pgn)) {
    if (config.board_side <= 0 || config.board_squares <= 0 ||
        config.piece_channels <= 0) {
      throw std::invalid_argument("Chess5DGameState: invalid board geometry");
    }
    if (config.board_squares != config.board_side * config.board_side) {
      throw std::invalid_argument(
          "Chess5DGameState: board_squares must equal board_side^2");
    }
    if (config.board_limit <= 0) {
      throw std::invalid_argument(
          "Chess5DGameState: board_limit must be positive");
    }
    board_count = static_cast<int>(position.get_boards().size());
    if (board_count >= config.board_limit) {
      markMaterialTerminal();
    } else {
      ensureActions();
    }
  }

  Impl(const Impl&) = default;

  GameConfig config;
  state position;
  int board_count = 0;
  int move_count = 0;

  // The action list is generated lazily from `position` in const methods, and
  // an empty frontier is itself the no-legal-action terminal condition.
  mutable bool done = false;
  mutable std::string terminal_reason;
  mutable double white_outcome = 0.0;

  mutable bool actions_ready = false;
  mutable std::vector<Action> actions;

  bool currentPlayerIsBlack() const { return position.get_present().c(); }

  void refreshBoardCount() {
    board_count = static_cast<int>(position.get_boards().size());
  }

  void markMaterialTerminal() {
    done = true;
    terminal_reason = "material";
    white_outcome = materialScore();
    actions.clear();
    actions_ready = true;
  }

  void markNoLegalActionTerminal() const {
    done = true;
    terminal_reason = "no_legal_action";
    // The player to move has no legal action and therefore loses.
    white_outcome = currentPlayerIsBlack() ? 1.0 : -1.0;
  }

  void rebuildActions() const {
    actions.clear();
    actions_ready = true;
    if (done) {
      return;
    }

    const auto [mandatory, optional, unplayable] =
        position.get_timeline_status();
    (void)unplayable;

    std::set<MoveKey> seen;
    auto process_lines = [&](const std::vector<int>& lines) {
      for (int line : lines) {
        const auto [t, c] = position.get_timeline_end(line).get();
        const board* b = position.get_board_ptr(line, t, c);
        if (b == nullptr) {
          continue;
        }
        const bitboard_t pieces =
            c ? (b->black() & ~b->white()) : (b->white() & ~b->black());
        for (int pos : marked_pos(pieces)) {
          const vec4 from(pos, vec4(0, 0, t, line));
          for (const vec4& to : position.gen_piece_move(from)) {
            const MoveKey key = moveKeyFromVecs(from, to);
            if (seen.insert(key).second) {
              actions.push_back(Action{false, key});
            }
          }
        }
      }
    };

    process_lines(mandatory);
    process_lines(optional);
    if (mandatory.empty()) {
      // SUBMIT is always last, matching the 5dc_ml model protocol.
      actions.push_back(Action{true, MoveKey{}});
    }

    if (actions.empty()) {
      markNoLegalActionTerminal();
    }
  }

  void ensureActions() const {
    if (!actions_ready) {
      rebuildActions();
    }
  }

  bool capturesRoyal(const MoveKey& key, bool mover_is_black) const {
    const piece_t captured =
        position.get_piece(vecFromMoveKey(key, 4), mover_is_black);
    return captured != NO_PIECE && captured != WALL_PIECE &&
           isRoyalPiece(captured);
  }

  std::pair<int, int> materialCount() const {
    int white = 0;
    int black = 0;
    for (const auto& [line, t, c, fen] : position.get_boards()) {
      (void)line;
      (void)t;
      (void)c;
      (void)fen;
      const board* b = position.get_board_ptr(line, t, c);
      if (b == nullptr) {
        continue;
      }
      for (int pos = 0; pos < BOARD_SIZE; ++pos) {
        const piece_t piece = b->get_piece(pos);
        if (piece == NO_PIECE || piece == WALL_PIECE) {
          continue;
        }
        const int value = pieceValue(piece);
        if (piece_color(piece_name(piece))) {
          black += value;
        } else {
          white += value;
        }
      }
    }
    return {white, black};
  }

  double materialScore() const {
    const auto [white, black] = materialCount();
    const double total = static_cast<double>(white + black) + 1e-8;
    return std::tanh(static_cast<double>(white - black) / total *
                     config.material_scale);
  }
};

Chess5DGameState::Chess5DGameState(GameConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

Chess5DGameState::Chess5DGameState(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

Chess5DGameState::~Chess5DGameState() = default;

std::unique_ptr<mcts::IGameState> Chess5DGameState::clone() const {
  return cloneState();
}

std::unique_ptr<Chess5DGameState> Chess5DGameState::cloneState() const {
  return std::unique_ptr<Chess5DGameState>(
      new Chess5DGameState(std::make_unique<Impl>(*impl_)));
}

int Chess5DGameState::actionCount() const {
  return static_cast<int>(actions().size());
}

bool Chess5DGameState::isLegal(mcts::ActionId action) const {
  return action >= 0 && action < actionCount();
}

void Chess5DGameState::applyAction(mcts::ActionId action_id) {
  const std::size_t count = actions().size();
  if (action_id < 0 || static_cast<std::size_t>(action_id) >= count) {
    throw std::out_of_range("Chess5DGameState::applyAction: bad action id");
  }
  if (impl_->done) {
    throw std::logic_error(
        "Chess5DGameState::applyAction: state is terminal");
  }

  const Action action = impl_->actions[static_cast<std::size_t>(action_id)];

  if (action.is_submit) {
    if (!impl_->position.submit<true>()) {
      throw std::runtime_error(
          "Chess5DGameState: engine rejected a generated SUBMIT");
    }
    ++impl_->move_count;
    impl_->refreshBoardCount();
    impl_->actions_ready = false;

    if (impl_->board_count >= impl_->config.board_limit) {
      impl_->markMaterialTerminal();
    } else {
      impl_->ensureActions();
    }
    return;
  }

  // The mover stays `currentPlayer()` across small steps; SUBMIT is what
  // advances the turn. Capture the color before applying anyway, exactly as
  // the reference env does.
  const bool mover_is_black = impl_->currentPlayerIsBlack();
  const bool captured_royal = impl_->capturesRoyal(action.move, mover_is_black);
  const vec4 from = vecFromMoveKey(action.move, 0);
  const vec4 to = vecFromMoveKey(action.move, 4);
  const ext_move move(from, to, kPromotionPiece);
  if (!impl_->position.apply_move<true>(move)) {
    throw std::runtime_error(
        "Chess5DGameState: engine rejected a generated move");
  }
  ++impl_->move_count;
  impl_->refreshBoardCount();
  impl_->actions_ready = false;

  if (captured_royal) {
    impl_->done = true;
    impl_->terminal_reason = "capture_king";
    impl_->white_outcome = mover_is_black ? -1.0 : 1.0;
    impl_->actions.clear();
    impl_->actions_ready = true;
  } else if (impl_->board_count >= impl_->config.board_limit) {
    impl_->markMaterialTerminal();
  } else {
    impl_->ensureActions();
  }
}

mcts::PlayerId Chess5DGameState::currentPlayer() const {
  return impl_->currentPlayerIsBlack() ? 1 : 0;
}

bool Chess5DGameState::isTerminal() const {
  impl_->ensureActions();
  return impl_->done;
}

double Chess5DGameState::terminalUtility() const {
  if (!impl_->done) {
    return 0.0;
  }
  return impl_->currentPlayerIsBlack() ? -impl_->white_outcome
                                       : impl_->white_outcome;
}

mcts::StateKey Chess5DGameState::stateKey() const {
  std::ostringstream out;
  out << "5dc:" << impl_->config.board_limit << ':'
      << std::setprecision(17) << impl_->config.material_scale << ':'
      << (impl_->done ? 1 : 0) << ':' << std::hexfloat
      << impl_->white_outcome << ':' << impl_->terminal_reason << '|'
      << engineIdentity(impl_->position);
  return out.str();
}

int Chess5DGameState::moveCount() const { return impl_->move_count; }

const GameConfig& Chess5DGameState::config() const { return impl_->config; }

const std::vector<Action>& Chess5DGameState::actions() const {
  impl_->ensureActions();
  return impl_->actions;
}

const Action& Chess5DGameState::action(mcts::ActionId action_id) const {
  const auto& list = actions();
  if (action_id < 0 || static_cast<std::size_t>(action_id) >= list.size()) {
    throw std::out_of_range("Chess5DGameState::action: bad action id");
  }
  return list[static_cast<std::size_t>(action_id)];
}

int Chess5DGameState::boardCount() const { return impl_->board_count; }

int Chess5DGameState::boardLimit() const {
  return impl_->config.board_limit;
}

float Chess5DGameState::urgency() const {
  return static_cast<float>(
      std::max(0, impl_->config.board_limit - impl_->board_count));
}

EncodedPosition Chess5DGameState::encodePosition() const {
  EncodedPosition encoded;
  encoded.board_side = impl_->config.board_side;
  encoded.board_squares = impl_->config.board_squares;
  encoded.piece_channels = impl_->config.piece_channels;
  encoded.side_to_move = impl_->currentPlayerIsBlack() ? 1 : 0;
  encoded.urgency = urgency();

  const auto boards = impl_->position.get_boards();
  encoded.num_boards = static_cast<int>(boards.size());
  const std::size_t planes_per_board =
      static_cast<std::size_t>(encoded.piece_channels) *
      static_cast<std::size_t>(encoded.board_squares);
  encoded.board_planes.assign(
      static_cast<std::size_t>(encoded.num_boards) * planes_per_board, 0.0f);
  encoded.l_coords.resize(static_cast<std::size_t>(encoded.num_boards));
  encoded.t_coords.resize(static_cast<std::size_t>(encoded.num_boards));

  for (int i = 0; i < encoded.num_boards; ++i) {
    const auto& [line, t, c, fen] = boards[static_cast<std::size_t>(i)];
    (void)fen;
    encoded.l_coords[static_cast<std::size_t>(i)] =
        static_cast<std::int64_t>(line + impl_->config.line_shift);
    encoded.t_coords[static_cast<std::size_t>(i)] =
        static_cast<std::int64_t>(t);

    const board* b = impl_->position.get_board_ptr(line, t, c);
    if (b == nullptr) {
      continue;
    }
    for (int y = 0; y < encoded.board_side; ++y) {
      for (int x = 0; x < encoded.board_side; ++x) {
        const int square = x + y * encoded.board_side;
        const int board_pos = ppos(x, y);
        const piece_t raw = b->get_piece(board_pos);
        if (raw == NO_PIECE || raw == WALL_PIECE) {
          continue;
        }
        const int channel = pieceChannel(raw);
        if (channel < 0) {
          continue;
        }
        const std::size_t index =
            static_cast<std::size_t>(i) * planes_per_board +
            static_cast<std::size_t>(channel) * encoded.board_squares +
            static_cast<std::size_t>(square);
        encoded.board_planes[index] = 1.0f;
      }
    }
  }

  return encoded;
}

RecordSnapshot Chess5DGameState::recordSnapshot() const {
  RecordSnapshot snapshot;
  std::tie(snapshot.sizeX, snapshot.sizeY) =
      impl_->position.get_board_size();
  snapshot.fen = impl_->position.show_fen();

  const auto native_boards = impl_->position.get_boards();
  snapshot.boards.reserve(native_boards.size());
  for (const auto& [l, t, c, fen] : native_boards) {
    snapshot.boards.push_back(RecordBoardSnapshot{l, t, c, fen});
  }
  std::sort(snapshot.boards.begin(), snapshot.boards.end(),
            [](const RecordBoardSnapshot& lhs,
               const RecordBoardSnapshot& rhs) {
              return std::tie(lhs.l, lhs.t, lhs.c) <
                     std::tie(rhs.l, rhs.t, rhs.c);
            });

  const auto [present_t, present_c] = impl_->position.get_present().get();
  snapshot.presentT = present_t;
  snapshot.presentC = present_c;
  std::tie(snapshot.mandatoryTimelines, snapshot.optionalTimelines,
           snapshot.unplayableTimelines) =
      impl_->position.get_timeline_status();
  return snapshot;
}

const std::string& Chess5DGameState::terminalReason() const {
  return impl_->terminal_reason;
}

std::uint64_t Chess5DGameState::nativeHash() const {
  return hash_state(impl_->position);
}

}  // namespace suzero::chess5d
