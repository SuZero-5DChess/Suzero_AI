#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <ostream>
#include <random>
#include <algorithm>
#include "game.h"
#include "hypercuboid.h"

namespace py = pybind11;

// =============================================
// ML helper: piece_t -> channel index for tensor
// =============================================
static int piece_to_channel(piece_t p)
{
    piece_t pn = piece_name(p);
    if(pn == NO_PIECE || pn == WALL_PIECE) return 0;
    int sign = piece_color(pn) ? -1 : 1;
    piece_t wp = to_white(pn);
    switch(wp) {
        case KING_W: return sign * 1;
        case QUEEN_W: return sign * 2;
        case ROOK_W: return sign * 3;
        case BISHOP_W: return sign * 4;
        case KNIGHT_W: return sign * 5;
        case PAWN_W: return sign * 6;
        case UNICORN_W: return sign * 7;
        case DRAGON_W: return sign * 8;
        case BRAWN_W: return sign * 9;
        case PRINCESS_W: return sign * 10;
        case ROYAL_QUEEN_W: return sign * 11;
        case COMMON_KING_W: return sign * 12;
        default: return 0;
    }
}

// Number of piece-type channels for the board tensor
// 13 white planes + 13 black planes + 1 unmoved flag = 27 channels
static constexpr int NUM_PIECE_CHANNELS = 27;

// =============================================
// Create a multi-channel tensor for a single board
// Returns shape [NUM_PIECE_CHANNELS, 8, 8] as a flat vector
// Channel layout:
//   0: white king, 1: white queen, 2: white rook, 3: white bishop,
//   4: white knight, 5: white pawn, 6: white unicorn, 7: white dragon,
//   8: white brawn, 9: white princess, 10: white royal_queen, 11: white common_king,
//   12: black king ... 23: black common_king
//   24: unmoved flag (for castling/en passant)
//   25: wall mask
//   26: occupied mask
// =============================================
static std::vector<float> board_to_planes(const board &b, int size_x, int size_y)
{
    std::vector<float> planes(NUM_PIECE_CHANNELS * BOARD_SIZE, 0.0f);
    auto set_plane = [&](int channel, int pos, float val) {
        planes[channel * BOARD_SIZE + pos] = val;
    };

    for(int y = 0; y < size_y; y++) {
        for(int x = 0; x < size_x; x++) {
            int pos = ppos(x, y);
            piece_t p = b.get_piece(pos);
            if(p == NO_PIECE) continue;
            if(p == WALL_PIECE) {
                set_plane(25, pos, 1.0f); // wall channel
                continue;
            }
            // occupied channel
            set_plane(26, pos, 1.0f);
            // unmoved flag channel
            if(piece_umove_flag(p)) {
                set_plane(24, pos, 1.0f);
            }
            // piece type channel
            piece_t pn = piece_name(p);
            bool is_black = piece_color(pn);
            piece_t wp = to_white(pn);
            int ch = -1;
            switch(wp) {
                case KING_W: ch = 0; break;
                case QUEEN_W: ch = 1; break;
                case ROOK_W: ch = 2; break;
                case BISHOP_W: ch = 3; break;
                case KNIGHT_W: ch = 4; break;
                case PAWN_W: ch = 5; break;
                case UNICORN_W: ch = 6; break;
                case DRAGON_W: ch = 7; break;
                case BRAWN_W: ch = 8; break;
                case PRINCESS_W: ch = 9; break;
                case ROYAL_QUEEN_W: ch = 10; break;
                case COMMON_KING_W: ch = 11; break;
                default: break;
            }
            if(ch >= 0) {
                int offset = is_black ? 12 : 0;
                set_plane(ch + offset, pos, 1.0f);
            }
        }
    }
    return planes;
}

// =============================================
// Enumerate all legal actions (via HC search) up to a limit
// =============================================
static std::vector<action> enumerate_legal_actions_impl(const state &s, int limit)
{
    std::vector<action> result;
    try {
        auto [w, ss] = HC_info::build_HC(s);
        for(moveseq mvs : w.search(ss))
        {
            std::vector<ext_move> emvs;
            std::transform(mvs.begin(), mvs.end(), std::back_inserter(emvs), [](full_move m){
                return ext_move(m);
            });
            result.push_back(action::from_vector(emvs, s));
            if(limit > 0 && (int)result.size() >= limit) break;
        }
    } catch (...) {
        // If HC_info::build_HC fails (e.g., no moves to make), return empty
    }
    return result;
}

// =============================================
// Pick a random legal action
// =============================================
static std::optional<action> random_action_impl(const state &s)
{
    try {
        auto [w, ss] = HC_info::build_HC(s);
        std::vector<action> candidates;
        for(moveseq mvs : w.search(ss))
        {
            std::vector<ext_move> emvs;
            std::transform(mvs.begin(), mvs.end(), std::back_inserter(emvs), [](full_move m){
                return ext_move(m);
            });
            candidates.push_back(action::from_vector(emvs, s));
            if(candidates.size() >= 200) break;
        }
        if(candidates.empty()) return std::nullopt;
        static thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, (int)candidates.size() - 1);
        return candidates[dist(rng)];
    } catch (...) {
        return std::nullopt;
    }
}

// =============================================
// Apply an action to a state, returning new state
// =============================================
static std::optional<state> apply_action_impl(const state &s, const action &act)
{
    return s.can_apply(act);
}

// =============================================
// Get per-timeline move info for factored policy
// =============================================
struct timeline_moves {
    int line_idx;      // timeline L index
    bool is_mandatory; // mandatory or optional
    std::vector<std::pair<vec4, vec4>> moves; // from, to pairs
};

static std::vector<timeline_moves> get_per_timeline_moves(const state &s)
{
    std::vector<timeline_moves> result;
    auto [mandatory, optional, unplayable] = s.get_timeline_status();
    auto [present_t, player] = s.get_present();

    auto process_lines = [&](const std::vector<int>& lines, bool is_mandatory) {
        for(int l : lines) {
            timeline_moves tm;
            tm.line_idx = l;
            tm.is_mandatory = is_mandatory;
            auto [t, c] = s.get_timeline_end(l);
            auto b_ptr = s.get_board(l, t, c);
            bitboard_t pieces = c ? (b_ptr->black() & ~b_ptr->white()) : (b_ptr->white() & ~b_ptr->black());
            for(int pos : marked_pos(pieces)) {
                vec4 p(pos, vec4(0,0,t,l));
                for(vec4 q : s.gen_piece_move(p, c)) {
                    tm.moves.push_back({p, q});
                }
            }
            result.push_back(std::move(tm));
        }
    };

    process_lines(mandatory, true);
    process_lines(optional, false);
    return result;
}

static std::vector<timeline_moves> get_per_timeline_pseudolegal_moves(const state &s)
{
    std::vector<timeline_moves> result;
    auto [mandatory, optional, unplayable] = s.get_timeline_status();

    auto process_lines = [&](const std::vector<int>& lines, bool is_mandatory) {
        for(int l : lines) {
            timeline_moves tm;
            tm.line_idx = l;
            tm.is_mandatory = is_mandatory;
            auto [t, c] = s.get_timeline_end(l);
            auto b_ptr = s.get_board(l, t, c);
            bitboard_t pieces = c ? (b_ptr->black() & ~b_ptr->white()) : (b_ptr->white() & ~b_ptr->black());
            for(int pos : marked_pos(pieces)) {
                vec4 p(pos, vec4(0,0,t,l));
                for(vec4 q : s.gen_piece_move(p, c)) {
                    tm.moves.push_back({p, q});
                }
            }
            result.push_back(std::move(tm));
        }
    };

    process_lines(mandatory, true);
    process_lines(optional, false);
    return result;
}

static bool is_royal_piece_impl(piece_t piece)
{
    piece_t p = static_cast<piece_t>(piece_name(piece));
    return p == KING_W || p == KING_B
        || p == KING_UW || p == KING_UB
        || p == COMMON_KING_W || p == COMMON_KING_B
        || p == ROYAL_QUEEN_W || p == ROYAL_QUEEN_B;
}

PYBIND11_MODULE(engine, m) {
    m.doc() = "5d chess engine";
    py::enum_<piece_t>(m, "Piece")
        .value("NO_PIECE", NO_PIECE)
        .value("WALL_PIECE", WALL_PIECE)
        .value("KING_UW", KING_UW)
        .value("ROOK_UW", ROOK_UW)
        .value("PAWN_UW", PAWN_UW)
        .value("KING_UB", KING_UB)
        .value("ROOK_UB", ROOK_UB)
        .value("PAWN_UB", PAWN_UB)
        .value("KING_W", KING_W)
        .value("QUEEN_W", QUEEN_W)
        .value("BISHOP_W", BISHOP_W)
        .value("KNIGHT_W", KNIGHT_W)
        .value("ROOK_W", ROOK_W)
        .value("PAWN_W", PAWN_W)
        .value("UNICORN_W", UNICORN_W)
        .value("DRAGON_W", DRAGON_W)
        .value("BRAWN_W", BRAWN_W)
        .value("PRINCESS_W", PRINCESS_W)
        .value("ROYAL_QUEEN_W", ROYAL_QUEEN_W)
        .value("COMMON_KING_W", COMMON_KING_W)
        .value("KING_B", KING_B)
        .value("QUEEN_B", QUEEN_B)
        .value("BISHOP_B", BISHOP_B)
        .value("KNIGHT_B", KNIGHT_B)
        .value("ROOK_B", ROOK_B)
        .value("PAWN_B", PAWN_B)
        .value("UNICORN_B", UNICORN_B)
        .value("DRAGON_B", DRAGON_B)
        .value("BRAWN_B", BRAWN_B)
        .value("PRINCESS_B", PRINCESS_B)
        .value("ROYAL_QUEEN_B", ROYAL_QUEEN_B)
        .value("COMMON_KING_B", COMMON_KING_B)
        .export_values();
    py::enum_<match_status_t>(m, "match_status_t")
        .value("PLAYING", match_status_t::PLAYING)
        .value("WHITE_WINS", match_status_t::WHITE_WINS)
        .value("BLACK_WINS", match_status_t::BLACK_WINS)
        .value("STALEMATE", match_status_t::STALEMATE)
        .def("__str__", [](match_status_t status) {
            std::ostringstream oss;
            oss << status;
            return oss.str();
        });

    py::class_<vec4>(m, "vec4")
        .def(py::init<int, int, int, int>(), py::arg("x"), py::arg("y"), py::arg("t"), py::arg("l"))
        .def("l", &vec4::l)
        .def("t", &vec4::t)
        .def("y", &vec4::y)
        .def("x", &vec4::x)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self >= py::self)
        .def(py::self <= py::self)
        .def(py::self > py::self)
        .def(py::self < py::self)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(-py::self)
        .def(py::self * int())
        .def("to_string", &vec4::to_string)
        .def("__repr__", [](const vec4 &v) { return v.to_string(); });
    py::class_<ext_move>(m, "ext_move")
        .def(py::init<vec4, vec4, piece_t>(),
             py::arg("from"),
             py::arg("to"),
             py::arg("promote_to") = QUEEN_W)
        .def("get_from", &ext_move::get_from)
        .def("get_to", &ext_move::get_to)
        .def("get_promote", &ext_move::get_promote)
        .def("to_string", &ext_move::to_string)
        .def("__repr__", [](const ext_move &m){
            return "<ext_move " + m.to_string() + ">";
        })
        .def(py::self == py::self);
    py::class_<action>(m, "action")
        .def("get_moves", &action::get_moves)
        .def(py::self == py::self)
        .def("__repr__", [](const action &a){
            return "<action with " + std::to_string(a.get_moves().size()) + " moves>";
        })
    ;

    // =============================================
    // game class - main interface for Python
    // =============================================
    py::class_<game>(m, "game")
        .def_readwrite("metadata", &game::metadata)
        .def_static(
            "from_pgn",
            [](const std::string &pgn, bool allow_submit_with_checks) {
                return game::from_pgn(pgn, allow_submit_with_checks);
            },
            py::arg("pgn"),
            py::arg("allow_submit_with_checks") = false
        )
        .def("get_current_state", &game::get_current_state)
        .def("get_current_present", &game::get_current_present)
        .def("get_current_boards", &game::get_current_boards)
        .def("get_current_timeline_status", &game::get_current_timeline_status)
        .def("gen_move_if_playable", &game::gen_move_if_playable)
        .def("get_match_status", &game::get_match_status)
        .def("get_movable_pieces", &game::get_movable_pieces)
        .def("is_playable", &game::is_playable)
        .def("can_undo", &game::can_undo)
        .def("can_redo", &game::can_redo)
        .def("can_submit", &game::can_submit)
        .def("undo", &game::undo)
        .def("redo", &game::redo)
        .def("apply_move", &game::apply_move)
        .def("submit", &game::submit)
        .def("currently_check", &game::currently_check)
        .def("get_current_checks", &game::get_current_checks)
        .def("get_board_size", &game::get_board_size)
        .def("suggest_action", &game::suggest_action)
        .def("get_comments", &game::get_comments)
        .def("has_parent", &game::has_parent)
        .def("visit_parent", &game::visit_parent)
        .def("get_child_actions", &game::get_child_actions)
        .def("visit_child",
             [](game &g, const action &a) {
                 return g.visit_child(a);
             },
             py::arg("action")
        )
        .def("show_pgn", &game::show_pgn);

    // =============================================
    // state class - exposed for ML use
    // =============================================
    py::class_<state>(m, "state")
        .def(py::init<const state&>()) // copy constructor
        .def("get_present", &state::get_present)
        .def("get_board_size", &state::get_board_size)
        .def("get_lines_range", &state::get_lines_range)
        .def("get_active_range", &state::get_active_range)
        .def("get_initial_lines_range", &state::get_initial_lines_range)
        .def("get_timeline_start", &state::get_timeline_start)
        .def("get_timeline_end", &state::get_timeline_end)
        .def("get_boards", &state::get_boards)
        .def("get_timeline_status", [](const state &s) {
            return s.get_timeline_status();
        })
        .def("get_piece", &state::get_piece)
        .def("get_movable_pieces", &state::gen_movable_pieces)
        .def("gen_piece_move", [](const state &s, vec4 p, bool c) {
            std::vector<vec4> out;
            for(vec4 q : s.gen_piece_move(p, c)) {
                out.push_back(q);
            }
            return out;
        })
        .def("get_match_status", [](const state &s) {
            auto mt = s.get_mate_type();
            auto [t, c] = s.get_present();
            switch(mt) {
                case state::mate_type::CHECKMATE:
                    return c ? match_status_t::WHITE_WINS : match_status_t::BLACK_WINS;
                case state::mate_type::SOFTMATE:
                    return c ? match_status_t::WHITE_WINS : match_status_t::BLACK_WINS;
                case state::mate_type::STALEMATE:
                    return match_status_t::STALEMATE;
                default:
                    return match_status_t::PLAYING;
            }
        })
        .def("show_fen", &state::show_fen)
        .def("to_string", &state::to_string)
        .def("__repr__", &state::to_string)
        // ML-specific methods
        .def("get_board_tensor", [](const state &s, int l, int t, bool c) -> py::array_t<float> {
            auto b_ptr = s.get_board(l, t, c);
            if(!b_ptr) {
                throw std::runtime_error("Board not found at (" + std::to_string(l) + "," + std::to_string(t) + "," + (c?"b":"w") + ")");
            }
            auto [sx, sy] = s.get_board_size();
            auto planes = board_to_planes(*b_ptr, sx, sy);
            std::vector<py::ssize_t> shape = {NUM_PIECE_CHANNELS, BOARD_LENGTH, BOARD_LENGTH};
            auto result = py::array_t<float>(shape);
            auto buf = result.mutable_unchecked<3>();
            for(int ch = 0; ch < NUM_PIECE_CHANNELS; ch++) {
                for(int y = 0; y < BOARD_LENGTH; y++) {
                    for(int x = 0; x < BOARD_LENGTH; x++) {
                        buf(ch, y, x) = planes[ch * BOARD_SIZE + ppos(x,y)];
                    }
                }
            }
            return result;
        }, py::arg("l"), py::arg("t"), py::arg("c"),
        "Get board tensor as numpy array [27, 8, 8] for a specific board (l, t, color)")
        .def("get_all_board_tensors", [](const state &s) -> py::dict {
            auto boards = s.get_boards();
            auto [sx, sy] = s.get_board_size();
            py::dict result;
            for(const auto &[l,t,c,fen] : boards) {
                auto b_ptr = s.get_board(l, t, c);
                if(!b_ptr) continue;
                auto planes = board_to_planes(*b_ptr, sx, sy);
                std::vector<py::ssize_t> shape = {NUM_PIECE_CHANNELS, BOARD_LENGTH, BOARD_LENGTH};
                auto arr = py::array_t<float>(shape);
                auto buf = arr.mutable_unchecked<3>();
                for(int ch = 0; ch < NUM_PIECE_CHANNELS; ch++) {
                    for(int y = 0; y < BOARD_LENGTH; y++) {
                        for(int x = 0; x < BOARD_LENGTH; x++) {
                            buf(ch, y, x) = planes[ch * BOARD_SIZE + ppos(x,y)];
                        }
                    }
                }
                result[py::make_tuple(l, t, c)] = arr;
            }
            return result;
        }, "Get all board tensors as dict {(l,t,c): ndarray[27,8,8]}")
        .def("material_count", [](const state &s) -> std::pair<int,int> {
            auto boards_info = s.get_boards();
            int white_material = 0, black_material = 0;
            auto piece_value = [](piece_t p) -> int {
                piece_t wp = to_white(piece_name(p));
                switch(wp) {
                    case KING_W: return 0;
                    case QUEEN_W: return 9;
                    case ROOK_W: return 5;
                    case BISHOP_W: return 3;
                    case KNIGHT_W: return 3;
                    case PAWN_W: return 1;
                    case UNICORN_W: return 3;
                    case DRAGON_W: return 5;
                    case BRAWN_W: return 1;
                    case PRINCESS_W: return 7;
                    case ROYAL_QUEEN_W: return 9;
                    case COMMON_KING_W: return 3;
                    default: return 0;
                }
            };
            for(const auto &[l,t,c,fen] : boards_info) {
                auto b_ptr = s.get_board(l, t, c);
                if(!b_ptr) continue;
                for(int pos = 0; pos < BOARD_SIZE; pos++) {
                    piece_t p = b_ptr->get_piece(pos);
                    if(p == NO_PIECE || p == WALL_PIECE) continue;
                    int val = piece_value(p);
                    if(piece_color(piece_name(p))) {
                        black_material += val;
                    } else {
                        white_material += val;
                    }
                }
            }
            return {white_material, black_material};
        }, "Count material for white and black")
        .def("can_apply_action", [](const state &s, const action &act) -> bool {
            return s.can_apply(act).has_value();
        }, py::arg("action"))
    ;

    // =============================================
    // timeline_moves struct for factored policy
    // =============================================
    py::class_<timeline_moves>(m, "TimelineMoves")
        .def_readonly("line_idx", &timeline_moves::line_idx)
        .def_readonly("is_mandatory", &timeline_moves::is_mandatory)
        .def_property_readonly("moves", [](const timeline_moves &tm) {
            py::list result;
            for(const auto &[from, to] : tm.moves) {
                result.append(py::make_tuple(from, to));
            }
            return result;
        })
        .def_property_readonly("num_moves", [](const timeline_moves &tm) {
            return tm.moves.size();
        })
        .def("__repr__", [](const timeline_moves &tm) {
            return "<TimelineMoves L=" + std::to_string(tm.line_idx) +
                   " mandatory=" + (tm.is_mandatory ? "true" : "false") +
                   " moves=" + std::to_string(tm.moves.size()) + ">";
        })
    ;

    // =============================================
    // Free functions for ML
    // =============================================
    m.def("enumerate_legal_actions", &enumerate_legal_actions_impl,
          py::arg("state"), py::arg("limit") = 0,
          "Enumerate legal actions (limit=0 means all)");

    m.def("random_action", &random_action_impl,
          py::arg("state"),
          "Pick a random legal action");

    m.def("apply_action", &apply_action_impl,
          py::arg("state"), py::arg("action"),
          "Apply an action to a state, returning new state or None");

    m.def("get_per_timeline_moves", &get_per_timeline_moves,
          py::arg("state"),
          "Get per-timeline move lists for factored policy");

    m.def("get_per_timeline_pseudolegal_moves", &get_per_timeline_pseudolegal_moves,
          py::arg("state"),
          "Get per-timeline pseudolegal move lists without check legality filtering");

    m.def("is_royal_piece", &is_royal_piece_impl,
          py::arg("piece"),
          "Return whether a piece enum value is royal in capture-king mode");

    m.def("count_legal_actions", [](const state &s, int limit) -> int {
        int count = 0;
        try {
            auto [w, ss] = HC_info::build_HC(s);
            for([[maybe_unused]] moveseq mvs : w.search(ss)) {
                count++;
                if(limit > 0 && count >= limit) break;
            }
        } catch (...) {}
        return count;
    }, py::arg("state"), py::arg("limit") = 0,
    "Count legal actions without storing them");

    m.def("create_state_from_pgn", [](const std::string &pgn) -> state {
        game g = game::from_pgn(pgn);
        return g.get_current_state();
    }, py::arg("pgn"),
    "Create a state directly from PGN string");

    m.attr("NUM_PIECE_CHANNELS") = NUM_PIECE_CHANNELS;
    m.attr("BOARD_LENGTH") = BOARD_LENGTH;
    m.attr("BOARD_SIZE") = BOARD_SIZE;

    m.attr("SHOW_NOTHING") = state::SHOW_NOTHING;
    m.attr("SHOW_RELATIVE") = state::SHOW_RELATIVE;
    m.attr("SHOW_PAWN") = state::SHOW_PAWN;
    m.attr("SHOW_CAPTURE") = state::SHOW_CAPTURE;
    m.attr("SHOW_PROMOTION") = state::SHOW_PROMOTION;
    m.attr("SHOW_MATE") = state::SHOW_MATE;
    m.attr("SHOW_LCOMMENT") = state::SHOW_LCOMMENT;
    m.attr("SHOW_ALL") = state::SHOW_ALL;
    m.attr("SHOW_SHORT") = state::SHOW_SHORT;

    // Add version information
    m.def("get_version", []() {
#ifdef PROJECT_VERSION_STRING
        return std::string(PROJECT_VERSION_STRING);
#else
        return std::string("unknown");
#endif
    });

    // Add __version__ attribute
#ifdef PROJECT_VERSION_STRING
    m.attr("__version__") = std::string(PROJECT_VERSION_STRING);
#else
    m.attr("__version__") = std::string("unknown");
#endif
}