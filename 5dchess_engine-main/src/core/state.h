#ifndef STATE_H
#define STATE_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <iostream>
#include "multiverse.h"
#include "action.h"
#include "promotion_header.h"
#include "generator.h"
#include "ast.h"
#include "pgnerror.h"

class state
{
    friend std::uint64_t hash_state(const state &s);

    multiverse m;
    // The current action's turn can differ from the multiverse's apparent present.
    turn present;
    promotion_options promotions;
    
    template<bool C>
    std::vector<vec4> gen_movable_pieces_impl(const std::vector<int> &lines) const;
    
    /*
     find_checks_impl<C>(lines)
     For all boards on the end of timelines specified in `lines` with color `C`,
     test if one of piece on that board with color `C` can capture an enemy royal piece.
     */
    template<bool C>
    generator<full_move> find_checks_impl(std::vector<int> lines) const &;

public:
    explicit state(multiverse mtv, promotion_options promotions = promotion_options::QUEEN) noexcept;
    // Load the main line. UNSAFE skips only royal-exposure checks.
    template<bool UNSAFE = false>
    static state from_pgn(const std::string &pgn);
    template<bool UNSAFE = false>
    static state from_ast(const pgnparser_ast::game &g);

    // standard copy-constructors
    state(const state &other) = default;
    state(state &&) noexcept = default;
    state &operator=(state other) noexcept {
        swap(*this, other);
        return *this;
    }
    friend void swap(state &a, state &b) noexcept {
        std::swap(a.m, b.m);
        std::swap(a.present, b.present);
        std::swap(a.promotions, b.promotions);
    }


    /*
     can_apply: Check if the move can be applied to the current state. If yes, return the new state after applying the move; otherwise return std::nullopt.
     Move overloads mirror checked apply_move promotion handling and do not submit.
     The action overload also performs checked submission. All overloads leave
     the source state unchanged.
    */
    std::optional<state> can_apply(full_move fm) const;
    std::optional<state> can_apply(ext_move mv) const;
    std::optional<state> can_apply(const action &act) const;
    bool test_present() const;
    std::optional<state> can_submit() const;
    
    /*
     apply_move: Apply move to the current state as a side effect. Return true if it is successfull.
     The full_move overload resolves the configured promotion before applying
     the move in either safety mode. The ext_move overload expects an explicit,
     normalized promotion choice; NO_PIECE means no promotion, not the default.
     UNSAFE=true skips validation and trusts the supplied move's legality.
     */
    template<bool UNSAFE = false>
    bool apply_move(full_move fm);
    template<bool UNSAFE = false>
    bool apply_move(ext_move mv);
    /*
     submit(): Submit the current state as a completed action.
     + UNSAFE=false requires present advancement and royal safety. Failure
     leaves the state unchanged. test_present() checks only turn completion.
     + UNSAFE=true skips the validation.
     */
    template<bool UNSAFE = false>
    bool submit();
    
    /*
     Pass on active boards at the stored present to distinguish checkmate
     from stalemate. The returned state retains its stored present.
     */
    state phantom() const;
    // Equivalent to phantom().find_checks(!get_present().c()).first(),
    // tested for presence, without cloning the multiverse. Includes physical checks.
    bool has_phantom_check() const;

    /*
     new_line(): return the index of a new line to be created by the current player.
    */
    int new_line() const noexcept;
    
    /*
     get_timeline_status() returns `std::make_tuple(mandatory_timelines, optional_timelines, unplayable_timelines)`
     where:
     mandatory_timelines are the timelines that current player must make a move on it
     optional_timelines are the timelines that current player can choose to play or not
     unplayable_timelines are the timelines that current player can't place a move on
     */
    std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> get_timeline_status() const;
    std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> get_timeline_status(int present_t, bool present_c) const;
    
    /*
     find_checks(): Generate moves by `attacker` that capture an enemy royal piece.
     */
    generator<full_move> find_checks(bool attacker) const &;
    generator<full_move> find_checks(bool attacker) const && = delete;

    // Initial setups scan both attacker colors. After submit(), action
    // validation scans only the current player (the previous mover's opponent).
    // A missing result means no exposure; this does not throw a validation error.
    std::optional<pgn_validation_error::royal_exposed> find_royal_exposure(
        bool is_from_initial_setup) const;

    std::vector<vec4> get_movable_pieces() const;
    std::vector<vec4> get_movable_pieces(const std::vector<int> &lines) const;
    std::vector<vec4> get_all_pieces(const std::vector<int> &lines) const;
    
    
    // A true witness means a legal action has already been established.
    mate_type get_mate_type(bool legal_action_witness = false) const;

    // wrappers for low-level functions
    std::pair<int, int> get_board_size() const noexcept;
    promotion_options get_promotion_options() const noexcept { return promotions; }
    // Resolve the requested promotion; NO_PIECE requests the configured default.
    // Non-promotion moves return an ext_move with NO_PIECE.
    // The checked mode rejects out-of-bounds squares
    // and absent source boards. UNSAFE trusts those structural prerequisites.
    // Neither mode validates movement or royal safety.
    template<bool UNSAFE = false>
    std::optional<ext_move> normalize_promotion(full_move move, piece_t requested = NO_PIECE) const;
    turn get_present() const noexcept;
    turn apparent_present() const noexcept;
    std::pair<int, int> get_initial_lines_range() const noexcept;
    std::pair<int, int> get_lines_range() const noexcept;
    std::pair<int, int> get_active_range() const noexcept;
    turn get_timeline_start(int l) const;
    turn get_timeline_end(int l) const;
    piece_t get_piece(vec4 p, bool color) const;
    // Share ownership of a read-only board; invalid indices throw.
    std::shared_ptr<const board> get_board(int l, int t, bool c) const;
    // Borrow a board, or return nullptr when the turn is absent.
    const board *get_board_ptr(int l, int t, bool c) const noexcept {
        return m.get_board_ptr(l,t,c);
    }
    std::vector<std::tuple<int,int,bool,std::string>> get_boards() const;
    // Generators borrow this state; moving, swapping, or modifying it invalidates them.
    generator<vec4> gen_piece_move(vec4 p) const &;
    generator<vec4> gen_piece_move(vec4 p) const && = delete;
    std::string to_string() const;
    std::string show_fen() const;
    std::string pretty_l(int l) const;
    std::string pretty_lt(vec4 p0) const;
    
    /* apply_move_ast: Resolve, normalize promotion, and apply one AST move. Does not submit.
     Throws pgn_validation_error; source-free ASTs use a fallback description. */
    ext_move apply_move_ast(const pgnparser_ast::move &move,
        const std::shared_ptr<const std::string> &source_text = nullptr);

    /*
    parse_move: Given a state `s` and a move in string format `move`, try to parse the move and match it to a unique full_move in the context of state `s`.
    - If successful, return a tuple with first index set to the matched full_move and second index set to the promotion piece if any.
    - If failed, return a tuple with first two indices set to nullopt and the third indices containing all possible matching full_moves. (.size()>1 ~> ambiguous; .size()==0 ~> cannot parse/no match)
    */
    using parse_pgn_res = std::tuple<std::optional<full_move>, std::optional<piece_t>, std::vector<full_move>>;
    parse_pgn_res parse_move(const pgnparser_ast::move &move) const;
    parse_pgn_res parse_move(const std::string &move) const;
};

#endif //STATE_H
