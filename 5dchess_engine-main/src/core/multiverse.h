//
//  board_2d.h
//  5dchess_engine
//
//  Created by ftxi on 2024/12/5.
//

#ifndef MULTIVERSE_H
#define MULTIVERSE_H

#include <string>
#include <vector>
#include <tuple>
#include <utility>
#include <map>
#include <memory>
#include "turn.h"
#include "timeline_layout.h"
#include "board.h"
#include "vec4.h"
#include "generator.h"

using movegen_t = generator<std::pair<vec4, bitboard_t>>;
using boards_info_t = std::tuple<int,int,bool,std::string>; // l, t, color, fen

/*
 The multiverse class.

 Behavior of copying a multiverse object is just copy the vector of vectors of pointers to the boards. It does not perform deep-copy of a board object. (Which is expected.)

 Odd/even timeline rules are stored in a timeline_layout value.
 */
class multiverse
{
private:
    timeline_layout layout;
    int size_x, size_y; // board size
    std::vector<std::vector<std::shared_ptr<const board>>> boards;
    // the following data are derivated from boards:
    int l_min, l_max, active_min, active_max;
    std::vector<turn> timeline_start, timeline_end;

    // private methods for move generation
    template<piece_t P, bool C>
    bitboard_t gen_physical_moves_impl(vec4 p) const;

    template<piece_t P, bool C, bool ONLY_SP>
    movegen_t gen_moves_impl(vec4 p) const &;

    template<bool C>
    generator<vec4> gen_board_move_impl(vec4 p0) const &;

    /*
     generate sliding moves in directions that are:
      + starting from `p`
      + in superphysical directions, moves as axesmode `TL`
        - when `TL` is `ORTHOGONAL`, moves to p+(*,*,1,0), p+(*,*,1,0), etc.
        - when `TL` is `DIAGONAL`, moves to p+(*,*,1,1), p+(*,*,1,-1), etc.
        - `BOTH` means both of the above
      + in physical directions, moves as axesmode `XY`
        - when `XY` is `ORTHOGONAL`, moves to p+(1,0,*,*), p+(0,1,*,*), etc.
        - when `XY` is `DIAGONAL`, moves to p+(1,1,*,*), p+(1,-1,*,*), etc.
        - `BOTH` means both of the above
     */
    enum class axesmode {ORTHOGONAL, DIAGONAL, BOTH};
    template<bool C, axesmode TL, axesmode XY>
    void gen_compound_moves(vec4 p, std::map<vec4, bitboard_t> &result) const;

    template<bool C>
    std::vector<std::pair<vec4, bitboard_t>> gen_purely_sp_rook_moves(vec4 p0) const;

    template<bool C>
    std::vector<std::pair<vec4, bitboard_t>> gen_purely_sp_bishop_moves(vec4 p0) const;

    template<bool C>
    std::vector<std::pair<vec4, bitboard_t>> gen_purely_sp_knight_moves(vec4 p0) const;

    void insert_board_impl(int l, int t, bool c, const std::shared_ptr<const board> &b_ptr);
    std::pair<int,int> calculate_active_range() const noexcept;
public:
    // constructor
    multiverse(timeline_layout layout, std::vector<boards_info_t> boards,
               int size_x = BOARD_LENGTH, int size_y = BOARD_LENGTH);

    // Stored boards are shared snapshots; do not modify them through retained aliases.
    // modifiers
    void insert_board(int l, int t, bool c, const std::shared_ptr<const board> &b_ptr);
    void append_board(int l, const std::shared_ptr<const board> &b_ptr);

    // getters
    std::pair<int, int> get_board_size() const noexcept;
    std::pair<int, int> get_initial_lines_range() const noexcept { return layout.initial_range(); }
    std::pair<int, int> get_lines_range() const noexcept;
    std::pair<int, int> get_active_range() const noexcept;
    turn get_timeline_start(int l) const;
    turn get_timeline_end(int l) const;

    // Share ownership of a read-only board; invalid indices throw.
    std::shared_ptr<const board> get_board(int l, int t, bool c) const;
    // Borrow an existing board without changing its shared ownership count.
    // Return nullptr when the turn is absent; valid while this multiverse lives.
    const board *get_board_ptr(int l, int t, bool c) const noexcept;

    template<bool SHOW_UMOVE=false>
    std::vector<boards_info_t> get_boards() const;

    std::string to_string() const;
    piece_t get_piece(vec4 a, bool color) const;
    bool get_umove_flag(vec4 a, bool color) const noexcept;

    /*
     This helper function returns (present_t, present_c)
     where: present_t is the time of present in L,T coordinate
            present_c is either false (for white) or true (for black)
     */
    turn get_present() const noexcept;

    // move generation
    template<bool C> bitboard_t gen_physical_moves(vec4 p) const;
    template<bool C> movegen_t gen_superphysical_moves(vec4 p) const &;
    template<bool C> movegen_t gen_superphysical_moves(vec4 p) const && = delete;
    template<bool C> movegen_t gen_moves(vec4 p) const &;
    template<bool C> movegen_t gen_moves(vec4 p) const && = delete;
    generator<vec4> gen_piece_move(vec4 p, bool board_color) const &;
    generator<vec4> gen_piece_move(vec4 p, bool board_color) const && = delete;

    // help functions
    bool inbound(vec4 a, bool color) const noexcept;
    std::string pretty_l(int l) const { return layout.pretty_l(l); }
    std::string pretty_lt(vec4 p0) const
    {
        return "(" + pretty_l(p0.l()) + "T" + std::to_string(p0.t()) + ")";
    }
};

#endif /* MULTIVERSE_H */
