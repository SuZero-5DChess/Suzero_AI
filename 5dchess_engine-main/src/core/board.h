//
//  board_2d.h
//  5dchess_backend
//
//  Created by ftxi on 2024/4/24.
//

#ifndef board_h
#define board_h

#include "pgnerror.h"
#include <iostream>
#include <array>
#include <string>
#include <stdexcept>
#include <cstddef>
#include <variant>
#include <string_view>
#include "piece.h"
#include "bitboard.h"

/*
The bitboards collection associated with a board.
 */
class board
{
    enum bitboard_indices {
        WHITE, BLACK, ROYAL, //flags
        LKING, LKNIGHT, LPAWN, LRAWN, //jumping pieces
        LROOK, LBISHOP, LUNICORN, LDRAGON, //sliding pieces
        BBS_INDICES_COUNT
    };
    std::array<bitboard_t, BBS_INDICES_COUNT> bbs;
    bitboard_t umove_mask;

public:
    board(std::string fen, int size_x = BOARD_LENGTH, int size_y = BOARD_LENGTH);
    // inline getter functions
    constexpr bitboard_t umove() const noexcept { return umove_mask; }

    constexpr bitboard_t white() const noexcept { return bbs[WHITE]; }
    constexpr bitboard_t black() const noexcept { return bbs[BLACK]; }
    constexpr bitboard_t wall() const noexcept { return bbs[WHITE] & bbs[BLACK]; }
    
    constexpr bitboard_t royal() const noexcept { return bbs[ROYAL]; }

    constexpr bitboard_t lking() const noexcept { return bbs[LKING]; }
    constexpr bitboard_t lknight() const noexcept { return bbs[LKNIGHT]; }
    constexpr bitboard_t lpawn() const noexcept { return bbs[LPAWN]; }
    constexpr bitboard_t lrawn() const noexcept { return bbs[LRAWN]; }

    constexpr bitboard_t lrook() const noexcept { return bbs[LROOK]; }
    constexpr bitboard_t lbishop() const noexcept { return bbs[LBISHOP]; }
    constexpr bitboard_t lunicorn() const noexcept { return bbs[LUNICORN]; }
    constexpr bitboard_t ldragon() const noexcept { return bbs[LDRAGON]; }

    constexpr bitboard_t king() const noexcept { return bbs[LKING] & bbs[ROYAL]; }
    constexpr bitboard_t common_king() const noexcept { return bbs[LKING] & ~bbs[ROYAL]; }
    constexpr bitboard_t knight() const noexcept { return bbs[LKNIGHT]; }
    constexpr bitboard_t pawn() const noexcept { return bbs[LPAWN] & ~bbs[LRAWN]; }
    constexpr bitboard_t brawn() const noexcept { return bbs[LPAWN] & bbs[LRAWN]; }
    constexpr bitboard_t rook() const noexcept { return bbs[LROOK] & ~bbs[LBISHOP]; }
    constexpr bitboard_t bishop() const noexcept { return bbs[LBISHOP] & ~bbs[LROOK]; }
    constexpr bitboard_t unicorn() const noexcept { return bbs[LUNICORN] & ~bbs[LDRAGON]; }
    constexpr bitboard_t dragon() const noexcept { return bbs[LDRAGON] & ~bbs[LUNICORN]; }
    constexpr bitboard_t princess() const noexcept { return bbs[LROOK] & bbs[LBISHOP] & ~bbs[LUNICORN]; }
    constexpr bitboard_t royal_queen() const noexcept { return bbs[LROOK] & bbs[LDRAGON] & bbs[ROYAL]; }
    constexpr bitboard_t queen() const noexcept { return bbs[LROOK] & bbs[LDRAGON] & ~bbs[ROYAL]; }
    
    // indirect getter functions
    constexpr bitboard_t sliding() const noexcept { return bbs[LROOK] | bbs[LBISHOP] | bbs[LUNICORN] | bbs[LDRAGON]; }
    constexpr bitboard_t occupied() const noexcept { return bbs[WHITE] | bbs[BLACK]; }
    
    template<bool C>
    constexpr bitboard_t friendly() const noexcept
    {
        // walls are considered as friendly
        // because friendly pieces are blockers
        if constexpr (C)
        {
            return bbs[BLACK];
        }
        else
        {
            return bbs[WHITE];
        }
    }
    
    template<bool C>
    constexpr bitboard_t hostile() const noexcept
    {
        if constexpr (C)
        {
            return bbs[WHITE] & ~bbs[BLACK]; // exclude walls
        }
        else
        {
            return bbs[BLACK] & ~bbs[WHITE];
        }
    }
    
    // modifications
    piece_t get_piece(int pos) const;
    void set_piece(int pos, piece_t p);
    std::shared_ptr<const board> replace_piece(int pos, piece_t p) const;
    std::shared_ptr<const board> move_piece(int from, int to) const;
    std::string to_string() const;
    
    template<bool SHOW_UMOVE=false>
    std::string get_fen(int size_x = BOARD_LENGTH, int size_y = BOARD_LENGTH) const;

    // all the pieces (both white and black) that attacks a given square
    bitboard_t attacks_to(int pos) const noexcept;
	// pieces hositile to `color` that attacks a given square
    bool is_under_attack(int pos, int color) const noexcept;

    // Unobstructed unmoved friendly rooks on the king's rank.
    bitboard_t castling_rooks(int pos, bool color) const noexcept;
};


#endif /* board_h */
