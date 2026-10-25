#ifndef MAGIC_H
#define MAGIC_H

#include "bitboard.h"

bitboard_t rook_attack(int pos, bitboard_t blocker) noexcept;
bitboard_t bishop_attack(int pos, bitboard_t blocker) noexcept;
bitboard_t queen_attack(int pos, bitboard_t blocker) noexcept;

#endif //MAGIC_H
