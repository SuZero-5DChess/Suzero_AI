#include "hash.h"
#include <array>
#include "board.h"
#include "multiverse.h"
#include "state.h"

namespace
{
// SplitMix64's public-domain mixing function by Sebastiano Vigna:
// https://prng.di.unimi.it/splitmix64.c
constexpr std::uint64_t mix(std::uint64_t value) noexcept
{
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

class accumulator
{
    std::uint64_t value = UINT64_C(0x9e3779b97f4a7c15);

public:
    constexpr void add(std::uint64_t word) noexcept
    {
        value = mix(value + word + UINT64_C(0x9e3779b97f4a7c15));
    }

    constexpr std::uint64_t result() const noexcept { return value; }
};

void append_board(accumulator &hash, const board &b) noexcept
{
    hash.add(UINT64_C(0x626f6172642d7631)); // "board-v1"
    const std::array fields{
        b.white(), b.black(), b.royal(),
        b.lking(), b.lknight(), b.lpawn(), b.lrawn(),
        b.lrook(), b.lbishop(), b.lunicorn(), b.ldragon(),
        b.umove()
    };
    for(bitboard_t field : fields)
        hash.add(field);
}

void append_multiverse(accumulator &hash, const multiverse &m)
{
    hash.add(UINT64_C(0x6d74762d7632)); // "mtv-v2"
    // The supported layouts are odd (0,0) and even (-1,0). Encode the
    // distinction without hashing the bounds themselves.
    const auto initial_lines = m.get_initial_lines_range();
    hash.add(initial_lines.first != initial_lines.second);

    // Shape ranges only bound traversal. Logical coordinates and board
    // contents already describe the shape, including historical boards.
    const auto [l_min, l_max] = m.get_lines_range();
    for(int l = l_min; l <= l_max; ++l)
    {
        const turn start = m.get_timeline_start(l);
        const turn end = m.get_timeline_end(l);
        for(turn at = start; at <= end; at = at.next_turn())
        {
            const board *b = m.get_board_ptr(l, at.t(), at.c());
            if(b == nullptr)
                continue;
            // Signed-to-uint64_t conversion is defined modulo 2^64, including
            // for negative timeline coordinates, independently of byte order.
            hash.add(static_cast<std::uint64_t>(l));
            hash.add(static_cast<std::uint64_t>(at.t()));
            hash.add(at.c());
            append_board(hash, *b);
        }
    }
}
} /* anonymous namespace */

std::uint64_t hash_board(const board &b) noexcept
{
    accumulator hash;
    append_board(hash, b);
    return hash.result();
}

std::uint64_t hash_multiverse(const multiverse &m)
{
    accumulator hash;
    append_multiverse(hash, m);
    return hash.result();
}

std::uint64_t hash_state(const state &s)
{
    accumulator hash;
    hash.add(UINT64_C(0x73746174652d7632)); // "state-v2"
    append_multiverse(hash, s.m);
    const auto [present, player] = s.get_present().get();
    hash.add(static_cast<std::uint64_t>(present));
    hash.add(player);
    hash.add(static_cast<std::uint8_t>(s.get_promotion_options()));
    return hash.result();
}
