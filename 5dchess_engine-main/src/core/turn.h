#ifndef TURN_H
#define TURN_H

#include <bit>
#include <cassert>
#include <compare>
#include <limits>
#include <string>
#include <utility>

class turn
{
    int v = 0;

    constexpr explicit turn(int v) noexcept : v(v) {}

public:
    const static turn max;

    constexpr turn() noexcept = default;

    constexpr turn(int t, bool c) noexcept
        : v(std::bit_cast<int>(static_cast<unsigned int>(t) << 1
                              | static_cast<unsigned int>(c)))
    {
        assert(0 <= t && t <= std::numeric_limits<int>::max() / 2);
    }

    constexpr static turn from_index(int v) noexcept
    {
        assert(v >= 0);
        return turn(v);
    }

    constexpr int index() const noexcept { return v; }
    constexpr int t() const noexcept { return v >> 1; }
    constexpr bool c() const noexcept { return static_cast<bool>(v & 1); }
    constexpr std::pair<int, bool> get() const noexcept { return {t(), c()}; }

    constexpr bool operator==(const turn &other) const noexcept = default;
    constexpr auto operator<=>(const turn &other) const noexcept = default;

    constexpr turn next_turn() const noexcept
    {
        assert(v < std::numeric_limits<int>::max());
        return turn(v + 1);
    }

    constexpr turn previous_turn() const noexcept
    {
        assert(v > 0);
        return turn(v - 1);
    }

    constexpr turn &operator++() noexcept { return *this = next_turn(); }
    constexpr turn &operator--() noexcept { return *this = previous_turn(); }
    constexpr turn operator++(int) noexcept
    {
        const turn old = *this;
        ++*this;
        return old;
    }
    constexpr turn operator--(int) noexcept
    {
        const turn old = *this;
        --*this;
        return old;
    }

    std::string to_string() const
    {
        return std::to_string(t()) + (c() ? "b" : "w");
    }
};

inline constexpr turn turn::max = turn::from_index(std::numeric_limits<int>::max());

enum class mate_type {NONE, CHECKMATE, SOFTMATE, STALEMATE};

#endif /* TURN_H */
