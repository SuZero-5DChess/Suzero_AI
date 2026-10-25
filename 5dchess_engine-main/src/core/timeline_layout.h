#ifndef TIMELINE_LAYOUT_H
#define TIMELINE_LAYOUT_H

#include <string>
#include <utility>

// Odd timelines have one central line (0); even timelines have -0 and +0.
class timeline_layout
{
    enum { ODD, EVEN } type;

    constexpr explicit timeline_layout(decltype(type) type) noexcept : type(type) {}

public:
    const static timeline_layout odd;
    const static timeline_layout even;

    constexpr bool operator==(const timeline_layout &other) const noexcept = default;

    constexpr bool is_even() const noexcept { return type == EVEN; }

    // Reflect a timeline across the center, or convert a negative label's magnitude.
    constexpr int mirror(int l) const noexcept { return is_even() ? ~l : -l; }

    constexpr std::pair<int, int> initial_range() const noexcept { return {mirror(0), 0}; }

    std::string pretty_l(int l) const
    {
        if(!is_even())
            return std::to_string(l);
        return std::string(l >= 0 ? "+" : "-") + std::to_string(l >= 0 ? l : mirror(l));
    }
};

inline constexpr timeline_layout timeline_layout::odd{ODD};
inline constexpr timeline_layout timeline_layout::even{EVEN};

#endif /* TIMELINE_LAYOUT_H */
