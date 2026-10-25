#include "variants.h"
#include <algorithm>

const std::map<std::string, variant_setup_t> default_variants = {
    {
        "Standard",
        {
            8,8,
            {
                std::make_tuple("r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*", pgnparser_ast::NIL, 0, 1, false)
            },
            false
        }
    },
    {
        "Standard - Turn Zero",
        {
            8,8,
            {
                std::make_tuple("r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*", pgnparser_ast::NIL, 0, 0, true),
                std::make_tuple("r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*", pgnparser_ast::NIL, 0, 1, false),
            },
            false
        }
    },
    {
        "Standard - Reversed Royalty",
        {
            8,8,
            {
                std::make_tuple("r*nbycbnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBYCBNR*", pgnparser_ast::NIL, 0, 1, false)
            },
            false
        }
    },
    {
        "Standard - Half Reflected",
        {
            8,8,
            {
                std::make_tuple("r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBK*QBNR*", pgnparser_ast::NIL, 0, 1, false)
            },
            false
        }
    },
    {
        "Standard - Defended Pawn",
        {
            8,8,
            {
                std::make_tuple("r*qbnk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*QBNK*BNR*", pgnparser_ast::NIL, 0, 1, false)
            },
            false
        }
    },
    {
        "Standard - Two Timelines",
        {
            8,8,
            {
                std::make_tuple("r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*", pgnparser_ast::NEGATIVE, 0, 1, false),
                std::make_tuple("r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*", pgnparser_ast::POSITIVE, 0, 1, false),
            },
            true
        }
    },
    {
        "Very Small - Open",
        {
            4,4,
            {
                std::make_tuple("nbrk/3p*/P*3/KRBN", pgnparser_ast::NIL, 0, 1, false),
            },
            false
        }
    },
};

multiverse create_multiverse_from_variant_setup(const variant_setup_t &variant_setup)
{
    const auto layout = variant_setup.is_even_timelines ? timeline_layout::even : timeline_layout::odd;
    std::vector<boards_info_t> boards_info(variant_setup.boards.size());
    std::transform(variant_setup.boards.begin(), variant_setup.boards.end(), boards_info.begin(), [layout](const auto &tup) {
        const auto &[fen, sign, l, t, c] = tup;
        const int signed_l = sign == pgnparser_ast::NEGATIVE ? layout.mirror(l) : l;
        return std::make_tuple(signed_l, t, c, fen);
    });
    return multiverse(layout, std::move(boards_info), variant_setup.size_x, variant_setup.size_y);
}
