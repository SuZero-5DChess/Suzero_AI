#ifndef VARIANTS_H
#define VARIANTS_H

#include <map>
#include <string>
#include <tuple>
#include <vector>
#include "ast.h"
#include "promotion_header.h"
#include "multiverse.h"

struct variant_setup_t
{
    int size_x;
    int size_y;
    std::vector<std::tuple<std::string, pgnparser_ast::token_t, int, int, bool>> boards;
    bool is_even_timelines;
    promotion_options promotions = promotion_options::QUEEN;
};

extern const std::map<std::string, variant_setup_t> default_variants;

multiverse create_multiverse_from_variant_setup(const variant_setup_t &variant_setup);
#endif // VARIANTS_H
