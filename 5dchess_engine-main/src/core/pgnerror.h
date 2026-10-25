#ifndef PGNERROR_H
#define PGNERROR_H

#include "action.h"
#include "ast.h"
#include "promotion_header.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

class board_fen_error : public std::invalid_argument
{
public:
    struct invalid_shape {
        std::size_t expected;
        std::size_t actual;
    };
    struct rank_too_short : invalid_shape {};
    struct rank_too_long : invalid_shape {};
    struct too_few_ranks : invalid_shape {};
    struct too_many_ranks : invalid_shape {};
    struct invalid_piece {};
    struct invalid_unmoved_marker {};

    using error_type_t = std::variant<rank_too_short, rank_too_long,
        too_few_ranks, too_many_ranks, invalid_piece, invalid_unmoved_marker>;

private:
    error_type_t error_type;
    std::string fen;
    std::size_t start, end;
    std::shared_ptr<const std::string> pgn_source = nullptr;
    std::optional<pgnparser_ast::source_range> pgn_range = std::nullopt;

    static std::string generate_errmsg(const error_type_t &error_type, std::string_view fen,
        std::size_t start, std::size_t end);
public:
    // Byte offsets into fen, with an exclusive end; an empty range marks missing input.
    board_fen_error(error_type_t error_type, std::string fen, std::size_t start, std::size_t end);

    // Attach the enclosing PGN location without changing FEN-relative offsets.
    board_fen_error(const board_fen_error &error, std::shared_ptr<const std::string> source,
                    pgnparser_ast::source_range range);
    const std::shared_ptr<const std::string> &get_pgn_source() const noexcept { return pgn_source; }
    const std::optional<pgnparser_ast::source_range> &get_pgn_range() const noexcept { return pgn_range; }
    const error_type_t &get_error_type() const noexcept { return error_type; }
    const std::string &get_fen() const noexcept { return fen; }
    std::size_t get_start() const noexcept { return start; }
    std::size_t get_end() const noexcept { return end; }
    std::string_view get_range() const { return std::string_view(fen).substr(start, end - start); }
};

class parse_error : public std::invalid_argument
{
public:
    struct unexpected_character {};
    struct unexpected_input {
        std::vector<std::string> expected;
    };
    struct unterminated_comment {};
    struct unterminated_metadata {};
    struct unterminated_header_value {};
    struct invalid_number {
        std::int64_t minimum;
        std::int64_t maximum;
    };
    struct integer_out_of_range : invalid_number {};
    struct turn_out_of_range : invalid_number {};
    struct timeline_out_of_range : invalid_number {};
    struct nonconsecutive_turn {
        turn expected;
        turn actual;
    };
    struct duplicate_header {
        std::string key;
    };

    using error_type_t = std::variant<unexpected_character, unexpected_input,
        unterminated_comment, unterminated_metadata, unterminated_header_value,
        integer_out_of_range, turn_out_of_range, timeline_out_of_range,
        nonconsecutive_turn, duplicate_header>;

private:
    error_type_t error_type;
    std::shared_ptr<const std::string> source;
    pgnparser_ast::source_range range;

    static std::string generate_errmsg(
        const error_type_t &error_type,
        const std::shared_ptr<const std::string> &source,
        pgnparser_ast::source_range range
    );

public:
    // Source must be non-null. Byte offsets use an exclusive end;
    // an empty range marks missing input, including EOF.
    parse_error(
        error_type_t error_type,
        std::shared_ptr<const std::string> source,
        pgnparser_ast::source_range range
    );

    const error_type_t &get_error_type() const noexcept { return error_type; }
    const std::string &get_source() const noexcept { return *source; }
    std::size_t get_start() const noexcept { return range.start; }
    std::size_t get_end() const noexcept { return range.end; }
    std::string_view get_range() const {
        return std::string_view(*source).substr(range.start, range.end - range.start);
    }
};

class pgn_validation_error : public std::invalid_argument
{
public:
    struct candidate {
        ext_move move;
        std::string notation;
    };
    struct invalid_move {};
    struct ambiguous_move {
        std::vector<candidate> candidates;
    };
    struct illegal_promotion {
        piece_t requested;
        promotion_options allowed;
    };
    struct illegal_move {
        ext_move move;
    };
    struct present_not_advanced {};
    struct royal_exposed {
        bool is_from_initial_setup;
        full_move capture;
        std::string notation;
    };

    struct invalid_size {
        enum class reason { format, number, out_of_range };
        reason cause;
        int maximum;
    };
    struct unknown_variant {};
    struct invalid_timeline {};
    struct invalid_promotions {
        ::invalid_promotions::reason cause;
        char piece;
    };
    struct missing_setup {};

    using error_type_t = std::variant<invalid_move, ambiguous_move,
        illegal_promotion, illegal_move, present_not_advanced, royal_exposed,
        invalid_size, unknown_variant, invalid_timeline, invalid_promotions, missing_setup>;

    struct source_location {
        std::shared_ptr<const std::string> source;
        pgnparser_ast::source_range range;
    };
    struct description {
        std::string text;
    };
    using subject_t = std::variant<source_location, description>;

private:
    error_type_t error_type;
    subject_t subject;

    static std::string generate_errmsg(const error_type_t &error_type, const subject_t &subject);

public:
    // Source locations require non-null text and a valid exclusive byte range.
    // Ambiguous moves require at least two candidates with distinct moves.
    pgn_validation_error(error_type_t error_type, subject_t subject);

    const error_type_t &get_error_type() const noexcept { return error_type; }
    const subject_t &get_subject() const noexcept { return subject; }
    const source_location *get_location() const noexcept {
        return std::get_if<source_location>(&subject);
    }
    std::string_view get_text() const;

    static subject_t make_subject(const pgnparser_ast::move &move,
        const std::shared_ptr<const std::string> &source = nullptr);
    static subject_t make_subject(const pgnparser_ast::actions &actions,
        const std::shared_ptr<const std::string> &source = nullptr);
};

#endif /* PGNERROR_H */
