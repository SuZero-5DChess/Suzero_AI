#include "pgnerror.h"
#include <algorithm>
#include <iomanip>
#include <set>
#include <sstream>
#include <type_traits>
#include <utility>

std::string board_fen_error::generate_errmsg(const error_type_t &error_type, std::string_view fen,
    std::size_t start, std::size_t end)
{
    if(start > end || end > fen.size())
        throw std::invalid_argument("board_fen_error: source range is outside the FEN");

    return std::visit([&]<typename T>(const T &reason) -> std::string {
        if constexpr(std::is_same_v<T, rank_too_short> || std::is_same_v<T, rank_too_long>)
            return "FEN rank has " + std::to_string(reason.actual)
                + " squares; expected " + std::to_string(reason.expected) + ".";
        else if constexpr(std::is_same_v<T, too_few_ranks> || std::is_same_v<T, too_many_ranks>)
            return "FEN rank count is " + std::to_string(reason.actual)
                + "; expected " + std::to_string(reason.expected) + ".";
        else if constexpr(std::is_same_v<T, invalid_piece>)
            return "Invalid FEN piece: \"" + std::string(fen.substr(start, end - start)) + "\".";
        else if constexpr(std::is_same_v<T, invalid_unmoved_marker>)
            return "Invalid unmoved marker in FEN: \"" + std::string(fen.substr(start, end - start))
                + "\". Only K, R, P, W, k, r, p, and w may be followed by '*'.";
    }, error_type);
}

board_fen_error::board_fen_error(error_type_t error_type, std::string fen,
    std::size_t start, std::size_t end)
    : std::invalid_argument(generate_errmsg(error_type, fen, start, end)),
      error_type(error_type), fen(std::move(fen)), start(start), end(end)
{}

board_fen_error::board_fen_error(const board_fen_error &error,
    std::shared_ptr<const std::string> source, pgnparser_ast::source_range range)
    : board_fen_error(error)
{
    if(!source || range.start > range.end || range.end > source->size())
        throw std::invalid_argument("board_fen_error: source range is outside the PGN");
    pgn_source = std::move(source);
    pgn_range = range;
}

std::string parse_error::generate_errmsg(const error_type_t &error_type,
    const std::shared_ptr<const std::string> &source, pgnparser_ast::source_range range)
{
    if(!source || range.start > range.end || range.end > source->size())
        throw std::invalid_argument("parse_error: source range is outside the input");

    const auto found = [&]() -> std::string {
        if(range.start == source->size()) return "end of input";
        if(range.start == range.end) return "missing input";
        std::ostringstream oss;
        oss << std::quoted(source->substr(range.start, range.end - range.start));
        return oss.str();
    };
    return std::visit([&]<typename T>(const T &reason) -> std::string {
        if constexpr(std::is_same_v<T, unexpected_character>)
            return "Unrecognized character: " + found() + ".";
        else if constexpr(std::is_same_v<T, unexpected_input>) {
            std::string expected;
            for(const auto &item : reason.expected) {
                if(!expected.empty()) expected += " or ";
                expected += item;
            }
            return "Expected " + expected + "; found " + found() + ".";
        }
        else if constexpr(std::is_same_v<T, unterminated_comment>)
            return "Comment is not closed; expected '}'.";
        else if constexpr(std::is_same_v<T, unterminated_metadata>)
            return "Metadata entry is not closed; expected ']'.";
        else if constexpr(std::is_same_v<T, unterminated_header_value>)
            return "Header value is not closed; expected a closing quote.";
        else if constexpr(std::is_base_of_v<invalid_number, T>) {
            std::string kind;
            if constexpr(std::is_same_v<T, integer_out_of_range>) kind = "Integer";
            else if constexpr(std::is_same_v<T, turn_out_of_range>) kind = "Turn number";
            else kind = "Timeline number";
            return kind + " " + found() + " is out of range; expected "
                + std::to_string(reason.minimum) + " to " + std::to_string(reason.maximum) + ".";
        }
        else if constexpr(std::is_same_v<T, nonconsecutive_turn>) {
            return "Expected turn " + reason.expected.to_string() + "; found " + reason.actual.to_string() + ".";
        }
        else if constexpr(std::is_same_v<T, duplicate_header>)
            return "Duplicate header key: " + reason.key + ".";
    }, error_type);
}

parse_error::parse_error(error_type_t error_type, std::shared_ptr<const std::string> source,
    pgnparser_ast::source_range range)
    : std::invalid_argument(generate_errmsg(error_type, source, range)),
      error_type(std::move(error_type)), source(std::move(source)), range(range)
{}

namespace
{
std::string_view subject_text(const pgn_validation_error::subject_t &subject)
{
    using error = pgn_validation_error;
    return std::visit(overloads {
        [](const error::source_location &location) -> std::string_view {
            const auto [start, end] = location.range;
            if(!location.source || start > end || end > location.source->size())
                throw std::invalid_argument("pgn_validation_error: source range is outside the input");
            return std::string_view(*location.source).substr(start, end - start);
        },
        [](const error::description &description) -> std::string_view {
            return description.text;
        }
    }, subject);
}

template<typename T>
pgn_validation_error::subject_t make_subject(const T &node,
    const std::shared_ptr<const std::string> &source)
{
    if(source && node.source)
        return pgn_validation_error::source_location{source, *node.source};
    std::ostringstream text;
    text << node;
    return pgn_validation_error::description{text.str()};
}
} /* anonymous namespace */

std::string pgn_validation_error::generate_errmsg(const error_type_t &error_type, const subject_t &subject)
{
    const std::string text(subject_text(subject));
    return std::visit([&]<typename T>(const T &reason) -> std::string {
        if constexpr(std::is_same_v<T, invalid_size>) {
            if(reason.cause == invalid_size::reason::format)
                return "Invalid Size value: \"" + text + "\". Expected widthxheight.";
            if(reason.cause == invalid_size::reason::number)
                return "Invalid Size value: \"" + text + "\". Width and height must be numbers.";
            return "Invalid Size value: \"" + text + "\". Width and height must be between 1 and "
                + std::to_string(reason.maximum) + ".";
        }
        else if constexpr(std::is_same_v<T, unknown_variant>)
            return "Unknown variant: \"" + text + "\".";
        else if constexpr(std::is_same_v<T, invalid_timeline>)
            return "Invalid Timeline value: \"" + text + "\". Expected Even or Odd.";
        else if constexpr(std::is_same_v<T, invalid_promotions>)
            return "Invalid Promotions value: \"" + text + "\". "
                + ::invalid_promotions(reason.cause, reason.piece).what();
        else if constexpr(std::is_same_v<T, missing_setup>)
            return "Missing initial setup: specify a Board or Variant header naming a variant, or provide 5DFEN boards.";
        else if constexpr(std::is_same_v<T, invalid_move>)
            return "Invalid move: " + text;
        else if constexpr(std::is_same_v<T, ambiguous_move>) {
            std::set<ext_move> distinct;
            for(const auto &candidate : reason.candidates)
                if(!distinct.insert(candidate.move).second)
                    throw std::invalid_argument("pgn_validation_error: duplicate ambiguity candidate");
            if(distinct.size() < 2)
                throw std::invalid_argument("pgn_validation_error: ambiguity requires two distinct candidates");
            std::string message = "Ambiguous move: " + text + "; possible candidates: ";
            for(std::size_t i = 0; i < std::min<std::size_t>(2, reason.candidates.size()); ++i) {
                if(i) message += ", ";
                message += reason.candidates[i].notation;
            }
            if(reason.candidates.size() > 2) message += ", ...";
            return message + " (totally " + std::to_string(reason.candidates.size()) + ")";
        }
        else if constexpr(std::is_same_v<T, illegal_promotion>)
            return "Illegal promotion: " + text;
        else if constexpr(std::is_same_v<T, illegal_move>)
            return "Illegal move: " + text;
        else if constexpr(std::is_same_v<T, present_not_advanced>)
            return "Cannot submit action: " + text + "; the moves do not advance the present.";
        else if constexpr(std::is_same_v<T, royal_exposed>) {
            const std::string prefix = reason.is_from_initial_setup
                ? "Illegal initial setup: royal piece exposed to capture by "
                : "Illegal action: " + text + "; royal piece exposed to capture by ";
            return prefix + reason.notation + ".";
        }
    }, error_type);
}

pgn_validation_error::pgn_validation_error(error_type_t error_type, subject_t subject)
    : std::invalid_argument(generate_errmsg(error_type, subject)),
      error_type(std::move(error_type)), subject(std::move(subject))
{}

std::string_view pgn_validation_error::get_text() const
{
    return subject_text(subject);
}

pgn_validation_error::subject_t pgn_validation_error::make_subject(const pgnparser_ast::move &move,
    const std::shared_ptr<const std::string> &source)
{
    return ::make_subject(move, source);
}

pgn_validation_error::subject_t pgn_validation_error::make_subject(const pgnparser_ast::actions &actions,
    const std::shared_ptr<const std::string> &source)
{
    return ::make_subject(actions, source);
}
