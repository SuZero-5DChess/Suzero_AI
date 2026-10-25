#undef NDEBUG
#include <cassert>
#include <iostream>
#include <vector>
#include <type_traits>
#include "pgnparser.h"
#include "utils.h"
#include "state.h"
#include "vec4.h"
#include "variants.h"

void test_board_coordinate_limits()
{
    const std::string fen = "8/8/8/8/8/8/8/8";
    auto parse = [&](const std::string &coordinates) {
        return pgnparser::parse_board_fen_metadata(fen + coordinates);
    };
    auto rejected = [&](const std::string &coordinates) {
        try
        {
            parse(coordinates);
            return false;
        }
        catch(const parse_error &)
        {
            return true;
        }
    };

    assert(std::get<3>(parse(":0:" + std::to_string(vec4::T_MAX) + ":w"))
           == vec4::T_MAX);
    assert(rejected(":0:" + std::to_string(vec4::T_MAX + 1) + ":w"));
    assert(rejected(":0:-1:w"));
    assert(std::get<2>(parse(":" + std::to_string(vec4::L_MAX) + ":0:w"))
           == vec4::L_MAX);
    assert(std::get<1>(parse(":-" + std::to_string(vec4::L_MAX) + ":0:w"))
           == pgnparser_ast::NEGATIVE);
    assert(rejected(":" + std::to_string(vec4::L_MAX + 1) + ":0:w"));
    assert(rejected(":-" + std::to_string(vec4::L_MAX + 1) + ":0:w"));
    assert(rejected(":0:5000:w"));
}

void test_short_board_rejected()
{
    const std::string input = R"(
[Size "8x8"]
[Board "Custom"]
[r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*:0:0:w]

1-0
)";
    try
    {
        state::from_pgn<true>(input);
        assert(false);
    }
    catch(const std::invalid_argument &)
    {
    }
}

void test_timeline_header()
{
    const std::string boards = R"(
[Board "Custom"]
[Size "8x8"]
[8/8/8/8/8/8/8/K6k:0:0:w]
)";
    auto setup = [&](const std::string &value) {
        auto game = pgnparser("[Timeline \"" + value + "\"]\n" + boards).parse_game();
        return state::from_ast<true>(*game);
    };

    assert(setup("Even").get_initial_lines_range() == timeline_layout::even.initial_range());
    assert(setup("odd").get_initial_lines_range() == timeline_layout::odd.initial_range());
    try
    {
        setup("Sideways");
        assert(false);
    }
    catch(const std::invalid_argument &)
    {
    }
}

void test_actions()
{
    pgnparser("1.{} {{ab {c}}d}\n / (2.{1987yhwqsi}))").test_lexer();
    std::cout << "void\n";
    std::cout << pgnparser("$(L+7T-6)").parse_relative_board() << "\n";
    std::cout << pgnparser("(L+7T6)").parse_absolute_board() << "\n";
    std::cout << pgnparser("(T5)").parse_absolute_board() << "\n";
    std::cout << pgnparser("(L+5T7)").parse_absolute_board() << "\n";
    std::cout << pgnparser("(2. Kf7)", false).parse_absolute_board() << "\n";
    std::cout << pgnparser("(16b.Qxg5 17.", false).parse_absolute_board() << "\n";
    std::cout << pgnparser("16b.Qxg5 17.", false).parse_actions() << "\n";
    std::cout << pgnparser(R"(16b.(-2)Kxc3{}(-1)Qxg5    (0T9)f3 {bloody}{gibberish} (1T7)e2>>$(T-1)b1+~??? {%command1%}
{new-line {nested}}
 
 17.)", false).parse_actions() << "\n";
    std::cout << pgnparser("16b. R5g6 c3(17.Ke5)", false).parse_actions() << "\n";
    
    std::cout << pgnparser("(L+7T6)Q>>a3+~?!!(~T5)(>L-10)").parse_move() << "\n";
//    std::cout << pgnparser("16b. R5g6 c3>>(17.Ke5)", false).parse_actions() << "\n"; //error
//    std::cout << pgnparser("c3>>(17.Ke5)", false).parse_move() << "\n"; //error
}

void test_gametree()
{
    std::string str=R"(1w.e3 ( 1b. e6 {f7-sac}2.f3 / Nf6 3.Bb5 / Pc6
)(1b. c6 {branch2}) 1b.Nf6 {main} 2.Nf3 / d5 3.c4 / c6 4.cxd5/Qxd5
)";
    //std::string str = "1w.e3 (1b.e6 2.f3/Nf6) (1b. c6 {falsch}) 1b.Nf6";
    std::cout << pgnparser(str).parse_gametree() << "\n";
//    std::cout << pgnparser("1w. e3 (1b.e6) 1b.Nf6 {main} 2.Nf3 / d5 3.c4 / c6 4.cxd5/Qxd5", true, {1,false}).parse_gametree() << "\n";
}

void test_game()
{
    std::string str = R"(
[Mode "5D"]
[Board "Standard"]
1. e3 / Nf6
2w. Bb5 {Beware!}
(2b. d5 {The right response})
2b. c6
3. c3 / cxb5
4. Qb3 / Qa5
5. Q>>xf7+~ (~T1) (>L1) {f7-sacrifice!} / (1T1)Kxf7
6. (1T2)Nh3 / (1T2)e6
7. (1T3)e3 / (1T3)Qf6
8. (1T4)Qh5*
)";
    std::string str2 = R"(
[Size "4x4"]
[Board "Custom"]
[Mode "5D"]
[nbrk/3p*/P*3/KRBN:0:1:w]
)";
    pgnparser_ast::game g = *pgnparser(str2).parse_game();
    std::cout << g << "\n\n" << std::endl;
    state s = state::from_ast<true>(g);
    std::cout << s.to_string();
}

namespace
{
parse_error parse_failure(const std::string &input, bool metadata = false)
{
    try {
        if(metadata)
            pgnparser::parse_board_fen_metadata(input);
        else
            pgnparser(input).parse_game();
    }
    catch(const parse_error &error) {
        assert(error.get_source() == input);
        assert(error.get_start() <= error.get_end() && error.get_end() <= input.size());
        assert(error.get_range() == std::string_view(error.get_source()).substr(
            error.get_start(), error.get_end() - error.get_start()));
        return error;
    }
    throw std::logic_error("Expected a parse_error");
}

template<typename Reason>
void check_failure(const std::string &input, std::size_t start, std::size_t end)
{
    const auto error = parse_failure(input);
    assert(std::holds_alternative<Reason>(error.get_error_type()));
    if(error.get_start() != start || error.get_end() != end)
        std::cerr << input << ": expected range " << start << ':' << end
                  << ", got " << error.get_start() << ':' << error.get_end() << '\n';
    assert(error.get_start() == start && error.get_end() == end);
}

void test_parse_diagnostics()
{
    static_assert(std::is_base_of_v<std::invalid_argument, parse_error>);
    check_failure<parse_error::unexpected_character>("1. e4 @", 6, 7);
    check_failure<parse_error::unexpected_input>("0-2", 2, 3);
    check_failure<parse_error::unexpected_input>("0-", 2, 2);
    check_failure<parse_error::unexpected_input>("1-", 2, 2);
    check_failure<parse_error::unexpected_input>("1/2-", 0, 4);
    check_failure<parse_error::unexpected_input>("O", 0, 1);
    check_failure<parse_error::unterminated_comment>("1. e4 {nested {", 6, 15);
    check_failure<parse_error::unterminated_metadata>("[Board", 0, 6);
    check_failure<parse_error::unterminated_header_value>("[Board \"Standard]", 7, 16);
    check_failure<parse_error::unexpected_input>("1. e8=", 6, 6);
    check_failure<parse_error::unexpected_input>("1. Nf", 5, 5);
    check_failure<parse_error::unexpected_input>("(1. e4", 6, 6);
    check_failure<parse_error::unexpected_input>("1. e4 /", 7, 7);
    check_failure<parse_error::unexpected_input>("1. e4 >", 7, 7);
    check_failure<parse_error::integer_out_of_range>("999999999999999999999999", 0, 24);
    const auto turn_literal = std::to_string(vec4::T_MAX + 1);
    check_failure<parse_error::turn_out_of_range>(turn_literal + ".", 0, turn_literal.size());
    const auto limits = parse_failure(turn_literal + ".");
    const auto &bounds = std::get<parse_error::turn_out_of_range>(limits.get_error_type());
    assert(bounds.minimum == 0 && bounds.maximum == vec4::T_MAX);

    const auto sequence = parse_failure("1. e4 3w. e5");
    assert(sequence.get_range() == "3w.");
    const auto &turn_error = std::get<parse_error::nonconsecutive_turn>(sequence.get_error_type());
    assert(turn_error.expected == turn(1, true) && turn_error.actual == turn(3, false));
    assert(std::string(sequence.what()) == "Expected turn 1b; found 3w.");
    check_failure<parse_error::nonconsecutive_turn>("1. e4 3. e5", 6, 8);

    const std::string duplicate = "[Board \"Standard\"]\n[BOARD \"Standard\"]";
    const auto header = parse_failure(duplicate);
    assert(header.get_range() == "BOARD");
    assert(std::get<parse_error::duplicate_header>(header.get_error_type()).key == "board");

    const std::string unicode = "{\xc3\xa9}\r\n1. e8=";
    const auto eof = parse_failure(unicode);
    assert(eof.get_start() == unicode.size() && eof.get_range().empty());
    assert(std::string(eof.what()) == "Expected a promotion piece; found end of input.");
    const auto expected = std::get<parse_error::unexpected_input>(eof.get_error_type()).expected;
    assert(expected == std::vector<std::string>{"a promotion piece"});

    for(const std::string input : {"", "0", "1", "1/", "1/2", "1/2-1", "1/2-1/", ">", "!", "?", "[", "{"}) {
        try { pgnparser(input).parse_game(); }
        catch(const parse_error &error) { assert(error.get_end() <= input.size()); }
    }
    // Parser alternatives remain usable independently.
    assert(!pgnparser("(2. Kf7)", false).parse_absolute_board());
    assert(!pgnparser("Nf").parse_standalone_move());
}

void test_metadata_diagnostics()
{
    const std::string fen = "8/8/8/8/8/8/8/8";
    const auto local = parse_failure(fen + ":0:5000:w", true);
    assert(std::holds_alternative<parse_error::turn_out_of_range>(local.get_error_type()));
    assert(local.get_range() == "5000");
    const std::string input = "{prefix}\n[" + local.get_source() + "]";
    const auto global = parse_failure(input);
    assert(global.get_start() == input.find("5000") && global.get_range() == "5000");
    const auto timeline = parse_failure(fen + ":5000:0:w", true);
    assert(std::holds_alternative<parse_error::timeline_out_of_range>(timeline.get_error_type()));
    assert(timeline.get_range() == "5000");
    for(const auto &suffix : {"", ":", ":0", ":0:", ":0:0", ":0:0:", ":0:0:z", ":0x:0:w", ":0:0:wx"})
        parse_failure(fen + suffix, true);
    const auto color = parse_failure(fen + ":0:0:", true);
    assert(color.get_range().empty());
    assert(std::get<parse_error::unexpected_input>(color.get_error_type()).expected
        == (std::vector<std::string>{"'w'", "'b'"}));
}

void test_source_ownership()
{
    const std::string input = "{prefix}\n1. e4?! {comment} (1b. c5) / e5";
    auto parsed = pgnparser(input).parse_game();
    assert(parsed && parsed->source_text && *parsed->source_text == input);
    const auto text = [&](pgnparser_ast::source_range range) {
        return parsed->source_text->substr(range.start, range.end - range.start);
    };
    const auto &branches = std::get<pgnparser_ast::gametree::variations_t>(parsed->gt.variations_or_outcome);
    const auto &first = branches.front().first;
    assert(text(*first.source) == "1. e4?!");
    assert(text(*first.moves.front().source) == "e4?!");
    const auto &continuations = std::get<pgnparser_ast::gametree::variations_t>(branches.front().second->variations_or_outcome);
    assert(text(*continuations.front().first.source) == "1b. c5");
    assert(text(*continuations.back().first.source) == "/ e5");
    assert(text(*continuations.front().first.moves.front().source) == "c5");
    const auto superphysical = pgnparser("1. (0T1)Q>>(1T1)f7+~ (~T1) (>L1)").parse_game();
    const auto &action = std::get<pgnparser_ast::gametree::variations_t>(superphysical->gt.variations_or_outcome).front().first;
    const auto range = *action.moves.front().source;
    assert(superphysical->source_text->substr(range.start, range.end - range.start)
        == "(0T1)Q>>(1T1)f7+~ (~T1) (>L1)");

    pgnparser_ast::move generated{pgnparser_ast::physical_move{}};
    pgnparser_ast::actions generated_action{{generated}, {}};
    assert(!generated.source && !generated_action.source);
    pgnparser_ast::game generated_game;
    assert(!generated_game.source_text);

    auto error = parse_failure("1. e8=");
    parse_error copy(error);
    const std::string message = copy.what();
    error = parse_failure("@");
    parse_error moved(std::move(copy));
    assert(moved.get_source() == "1. e8=" && moved.get_range().empty());
    assert(moved.what() == message);
    error = moved;
    assert(error.get_source() == moved.get_source() && error.what() == message);
    const auto source = std::make_shared<const std::string>("abcd");
    const parse_error missing(parse_error::unexpected_input{{"a number"}}, source, {2, 2});
    assert(std::string(missing.what()) == "Expected a number; found missing input.");
    for(const auto range : {pgnparser_ast::source_range{3, 2}, {0, 5}, {5, 5}}) {
        bool rejected = false;
        try { parse_error invalid(parse_error::unexpected_character{}, source, range); }
        catch(const std::invalid_argument &) { rejected = true; }
        assert(rejected);
    }
    bool rejected_null = false;
    try { parse_error invalid(parse_error::unexpected_character{}, nullptr, {0, 0}); }
    catch(const std::invalid_argument &) { rejected_null = true; }
    assert(rejected_null);
}
} /* anonymous namespace */

int main()
{
    for(const std::string notation : {"/ e4 / e5", "(/ e4) / d4"})
    {
        const auto parsed = pgnparser(notation, true, {0,false}).parse_gametree();
        assert(parsed.has_value());
        const auto &branches = std::get<pgnparser_ast::gametree::variations_t>(
            parsed->variations_or_outcome);
        assert(!branches.empty());
    }
    //test_actions();
    //test_gametree();
    test_parse_diagnostics();
    test_metadata_diagnostics();
    test_source_ownership();
    test_game();
    test_board_coordinate_limits();
    test_short_board_rejected();
    test_timeline_header();
    std::cout << "---= parse_game.cpp: all tests passed =---" <<std::endl;
    return 0;
}
