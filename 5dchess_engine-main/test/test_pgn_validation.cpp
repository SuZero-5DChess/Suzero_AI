#undef NDEBUG
#include "game.h"
#include "pgnparser.h"
#include "pgnerror.h"
#include "state.h"
#include "pgn_loading.h"
#include <cassert>
#include <set>
#include <type_traits>
#include <utility>

namespace
{
using error = pgn_validation_error;
const std::string standard = "[Board \"Standard - Turn Zero\"]\n";
const std::string knights = "[Board \"Custom\"]\n[7k/8/8/8/8/8/3N3N/K7:0:1:w]\n";

template<typename Load>
error failure(Load load)
{
    try { load(); }
    catch(const error &e) { return e; }
    throw std::logic_error("Expected pgn_validation_error");
}

void same_failure(const std::string &pgn)
{
    const auto from_game = failure([&] { game::from_pgn(pgn); });
    const auto from_state = failure([&] { state::from_pgn(pgn); });
    assert(from_game.get_error_type().index() == from_state.get_error_type().index());
    assert(std::string(from_game.what()) == from_state.what());
    assert(from_game.get_text() == from_state.get_text());
    assert(bool(from_game.get_location()) == bool(from_state.get_location()));
    if(const auto *location = from_game.get_location()) {
        assert(location->range.start == from_state.get_location()->range.start);
        assert(location->range.end == from_state.get_location()->range.end);
    }
}

void test_moves()
{
    static_assert(std::is_base_of_v<std::invalid_argument, error>);
    const std::string pgn = standard + "{prefix}\n(1. (0T3)Wb2b2?!) 1. e4";
    const auto invalid = failure([&] { game::from_pgn(pgn); });
    assert(std::holds_alternative<error::invalid_move>(invalid.get_error_type()));
    assert(std::string(invalid.what()) == "Invalid move: (0T3)Wb2b2?!");
    assert(invalid.get_location() && *invalid.get_location()->source == pgn);
    assert(invalid.get_location()->range.start == pgn.find("(0T3)"));
    same_failure(standard + "1. (0T3)Wb2b2?!");

    // Initial-position loading must not interpret the move history.
    const auto [initial, setup_error] = pgn_loading::load_initial_position<false>(*pgnparser(pgn).parse_game());
    assert(!setup_error);
    assert(initial.get_present() == (turn{1, false}));

    const auto ambiguous = failure([&] { state::from_pgn(knights + "1. Nf3"); });
    const auto &candidates = std::get<error::ambiguous_move>(ambiguous.get_error_type()).candidates;
    assert(candidates.size() == 2 && candidates[0].move != candidates[1].move);
    assert(std::string(ambiguous.what()) == "Ambiguous move: Nf3; possible candidates: "
        + candidates[0].notation + ", " + candidates[1].notation + " (totally 2)");
    for(const auto &candidate : candidates) {
        auto position = state::from_pgn(knights);
        assert(position.apply_move(candidate.move));
        state::from_pgn(knights + "1. " + candidate.notation);
    }
    same_failure(knights + "1. Nf3");

    const std::string three = "[Board \"Custom\"]\n[7k/8/8/8/3N4/8/3N3N/K7:0:1:w]\n1. Nf3";
    const auto multiple = failure([&] { game::from_pgn(three); });
    const auto &all = std::get<error::ambiguous_move>(multiple.get_error_type()).candidates;
    assert(all.size() == 3);
    assert(std::string(multiple.what()) == "Ambiguous move: Nf3; possible candidates: "
        + all[0].notation + ", " + all[1].notation + ", ... (totally 3)");
    same_failure(three);
}

void test_promotions()
{
    const std::string setup = "[Board \"Custom\"]\n[Promotions \"N\"]\n"
        "[7k/P7/8/8/8/8/8/K7:0:1:w]\n";
    const auto prohibited = failure([&] { state::from_pgn(setup + "1. a8=Q"); });
    const auto &reason = std::get<error::illegal_promotion>(prohibited.get_error_type());
    assert(reason.requested == QUEEN_W && reason.allowed == promotion_options::KNIGHT);
    assert(std::string(prohibited.what()) == "Illegal promotion: a8=Q");
    same_failure(setup + "1. a8=Q");
    const auto promoted = state::from_pgn(setup + "1. a8");
    const auto explicit_promotion = state::from_pgn(setup + "1. a8=N");
    assert(promoted.show_fen() == explicit_promotion.show_fen());
    auto parsed = pgnparser(setup).parse_game();
    state position = state::from_ast<true>(*parsed);
    auto move = pgnparser("a8").parse_standalone_move();
    assert(position.apply_move_ast(*move).promote_to == KNIGHT_W);

    const std::string capture = "[Board \"Custom\"]\n[2r4k/1P1P4/8/8/8/8/8/K7:0:1:w]\n1. xc8=N";
    const auto ambiguous = failure([&] { state::from_pgn(capture); });
    for(const auto &candidate : std::get<error::ambiguous_move>(ambiguous.get_error_type()).candidates) {
        assert(candidate.move.promote_to == KNIGHT_W);
        assert(candidate.notation.ends_with("=N"));
    }
    same_failure(capture);
}

void test_actions_and_setups()
{
    const std::string incomplete = "[Size \"8x8\"] [7k/8/8/8/8/8/8/K7:0:1:w] "
        "[7k/8/8/8/8/8/8/K7:1:1:w] 1. (0T1)a1b1";
    const auto present = failure([&] { state::from_pgn(incomplete); });
    assert(std::holds_alternative<error::present_not_advanced>(present.get_error_type()));
    assert(present.get_text() == "1. (0T1)a1b1");
    assert(std::string(present.what()) == "Cannot submit action: 1. (0T1)a1b1; the moves do not advance the present.");
    same_failure(incomplete);

    const std::string mate = standard + "1. f3 / e5 2. g4 / Qh4";
    state::from_pgn(mate);
    const auto exposed = failure([&] { state::from_pgn(mate + " 3. a3"); });
    const auto &royal = std::get<error::royal_exposed>(exposed.get_error_type());
    assert(!royal.is_from_initial_setup);
    assert(royal.notation == "(0T3)Qh4xe1");
    assert(std::string(exposed.what()) == "Illegal action: 3. a3; royal piece exposed to capture by (0T3)Qh4xe1.");
    same_failure(mate + " 3. a3");
    assert(game::from_pgn(mate + " 3. a3", {.allow_illegal_history = true}).has_illegal_history());
    state::from_pgn<true>(mate + " 3. a3");

    const std::string setup = "[Board \"Custom\"]\n[7k/8/8/8/8/8/8/K7:0:1:w]\n[7k/3r4/8/8/3K4/8/8/8:1:1:b]";
    const std::string exposed_setup = "[Size \"5x5\"]\n[Timeline \"odd\"]\n[k4/5/5/5/R4:0:1:w]";
    failure([&] { state::from_pgn(exposed_setup); });
    same_failure(exposed_setup);
    // The returned diagnostic owns its source after the parsed game is destroyed.
    const auto [exposed_initial, setup_error] = pgn_loading::load_initial_position<false>(*pgnparser(exposed_setup).parse_game());
    assert(exposed_initial.get_board_size() == (std::pair{5, 5}));
    assert(setup_error && setup_error->get_location());
    assert(setup_error->get_text() == "[k4/5/5/5/R4:0:1:w]");
    const auto [unchecked_initial, unchecked_error] =
        pgn_loading::load_initial_position<true>(*pgnparser(exposed_setup).parse_game());
    assert(!unchecked_error);
    assert(unchecked_initial.show_fen() == exposed_initial.show_fen());
    auto generated = pgnparser(exposed_setup).parse_game();
    generated->source_text.reset();
    const auto [generated_initial, generated_error] = pgn_loading::load_initial_position<false>(*generated);
    assert(generated_error && !generated_error->get_location());
    assert(generated_error->get_text() == "initial setup");

    assert(game::from_pgn(exposed_setup, {.allow_illegal_history = true}).has_illegal_history());

    const std::string checkmate = "[Size \"5x5\"]\n[Timeline \"odd\"]\n[k4/5/5/5/RR3:0:0:b]";
    state::from_pgn(checkmate);
    const auto won = game::from_pgn(checkmate);
    assert(!won.has_illegal_history());
    assert(won.get_match_status() == game::match_status_t::WHITE_WINS);

    const auto initial = failure([&] { state::from_pgn(setup); });
    const auto &initial_reason = std::get<error::royal_exposed>(initial.get_error_type());
    assert(initial_reason.is_from_initial_setup);
    assert(initial_reason.notation == "(1T1)Rd7xd4");
    assert(initial.get_location());
    assert(initial.get_text() == "[7k/3r4/8/8/3K4/8/8/8:1:1:b]");
    assert(std::string(initial.what()) == "Illegal initial setup: royal piece exposed to capture by (1T1)Rd7xd4.");
    same_failure(setup);
}

void test_fen_source_mapping()
{
    const std::string pgn = "[Event \"caf\xc3\xa9\"]\n[Board \"Custom\"]\n[7k/8/8/8/8/8/8/K6:0:1:w]";
    try { game::from_pgn(pgn); }
    catch(const board_fen_error &e) {
        assert(e.get_pgn_source() && *e.get_pgn_source() == pgn);
        assert(e.get_pgn_range());
        const auto range = *e.get_pgn_range();
        assert(pgn.substr(range.start, range.end - range.start) == "K6");
        assert(e.get_range() == "K6");
        return;
    }
    throw std::logic_error("Expected located FEN error");
}

void test_source_free_and_lifetime()
{
    const auto create = [] {
        auto parsed = pgnparser(standard + "1. e2e5?!").parse_game();
        return failure([&] { state::from_ast(*parsed); });
    };
    auto original = create();
    auto copied = original;
    auto moved = std::move(original);
    assert(copied.get_text() == "e2e5?!" && moved.get_text() == copied.get_text());
    copied = moved;
    assert(copied.get_location() && copied.get_location()->source->find("e2e5?!") != std::string::npos);

    auto generated = pgnparser(standard + "1. e2e5").parse_game();
    generated->source_text.reset();
    auto &act = std::get<pgnparser_ast::gametree::variations_t>(generated->gt.variations_or_outcome).front().first;
    act.source.reset();
    act.moves.front().source.reset();
    const auto fallback = failure([&] { state::from_ast(*generated); });
    assert(!fallback.get_location());
    assert(fallback.get_text().starts_with("mv:pm{"));
    assert(std::holds_alternative<error::invalid_move>(fallback.get_error_type()));

    // Source-free generated moves can also be applied successfully.
    auto move = pgnparser("e4").parse_standalone_move();
    move->source.reset();
    auto position = state::from_pgn(standard);
    const auto applied = position.apply_move_ast(*move);
    assert(applied.promote_to == NO_PIECE);
}

void test_header_validation()
{
    const auto check = []<typename Reason>(const std::string &header, const std::string &value,
        Reason) {
        const std::string pgn = "[Event \"😀 café\"]\n[" + header + " \"" + value + "\"]";
        same_failure(pgn);
        const auto e = failure([&] { game::from_pgn(pgn, {.allow_illegal_history = true}); });
        assert(std::holds_alternative<Reason>(e.get_error_type()));
        assert(e.get_text() == value);
        const auto start = pgn.rfind('"', pgn.size() - 3) + 1;
        assert(e.get_location());
        assert(e.get_location()->range.start == start);
        assert(e.get_location()->range.end == start + value.size());
        assert(std::string(e.what()).find("derive_variant_setup") == std::string::npos);
        auto parsed = pgnparser(pgn).parse_game();
        assert(parsed->header_sources.contains("event"));
        parsed->source_text.reset();
        parsed->header_sources.clear();
        const auto fallback = failure([&] { state::from_ast(*parsed); });
        assert(!fallback.get_location());
        assert(fallback.get_text() == value);
        assert(std::string(fallback.what()) == e.what());
        return e;
    };
    check("BoArD", "Unknown", error::unknown_variant{});
    check("VaRiAnT", "", error::unknown_variant{});
    const auto format = check("Size", "8", error::invalid_size{});
    assert(std::get<error::invalid_size>(format.get_error_type()).cause == error::invalid_size::reason::format);
    const auto number = check("Size", "ax8", error::invalid_size{});
    assert(std::get<error::invalid_size>(number.get_error_type()).cause == error::invalid_size::reason::number);
    const auto limit = check("Size", "99999999999999999999x8", error::invalid_size{});
    assert(std::get<error::invalid_size>(limit.get_error_type()).cause == error::invalid_size::reason::out_of_range);
    check("Size", "0x8", error::invalid_size{});
    check("Timeline", "Sideways", error::invalid_timeline{});
    const auto piece = check("Promotions", "K", error::invalid_promotions{});
    assert(std::get<error::invalid_promotions>(piece.get_error_type()).cause == invalid_promotions::reason::invalid_piece);
    const auto wildcard = check("Promotions", "*Q", error::invalid_promotions{});
    assert(std::get<error::invalid_promotions>(wildcard.get_error_type()).cause == invalid_promotions::reason::mixed_wildcard);

    const auto missing = failure([] { state::from_pgn("[Event \"No setup\"]"); });
    assert(std::holds_alternative<error::missing_setup>(missing.get_error_type()));
    assert(!missing.get_location());
    same_failure("[Board \"Custom\"]");
    same_failure("[Board \"Unknown board\"][Variant \"Unknown variant\"]");
    const auto precedence = failure([] {
        state::from_pgn("[Board \"Unknown board\"][Variant \"Unknown variant\"]");
    });
    assert(precedence.get_text() == "Unknown board");
    state::from_pgn(standard + "[Variant \"Ignored\"][Promotions \"\"][Timeline \"oDd\"]");
    state::from_pgn(knights + "[Variant \"Ignored with explicit boards\"]");
    state::from_pgn(knights + "[Size \" +8trailingx8trailing\"]");
}

void test_error_invariants()
{
    const auto text = std::make_shared<const std::string>("move");
    for(const auto range : {pgnparser_ast::source_range{3, 2}, {0, 5}, {5, 5}}) {
        bool rejected = false;
        try { error e(error::invalid_move{}, error::source_location{text, range}); }
        catch(const std::invalid_argument &) { rejected = true; }
        assert(rejected);
    }
    bool null_rejected = false;
    try { error e(error::invalid_move{}, error::source_location{nullptr, {0, 0}}); }
    catch(const std::invalid_argument &) { null_rejected = true; }
    assert(null_rejected);
    const ext_move move(vec4(0, 0, 1, 0), vec4(0, 1, 1, 0), NO_PIECE);
    for(std::size_t count = 0; count <= 2; ++count) {
        bool rejected = false;
        try { error e(error::ambiguous_move{std::vector<error::candidate>(count, {move, "a1a2"})},
            error::description{"a2"}); }
        catch(const std::invalid_argument &) { rejected = true; }
        assert(rejected); // Zero/one candidate, or two identical candidates.
    }
    const error illegal(error::illegal_move{move}, error::description{"a1a2"});
    assert(std::get<error::illegal_move>(illegal.get_error_type()).move == move);
    assert(std::string(illegal.what()) == "Illegal move: a1a2");
}
} /* anonymous namespace */

int main()
{
    test_moves();
    test_promotions();
    test_actions_and_setups();
    test_fen_source_mapping();
    test_source_free_and_lifetime();
    test_error_invariants();
    test_header_validation();
}
