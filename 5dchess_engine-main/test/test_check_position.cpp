#undef NDEBUG
#include <cassert>
#include <random>
#include <set>
#include "check_position.h"
#include <algorithm>
#include <iostream>

namespace check_position_tests
{
namespace {
void compare(const check_position &view, const state &expected)
{
    for (bool c : {false,true}) {
        std::set<full_move> reference;
        for (full_move mv : expected.find_checks(c)) reference.insert(mv);
        const auto moves = view.checks(c);
        const std::set<full_move> actual(moves.begin(),moves.end());
        if (actual != reference) {
            std::cerr << "color " << c << '\n' << expected.to_string();
            for (auto mv : reference) if (!actual.contains(mv)) std::cerr << "missing " << mv << '\n';
            for (auto mv : actual) if (!reference.contains(mv)) std::cerr << "extra " << mv << '\n';
            std::abort();
        }
        const auto first = view.first_check(c);
        assert(first.has_value() == !reference.empty());
        if (first) assert(reference.contains(*first));
        std::set<full_move> sp;
        for (auto mv : reference) if (mv.from.tl() != mv.to.tl()) sp.insert(mv);
        const auto sp_moves = view.checks(c,false);
        assert(std::set<full_move>(sp_moves.begin(),sp_moves.end()) == sp);
        const auto first_sp = view.first_check(c,false);
        assert(first_sp.has_value() == !sp.empty());
        if (first_sp) assert(sp.contains(*first_sp));
    }
}

void compare_phantom(const state &s)
{
    const auto before = s.get_boards();
    const state expected = s.phantom();
    const auto view = check_position::for_phantom(s);
    compare(view,expected);
    const auto [lo, hi] = s.get_lines_range();
    for (int l = lo; l <= hi; ++l) {
        const auto start = s.get_timeline_start(l);
        const auto [t, c] = s.get_timeline_end(l).get();
        const auto end = expected.get_timeline_end(l);
        for (turn at = start; at <= end; at = at.next_turn()) {
            const board *actual = view.get_board_ptr(l,at.t(),at.c());
            assert(actual);
            assert(actual->get_fen<true>() ==
                   expected.get_board_ptr(l,at.t(),at.c())->get_fen<true>());
            const board *borrowed = at <= turn{t,c}
                ? s.get_board_ptr(l,at.t(),at.c()) : s.get_board_ptr(l,t,c);
            assert(actual == borrowed);
        }
        const auto after_end = end.next_turn();
        assert(!view.get_board_ptr(l,after_end.t(),after_end.c()));
    }
    const bool checked = expected.find_checks(!s.get_present().c()).first().has_value();
    assert(s.has_phantom_check() == checked);
    assert(s.get_boards() == before);
}

std::shared_ptr<board> random_board(std::mt19937 &rng, int width, int height)
{
    std::string empty;
    for(int y = 0; y < height; ++y) {
        if(y) empty += '/';
        empty += std::to_string(width);
    }
    auto b = std::make_shared<board>(empty,width,height);
    constexpr piece_t pieces[] = {KING_W, ROYAL_QUEEN_W, COMMON_KING_W,
        QUEEN_W, PRINCESS_W, ROOK_W, BISHOP_W, UNICORN_W, DRAGON_W,
        KNIGHT_W, PAWN_W, BRAWN_W, PAWN_UW, BRAWN_UW, KING_UW, ROOK_UW};
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        piece_t p = NO_PIECE;
        if (rng()%5 == 0) {
            p = pieces[rng()%std::size(pieces)];
            if (rng()%2) p = to_black(p);
        }
        b->set_piece(ppos(x,y),p);
    }
    return b;
}

void randomized_differential()
{
    std::mt19937 rng(55332);
    for (int trial = 0; trial < 120; ++trial) {
        const int width = trial%2 ? 8 : 5, height = trial%3 ? 8 : 6;
        std::vector<boards_info_t> boards;
        for (int l = -2; l <= 2; ++l) {
            const int begin = rng()%5, end = begin+2+rng()%12;
            for (int ply = begin; ply <= end; ++ply)
                boards.emplace_back(l,ply/2,ply%2,random_board(rng,width,height)->get_fen<true>(width,height));
        }
        multiverse original(timeline_layout::odd, boards,width,height);
        state s(original);
        check_position view(s,-3,3);
        compare(view,s);
        compare_phantom(s);

        // Simultaneously add boards on several existing lines, and branches
        // on both sides. The independently materialized position is the oracle.
        auto materialized = original;
        std::vector<std::shared_ptr<board>> additions;
        for (int l = -3; l <= 3; ++l) {
            if (l != -3 && l != 3 && rng()%2) continue;
            auto b = random_board(rng,width,height);
            const turn at = (l == -3 || l == 3)
                ? turn{static_cast<int>(rng()%8),static_cast<bool>(rng()%2)}
                : s.get_timeline_end(l).next_turn();
            view.add_board(l,at,*b);
            materialized.insert_board(l,at.t(),at.c(),b);
            additions.push_back(std::move(b));
        }
        state expected(materialized);
        compare(view,expected);
        compare_phantom(expected);
    }
}

void long_rays_and_blockers()
{
    const std::string empty = "8/8/8/8/8/8/8/8";
    for (bool blocked : {false,true}) {
        std::vector<boards_info_t> boards;
        for (int ply = 0; ply <= 20; ++ply) {
            board b(empty);
            if (ply == 0) b.set_piece(ppos(3,3),KING_B);
            if (ply == 20) b.set_piece(ppos(3,3),ROOK_W);
            if (blocked && ply == 10) b.set_piece(ppos(3,3),PAWN_B);
            boards.emplace_back(0,ply/2,ply%2,b.get_fen<true>());
        }
        multiverse m(timeline_layout::odd, boards);
        state s(m);
        check_position view(s);
        compare(view,s);
        assert(view.first_check(false,false).has_value() == !blocked);
    }

    // A missing intermediate board stops a slider, even if a farther board
    // exists. Adding that intermediate board must immediately open the ray.
    board attacker(empty), target(empty), bridge(empty);
    attacker.set_piece(ppos(3,3),ROOK_W);
    target.set_piece(ppos(3,3),KING_B);
    multiverse m(timeline_layout::odd, {{0,3,false,attacker.get_fen<true>()},
                      {1,2,true,empty}, {2,3,false,target.get_fen<true>()}});
    state s(m);
    check_position view(s);
    compare(view,s);
    assert(!view.first_check(false,false));
    view.add_board(1,{3,false},bridge);
    auto expected = m;
    expected.append_board(1,std::make_shared<board>(bridge));
    compare(view,state(expected));
    assert(view.first_check(false,false));
}

void new_boards_check_each_other()
{
    for (bool mover : {false,true}) {
        multiverse original(timeline_layout::odd, {{0,2,mover,"8/8/8/8/8/8/8/8"}});
        state s(original);
        check_position view(s,-1,1);
        auto attacker = std::make_shared<board>("8/8/8/8/8/8/8/8");
        auto target = std::make_shared<board>(*attacker);
        attacker->set_piece(ppos(3,3),mover ? ROOK_W : ROOK_B);
        target->set_piece(ppos(3,3),mover ? KING_B : KING_W);
        const auto at = turn{2,mover}.next_turn();
        view.add_board(0,at,*attacker);
        view.add_board(1,at,*target);
        auto expected = original;
        expected.append_board(0,attacker);
        expected.insert_board(1,at.t(),at.c(),target);
        compare(view,state(expected));
        assert(view.first_check(!mover,false));
        // A branch has no inherited history and uncreated reserved lines are absent.
        assert(!view.get_board_ptr(1,2,mover));
        assert(!view.get_board_ptr(-1,at.t(),at.c()));
    }
}
}

int run()
{
    new_boards_check_each_other();
    randomized_differential();
    long_rays_and_blockers();
    return 0;
}
} /* namespace check_position_tests */

namespace phantom_checks_tests
{
namespace {
constexpr const char *empty = "8/8/8/8/8/8/8/8";

state cross_line_position(bool mover)
{
    board source(empty), target(empty);
    source.set_piece(ppos(3,3),mover ? ROOK_W : ROOK_B);
    target.set_piece(ppos(3,3),mover ? KING_B : KING_W);
    multiverse m(timeline_layout::odd, {{0,1,mover,source.get_fen<true>()},
                      {1,1,mover,target.get_fen<true>()}});
    return state(m);
}

state physical_position(bool mover)
{
    board b(empty);
    b.set_piece(ppos(3,3),mover ? KING_B : KING_W);
    b.set_piece(ppos(3,6),mover ? ROOK_W : ROOK_B);
    multiverse m(timeline_layout::odd, {{0,1,mover,b.get_fen<true>()}});
    return state(m);
}

void compare_phantom(const state &s)
{
    const bool player = s.get_present().c();
    const state expected = s.phantom();
    const auto view = check_position::for_phantom(s);
    std::set<full_move> reference, actual;
    for (auto mv : expected.find_checks(!player)) reference.insert(mv);
    for (auto mv : view.checks(!player)) actual.insert(mv);
    assert(reference == actual);
    assert(s.has_phantom_check() == !reference.empty());
    const auto [lo, hi] = s.get_lines_range();
    for (int l = lo; l <= hi; ++l) {
        const auto end = expected.get_timeline_end(l);
        for (auto at = s.get_timeline_start(l); at <= end; at = at.next_turn()) {
            const auto *b = view.get_board_ptr(l,at.t(),at.c());
            assert(b);
            assert(b->get_fen<true>() == expected.get_board_ptr(l,at.t(),at.c())->get_fen<true>());
        }
        const auto after = end.next_turn();
        assert(!view.get_board_ptr(l,after.t(),after.c()));
    }
}

void test_phantom_cases()
{
    for (bool mover : {false,true}) {
        const state physical = physical_position(mover);
        compare_phantom(physical);
        assert(physical.has_phantom_check());
        const state cross = cross_line_position(mover);
        compare_phantom(cross);
        const auto view = check_position::for_phantom(cross);
        const auto hit = view.first_check(!mover,false);
        assert(hit && hit->from.l() != hit->to.l());
        const auto at = turn{1,mover}.next_turn();
        assert(hit->from.t() == at.t() && hit->to.t() == at.t());

        // During an action the stored player must be used, even after the
        // apparent present has changed to the opponent before submit().
        const std::string fen = mover ? "8/8/8/8/8/8/8/r7" : "8/8/8/8/8/8/8/R7";
        multiverse m(timeline_layout::odd, {{0,1,mover,fen}, {1,1,mover,fen}});
        state partial(m);
        compare_phantom(partial);
        assert(partial.apply_move<true>(full_move(vec4(0,0,1,0),vec4(0,1,1,0))));
        compare_phantom(partial);
        assert(partial.apply_move<true>(full_move(vec4(0,0,1,1),vec4(0,1,1,1))));
        assert(partial.get_present().c() != partial.apparent_present().c());
        compare_phantom(partial);
        const auto before_submit = check_position::for_phantom(partial);
        // Neither endpoint matches the stored player: append nothing.
        for (int l : {0,1}) {
            const auto end = partial.get_timeline_end(l);
            const auto after = end.next_turn();
            assert(!before_submit.get_board_ptr(l,after.t(),after.c()));
        }
        assert(partial.submit());
        compare_phantom(partial);

        multiverse even(timeline_layout::even, {{-1,1,mover,fen}, {0,1,mover,fen}});
        compare_phantom(state(even));
    }
}

void test_present_boundaries()
{
    for (bool mover : {false,true}) {
        multiverse m(timeline_layout::odd, {
            {0,2,mover,empty}, {1,3,mover,empty}, {2,2,mover,empty}
        });
        const state s(m);
        assert((s.get_active_range() == std::pair{0,1}));
        compare_phantom(s);
        const auto view = check_position::for_phantom(s);
        for (int l : {0,1,2}) {
            const auto after = s.get_timeline_end(l).next_turn();
            assert(static_cast<bool>(view.get_board_ptr(l,after.t(),after.c()))
                   == (l == 0));
        }
    }
}

void compare_check_symbols(const state &s)
{
    const auto before = s.get_boards();
    const auto stored = s.get_present();
    auto opponent_present = s.apparent_present();
    if (opponent_present.c() == stored.c())
        opponent_present = opponent_present.next_turn();
    assert(opponent_present.c() != stored.c());
    multiverse expected(timeline_layout::odd, before);
    const auto [lo, hi] = s.get_active_range();
    for (int l = lo; l <= hi; ++l) {
        if (s.get_timeline_end(l) == opponent_present)
            expected.append_board(l, s.get_board(l,
                opponent_present.t(), opponent_present.c()));
    }
    const auto view = check_position::for_check_symbols(s);
    check_position_tests::compare(view,state(expected));
    for (int l = s.get_lines_range().first; l <= s.get_lines_range().second; ++l) {
        const auto end = expected.get_timeline_end(l);
        for (auto at = s.get_timeline_start(l); at <= end; at = at.next_turn()) {
            const auto *b = view.get_board_ptr(l,at.t(),at.c());
            assert(b && b->get_fen<true>() == expected.get_board_ptr(l,at.t(),at.c())->get_fen<true>());
        }
        const auto after = end.next_turn();
        assert(!view.get_board_ptr(l,after.t(),after.c()));
    }
    assert(s.get_boards() == before && s.get_present() == stored);
}

void test_check_symbol_contexts()
{
    for (bool mover : {false,true}) {
        const std::string fen = mover ? "8/8/8/8/8/8/8/r7" : "8/8/8/8/8/8/8/R7";
        multiverse m(timeline_layout::odd, {
            {0,2,mover,fen}, {1,3,mover,fen}, {2,2,mover,fen}
        });
        state partial(m);
        // Move inactive and future boards before the mandatory present board.
        for (int l : {2,1,0}) {
            const int t = partial.get_timeline_end(l).t();
            assert(partial.apply_move<true>(full_move(vec4(0,0,t,l),vec4(0,1,t,l))));
            compare_check_symbols(partial);
        }
        assert(partial.submit());
        compare_phantom(partial);

        // A historical branch moves the apparent present behind the stored one.
        std::vector<boards_info_t> history;
        for (auto at = turn{1,mover}; at <= turn{3,mover}; at = at.next_turn())
            history.emplace_back(0,at.t(),at.c(),fen);
        state branching(multiverse(timeline_layout::odd, history));
        assert(branching.apply_move<true>(full_move(vec4(0,0,3,0),vec4(0,1,1,0))));
        assert(branching.apparent_present() == turn(1,mover).next_turn());
        compare_check_symbols(branching);
    }
}

}

int run()
{
    test_phantom_cases();
    test_present_boundaries();
    test_check_symbol_contexts();
    return 0;
}
} /* namespace phantom_checks_tests */

void test_royal_exposure()
{
    // Exposure checks select the attacker by validation context and include
    // inactive endpoints without manufacturing ordinary checks.
    for (bool defender : {false,true}) {
        const std::string checked = defender
            ? "7K/3R4/8/8/3k4/8/8/8"
            : "7k/3r4/8/8/3K4/8/8/8";
        state ordinary(multiverse(timeline_layout::odd, {{0,1,defender,checked}}));
        assert(ordinary.has_phantom_check());
        assert(!ordinary.find_royal_exposure(true));
        assert(!ordinary.find_royal_exposure(false));

        state exposed(multiverse(timeline_layout::odd, {{0,1,!defender,checked}}));
        const auto initial_exposure = exposed.find_royal_exposure(true);
        const auto action_exposure = exposed.find_royal_exposure(false);
        assert(initial_exposure && initial_exposure->is_from_initial_setup);
        assert(action_exposure && !action_exposure->is_from_initial_setup);
        assert(initial_exposure->capture == action_exposure->capture);

        const std::string safe = "7k/8/8/8/8/8/8/K7";
        const int opponent_t = defender ? 2 : 1;
        for (int target : {1,2}) {
            std::vector<boards_info_t> boards{{0,1,defender,safe}};
            if (target == 2) boards.emplace_back(1,opponent_t,!defender,"8/8/8/8/8/8/8/8");
            boards.emplace_back(target,opponent_t,!defender,checked);
            const state setup(multiverse(timeline_layout::odd,boards));
            assert(setup.get_present().c() == defender);
            // A setup must also inspect sources of the off-turn color.
            assert(setup.find_royal_exposure(true));
            assert(!setup.find_royal_exposure(false));
        }

        // An unresolved check is detected both before and after submission.
        assert(ordinary.apply_move<true>(full_move(vec4(3,3,1,0),vec4(3,4,1,0))));
        assert(ordinary.find_royal_exposure(true));
        assert(ordinary.test_present());
        assert(!ordinary.submit());
        assert(ordinary.submit<true>());
        assert(ordinary.find_royal_exposure(false));

        // Action validation also detects cross-board captures.
        const std::string rook = defender
            ? "8/8/8/8/3R4/8/8/8" : "8/8/8/8/3r4/8/8/8";
        const std::string royal = defender
            ? "8/8/8/8/3k4/8/8/8" : "8/8/8/8/3K4/8/8/8";
        const state cross(multiverse(timeline_layout::odd, {
            {0,1,!defender,rook}, {1,1,!defender,royal}
        }));
        assert(cross.find_royal_exposure(true));
        assert(cross.find_royal_exposure(false));
    }
}

int main()
{
    test_royal_exposure();
    std::cerr << "Running check_position tests\n";
    if(const int result = check_position_tests::run(); result != 0)
        return result;
    std::cerr << "Running phantom_checks tests\n";
    if(const int result = phantom_checks_tests::run(); result != 0)
        return result;
    return 0;
}
