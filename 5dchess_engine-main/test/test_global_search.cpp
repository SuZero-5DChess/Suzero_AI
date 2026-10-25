#undef NDEBUG
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>

#include "finetree.h"
#include "rollout_policy.h"

namespace
{
state slow_position()
{
    const auto path = std::filesystem::path(__FILE__).parent_path()
        / "pgn/slow/flat-ucb-match-123.5dpgn";
    std::ifstream input(path);
    assert(input);
    std::ostringstream text;
    text << input.rdbuf();
    return state::from_pgn(text.str());
}

template<class Order>
std::set<moveseq> fine_actions(const state &position, const Order &order)
{
    auto root = fine_node<>::make_root(position);
    std::set<moveseq> actual;
    const auto visit = [&](auto &&self, fine_node<> *node) -> void {
        if(node->is_ceiling())
        {
            auto moves = node->to_action();
            assert(position.can_apply(action::from_moveseq(moves, position)));
            assert(actual.insert(std::move(moves)).second);
            return;
        }
        const auto existing = node->get_children().size();
        for(std::size_t i = 0; i < existing; ++i)
            self(self, node->get_children()[i]);
        for([[maybe_unused]] index_t index : node->search(order))
            self(self, node->get_children().back());
    };
    visit(visit, root.get());
    return actual;
}

void compare_action_sets()
{
    // Repeated Present conflicts, shared prefixes, and reactivated timelines
    // exercise global cuts propagated from temporary nodes to their ancestors.
    for(int reactivated_turn : {3, 5})
    {
        const auto position = state::from_pgn(
            "[Board \"custom\"]\n[Size \"4x4\"]\n"
            "[3k/4/4/KN2:-2:" + std::to_string(reactivated_turn) + ":b]\n"
            "[3k/4/4/KN2:-1:5:w]\n"
            "[3k/4/4/KN2:-1:5:b]\n"
            "[3k/4/4/KN2:-1:6:w]\n"
            "[3k/4/4/KN2:0:5:w]\n");
        auto [info, space] = HC_info::build_HC(position);
        std::set<moveseq> expected;
        for(const auto &moves : info.iterative_search(space)) expected.insert(moves);
        assert(expected.size() == (reactivated_turn == 3 ? 150 : 146));
        assert(fine_actions(position, natural_HC_ordering{}) == expected);
        for(unsigned seed : {7, 42})
        {
            std::mt19937 rng(seed);
            assert(fine_actions(position, random_HC_ordering(info.get_universe(), rng)) == expected);
            random_action_selection random;
            const auto move = random(position, {}, &rng);
            assert(move && expected.contains(*move));
            weighted_action_selection weighted(move_evaluation_weights{});
            const auto weighted_move = weighted(position, {}, &rng);
            assert(weighted_move && expected.contains(*weighted_move));
        }
    }
}

void slow_searches()
{
    const auto position = slow_position();
    assert(!position.get_present().c());
    auto start = std::chrono::steady_clock::now();
    const auto zero = evaluate_zero_position(position, {});
    assert(zero && zero->score == -WINNING_SCORE);
    assert(zero->data.end == rollout_details::termination::BLACK_WINS);
    std::cerr << "zero evaluator: " << std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count() << " ms\n";

    {
        start = std::chrono::steady_clock::now();
        auto root = fine_node<>::make_root(position);
        assert(!root->search(natural_HC_ordering{}).first());
        assert(root->is_terminal());
        std::cerr << "natural fine search: " << std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count() << " ms\n";
    }
    for(unsigned seed : {7, 42, 123})
    {
        std::mt19937 rng(seed);
        start = std::chrono::steady_clock::now();
        assert(!random_action_selection{}(position, {}, &rng));
        const auto after_random = std::chrono::steady_clock::now();
        weighted_action_selection weighted;
        assert(!weighted(position, {}, &rng));
        const auto after_weighted = std::chrono::steady_clock::now();
        auto root = fine_node<>::make_root(position);
        random_HC_ordering order(root->get_hc_info().get_universe(), rng);
        assert(!root->search(order).first());
        assert(root->is_terminal());
        std::cerr << "seed " << seed << " (ms): random="
                  << std::chrono::duration<double, std::milli>(after_random - start).count()
                  << " weighted="
                  << std::chrono::duration<double, std::milli>(after_weighted - after_random).count()
                  << " fine=" << std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - after_weighted).count() << '\n';
    }
    struct observer {} observed;
    rollout_default_policy rollout({}, {}, 1, 42);
    const auto result = rollout.evaluate(position, {}, observed);
    assert(result && result->score == -WINNING_SCORE && result->data.num_actions == 0);

    std::stop_source cancelled;
    cancelled.request_stop();
    std::mt19937 rng(42);
    assert(!evaluate_zero_position(position, cancelled.get_token()));
    assert(!random_action_selection{}(position, cancelled.get_token(), &rng));
    weighted_action_selection weighted(move_evaluation_weights{});
    assert(!weighted(position, cancelled.get_token(), &rng));
    assert(!rollout.evaluate(position, cancelled.get_token(), observed));
}
} // namespace

int main()
{
    compare_action_sets();
    slow_searches();
}
