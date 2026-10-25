
#include <algorithm>
#include <cassert>
#include <limits>
#include "graph.h"

template<HCOrdering Order>
std::optional<point> HC_info::take_point(HC &hc, const Order &order) const
{
    assert(hc.dimension() == dimension());
    graph g(dimension());
    std::vector<index_t> must_include;
    constexpr index_t invalid_index = std::numeric_limits<index_t>::max();

    const size_t num_edges = static_cast<std::size_t>(dimension()) * dimension();
    std::vector<index_t> edge_refs(num_edges, invalid_index);
    const auto edge_ref = [&](index_t p, index_t q) -> index_t & {
        return edge_refs[static_cast<std::size_t>(p) * dimension() + q];
    };

    point result(dimension(), invalid_index);
    for(index_t n = 0; n < dimension(); n++)
    {
        bool has_nonjump = false;
        integer_set ghost_arrive_indices;
        order.for_each(n, hc[n], [&](index_t i) {
            const entry &loc = axis_coords[n][i];
            std::visit(overloads {
                [&](const physical_entry &) {
                    if(!has_nonjump)
                    {
                        has_nonjump = true;
                        result[n] = i;
                    }
                },
                [&](const arriving_entry &arriving) {
                    index_t from_axis = line_to_axis_unsafe(arriving.m.from.l());
                    assert(from_axis != no_axis);
                    if(!hc[from_axis].contains(arriving.idx))
                    {
                        ghost_arrive_indices.insert(i);
                        return;
                    }
                    index_t &departure_ref = edge_ref(from_axis, n);
                    if(departure_ref == invalid_index)
                    {
                        g.add_edge(from_axis, n);
                        assert(from_axis != n);
                        departure_ref = arriving.idx;
                        edge_ref(n, from_axis) = i;
                        assert(arriving.idx != invalid_index);
                    }
                },
                [](const departing_entry &) {},
                [&](const null_entry &) {
                    if(!has_nonjump)
                    {
                        has_nonjump = true;
                        result[n] = i;
                    }
                },
            }, loc);
        });

        hc[n].minus(ghost_arrive_indices);
        if(hc[n].empty())
        {
            return std::nullopt;
        }
        if(!has_nonjump)
        {
            must_include.push_back(n);
        }
    }

    auto matching = g.find_matching(must_include);
    if(!matching)
    {
        return std::nullopt;
    }
    for(const auto &[u, v] : *matching)
    {
        result[u] = edge_ref(u, v);
        result[v] = edge_ref(v, u);
        assert(result[u] != invalid_index);
        assert(result[v] != invalid_index);
    }
#ifndef NDEBUG
    for(index_t i : result)
    {
        assert(i != invalid_index && "some axis is still null");
    }
#endif
    assert(hc.contains(result));
    return result;
}

template<HCOrdering Order>
std::optional<point> HC_info::take_point(
    HC &hc, std::span<const slice> negative, const Order &order) const
{
    if(negative.empty()) return take_point(hc, order);

    // Explore the complement lazily: a temporary region at depth k already
    // avoids slices [0, k). Matching is only run after all slices are avoided.
    std::vector<std::pair<HC, std::size_t>> pending;
    pending.emplace_back(hc, 0);
    while(!pending.empty())
    {
        auto [region, next] = std::move(pending.back());
        pending.pop_back();
        if(region.empty()) continue;
        while(next < negative.size() && !region.intersects(negative[next])) ++next;
        if(next == negative.size())
        {
            if(auto pt = take_point(region, order)) return pt;
            continue;
        }
        auto pieces = region.remove_slice_carefully(negative[next]);
        for(HC &piece : pieces)
            pending.emplace_back(std::move(piece), next + 1);
    }
    return std::nullopt;
}

template<HCOrdering Order>
generator<moveseq> HC_info::deferred_search(search_space ss, Order order) const &
{
    // Negative information belongs to this search invocation, alongside its
    // positive HC cover. Temporary point-selection fragments are discarded.
    std::vector<slice> negative;
    const auto codimension = [&](const slice &problem) {
        index_t result = 0;
        for(const auto &[axis, values] : problem.get_fixed_axes())
            result += !universe[axis].is_subset_of(values);
        return result;
    };
    while(!ss.empty())
    {
        HC hc = std::move(ss.back());
        ss.pop_back();
        if(!prune_retained_slices(hc, negative)) continue;
        // Prune unsupported arrivals before temporary slicing. A matching
        // point outside all retained slices needs no temporary regions.
        auto pt = take_point(hc, order);
        if(!pt || !prune_retained_slices(hc, negative)) continue;
        if(std::any_of(negative.begin(), negative.end(),
            [&](const slice &problem) { return problem.contains(*pt); }))
        {
            pt = take_point(hc, negative, order);
        }
        if(!pt) continue;

        auto problem = find_problem(*pt, universe);
        if(!problem)
        {
            co_yield to_action(*pt);
            ss.concat(hc.remove_point(*pt));
            continue;
        }

        ss.push_back(std::move(hc));
        if(codimension(*problem) <= 1)
        {
            ss.remove_slice_globally(*problem);
            continue;
        }
        negative.push_back(std::move(*problem));
        if(negative.size() > ss.size())
        {
            auto cheapest = std::min_element(negative.begin(), negative.end(),
                [&](const slice &a, const slice &b) {
                    return codimension(a) < codimension(b);
                });
            // A popped constraint must be removed from every positive region.
            ss.remove_slice_globally(*cheapest);
            negative.erase(cheapest);
        }
    }
}

template<HCOrdering Order>
generator<moveseq> HC_info::iterative_search(search_space ss, Order order) const &
{
    while(!ss.empty())
    {
        HC hc = std::move(ss.back());
        ss.pop_back();
        auto pt_opt = take_point(hc, order);
        if(!pt_opt)
        {
            continue;
        }

        const point &pt = *pt_opt;
        auto problem = find_problem(pt, hc);
        if(problem)
        {
            ss.concat(hc.remove_slice(*problem));
        }
        else
        {
            co_yield to_action(pt);
            ss.concat(hc.remove_point(pt));
        }
    }
}

template<HCOrdering Order>
generator<moveseq> HC_info::search(search_space ss, Order order) const &
{
    constexpr std::size_t adaptive_problem_limit = 100;
    std::size_t num_problems = 0;
    while(!ss.empty() && num_problems < adaptive_problem_limit)
    {
        HC hc = std::move(ss.back());
        ss.pop_back();
        auto pt = take_point(hc, order);
        if(!pt) continue;

        auto problem = find_problem(*pt, hc);
        if(!problem)
        {
            co_yield to_action(*pt);
            ss.concat(hc.remove_point(*pt));
            continue;
        }
        ++num_problems;
        ss.push_back(std::move(hc));
        ss.remove_slice_backwards(*problem, true, 10);
    }

    for(auto &&moves : stable_search(std::move(ss), std::move(order)))
        co_yield std::move(moves);
}

template<HCOrdering Order>
generator<moveseq> HC_info::stable_search(search_space ss, Order order) const &
{
    while(!ss.empty())
    {
        HC hc = std::move(ss.back());
        ss.pop_back();
        auto pt = take_point(hc, order);
        if(!pt) continue;

        auto problem = find_problem(*pt, universe);
        if(problem)
        {
            ss.push_back(std::move(hc));
            ss.remove_slice_globally(*problem);
        }
        else
        {
            co_yield to_action(*pt);
            ss.concat(hc.remove_point(*pt));
        }
    }
}
