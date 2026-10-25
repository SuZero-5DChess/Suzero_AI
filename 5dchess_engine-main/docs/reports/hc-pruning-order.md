# Order-preserving HC pruning

Measured on October 9, 2026, comparing the adaptive `HC_info::search()` loop
at commit `e6fd92f` with the order-preserving implementation.

## Change

Problem propagation replaces each affected hypercuboid at its original
position. It preserves both the relative order of the original hypercuboids
and the order of each hypercuboid's remaining pieces. The density threshold
remains `10 * disjoint_count < intersect_count`, and secondary removals still
require codimension at most one. The existing fine-tree removal policy retains
its default weight of two.

## Measurement

A temporary native benchmark outside the repository loaded each file's
submitted main-line endpoint using `pgnparser::parse_game()` and `state`.
Parsing and replay were excluded from the measured interval. Each trial
constructed a fresh HC context and search space, then consumed the first ten
legal actions from a single generator, stopping at exhaustion if fewer were
available. Every returned action was independently checked with
`state::can_apply()` after timing.

The previous search loop and corrected search ran in the same executable
against the same core library, alternating their order between trials. Results
are medians of 21 trials per policy after one discarded warm-up trial. The
compiler was Homebrew GCC 15.3.0, using C++23, `-O3`, `-DNDEBUG`, and LTO on
macOS 15.8.1. These are local measurements, not cross-platform guarantees.

| Endpoint | Actions found | Old search (ms) | Corrected search (ms) | Reduction |
| --- | ---: | ---: | ---: | ---: |
| `test/pgn/100timelines.5dpgn` | 0 | 8.511 | 4.849 | 43.0% |
| Attached `222.5dpgn` | 10 | 0.823 | 0.479 | 41.8% |
| Attached `22222222222222222.5dpgn` | 10 | 1.766 | 1.165 | 34.0% |

The repository fixture is named `100timelines.5dpgn`, rather than
`100_timelines.5dpgn`. Its endpoint, after turn 11w, has 120 timelines and is
checkmate. Both policies exhausted their search without yielding an action;
the independent `5dtools checkmate stable` command also reported checkmate.
Consequently its row measures full exhaustion, rather than ten yielded actions.
Both attachments yielded ten actions under both policies.

HC construction dominates the longer attachment:

| Endpoint | Old HC build + search (ms) | Corrected HC build + search (ms) |
| --- | ---: | ---: |
| `100timelines.5dpgn` | 16.662 | 12.969 |
| `222.5dpgn` | 1.970 | 1.616 |
| `22222222222222222.5dpgn` | 91.481 | 89.158 |

For `222.5dpgn`, the first ten actions each create one branch under both
policies. For `22222222222222222.5dpgn`, the old policy's first ten actions each
create one branch; the corrected policy's first ten actions create no branches.
The timings therefore compare each policy's first ten legal actions, rather
than requiring identical action sequences.

## Validation

The Release build with `TOOLS=ON` and `TEST=ON` passed all 49 CTest tests.
Regression coverage checks in-place replacement order, the retained adaptive
density threshold, and ten legal non-branching actions from a small standard
position where the previous policy yielded branching actions first.

See the [hypercuboid guide](../hypercuboid.md#revisiting-the-hypercuboid-search)
for the pruning policy.
