# Move generation

Move generation produces candidate piece destinations. For action generation,
please refer to [action-generation.md](action-generation.md).

## Files

- [`bitboard.h`](../../src/core/bitboard.h), `bitboard.cpp`, and `bitboard.inl`: board-square sets and spatial geometry.
- [`magic.h`](../../src/core/magic.h) and `magic.cpp`: magic bitboards for sliding pieces, accounting for blockers.
- [`multiverse.h`](../../src/core/multiverse.h) and `multiverse.cpp`: physical and superphysical piece movement; [`move_geometry.h`](../../src/core/move_geometry.h) supplies shared time/timeline directions.
- [`state.h`](../../src/core/state.h) and `state.cpp`: player context, movable pieces, and checking moves; [`check_position.h`](../../src/core/check_position.h) and `check_position.cpp` provide dedicated check scans over borrowed board views.
- [`test_movegen.cpp`](../../test/test_movegen.cpp): pawn, brawn, and castling regression tests.

Run the `test_movegen` test executable with `-l` to print each test group,
position FEN, expected move count, and passing result.

## Public API

- `multiverse::gen_physical_moves<C>(p)`: destination squares on the source board.
- `multiverse::gen_superphysical_moves<C>(p)` and `gen_moves<C>(p)`: lazy destinations grouped by board; the latter includes physical moves. `C` is `false` for White and `true` for Black.
- `multiverse::gen_piece_move(p, color)`: individual destination coordinates. `state::gen_piece_move(p)` uses the current player. `state::get_movable_pieces()` lists sources on playable timelines.
- `state::find_checks(attacker)`: moves capturing enemy royals. `check_position::checks()` and `first_check()` scan for all checks or one check without general move generation.

Piece destinations alone do not establish turn legality. Lazy generators borrow
their source objects, which must remain alive and unchanged during iteration.

## Pawn and brawn movement

An unmoved pawn or brawn can advance two squares on its board or two timelines
in its forward direction. Both the intermediate and destination squares must
be empty, and the required boards must exist. A moved piece can advance only
one step. White's forward directions are increasing ranks and decreasing
timeline indices; Black's are decreasing ranks and increasing timeline indices.

A brawn interprets pawn captures as one-step diagonals along any two axes,
including at least one forward direction and no backward rank or timeline
direction. File and time can use either direction. This gives nine geometric
possibilities:

| Capture directions | Possibilities | Pawn | Brawn |
|---|---:|---|---|
| Forward rank + left/right file | 2 | Yes | Yes |
| Forward timeline + past/future time | 2 | Yes | Yes |
| Forward timeline + left/right file | 2 | No | Yes |
| Forward timeline + forward rank | 1 | No | Yes |
| Forward rank + past time | 1 | No | Yes |
| Forward rank + future time | 1 | No | Geometric only |

Only **eight** are available in play: a piece moving from its timeline's end
at `(L, T)` has no same-color board at `(L, T+1)`, so the last row cannot be a
capture. Another timeline may already extend further ahead, allowing the
future-time capture in the second row. Pawns have at most **four** ordinary
capture destinations. Every ordinary capture requires an enemy at the
destination; these diagonals do not allow quiet moves.

Pawns and brawns can capture an adjacent enemy pawn or brawn en passant after
its physical double-step, provided the landing square is empty.

