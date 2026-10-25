# Parsing and printing

This guide describes the conversion between PGN text, its syntax tree, and game
objects, together with FEN parsing and loading diagnostics. The notation itself
is specified by [Branched 5DPGN](../branched-5dpgn.md) and the
[formal grammar](../pgn-bnf.txt). See [Core game logic](core.md) for position
representation and move generation, and [Client architecture](client.md) for
history navigation and editing.

## Conversion stages

```text
PGN text -> pgnparser_ast::game -> state or game
state, moves, actions, or game -> notation text
```

[`pgnparser`](../../src/core/pgnparser.h), implemented in
[`pgnparser.cpp`](../../src/core/pgnparser.cpp), recognizes syntax and constructs
the types in [`ast.h`](../../src/core/ast.h). It does not determine whether a
move is applicable in a position. Its optional parsing methods allow grammar
alternatives to fail without throwing; committed syntax errors throw
`parse_error`. The full-game parser rejects unconsumed trailing input and
retains useful failure locations across backtracking.

`parse_board_fen_metadata()` extracts a FEN string and timeline/turn/color
coordinates from a board entry. The internal
`pgn_loading::load_initial_position<UNSAFE>()` in
[`pgn_loading.cpp`](../../src/core/pgn_loading.cpp) interprets headers, default
variants, board dimensions, timeline layout, and promotion options to construct
the initial state. Both state and client game loading use this operation. It
returns an optional royal-exposure diagnostic so callers can reject the position
or retain illegal history; it does not apply moves. `variant_setup_t` holds only
setup data, with no source text or locations. Configuration failures are
translated into `pgn_validation_error`; malformed promotion-header values
originate as typed `invalid_promotions` exceptions from `parse_promotions()`.

`state::parse_move()` resolves a syntactic move against a position and returns a
unique move, no match, or matching candidates. Its probing behavior is also
used when choosing short notation during printing.

`state::apply_move_ast()` is the shared conversion boundary for loaders and
tools. It resolves one AST move, normalizes promotion, applies it, and returns
the normalized `ext_move`. It throws `pgn_validation_error` on failure. It does
not submit a turn, choose variations, or construct history nodes.

## Source locations and ownership

`pgnparser_ast::source_range` records zero-based byte offsets with an exclusive
end. The parser and the AST root share one immutable original input string;
`pgnparser_ast::game::source_text` keeps it alive after the parser is destroyed.
Move and action nodes store optional ranges into that string, rather than owning
individual source fragments. Action ranges include the turn label and moves,
excluding trailing comments and whitespace. Header locations are stored separately
in `game::header_sources`, keyed by the same lowercase names as `headers`. Each
entry records the complete declaration and the value inside its quotes, including
an empty range for an empty value.

Programmatically constructed ASTs may omit source text and ranges. Loading can
still interpret their semantic fields; diagnostics use an owned AST-description
fallback when original text is unavailable. Source ranges describe provenance
and do not become positions in newly printed output.

`parse_error` retains shared ownership of its source. Validation errors retain
either a source location with shared ownership or an owned fallback description.
Their text remains available after the parser and AST have been destroyed.
Offsets are bytes, so consumers using another indexing convention must convert
them before highlighting text.

## Loading policies

`state::from_pgn<UNSAFE>(text)` parses text and delegates to
`state::from_ast<UNSAFE>(parsed_game)`. Both default to `UNSAFE=false`, validating
the initial setup and each submitted action on the main line (the last variation
at each branch). They throw on royal-safety violations, including an illegal
earlier action even if subsequent actions resolve the exposure. Sibling
variations are outside this single-state loader's scope. With `UNSAFE=true`,
royal-exposure checks are skipped; move applicability and completed turns are
still required, and malformed syntax, setup data, and promotions are rejected.
Direct construction with `state(multiverse, promotions)` performs no royal-safety
validation.

The setup loader is declared only in the internal `pgn_loading.h`, shared by
`state.cpp` and `client/game.cpp`. Its checked specialization returns an optional
royal-exposure diagnostic; the unchecked specialization skips detection. Client
loading uses the checked specialization even when accepting illegal history,
so the root receives the correct illegal-history flag.

`game::from_pgn(text, options)` uses `game::load_options`, with
`allow_illegal_history` defaulting to false. It loads metadata, comments, and all
variations. Strict loading rejects exposed royals in the initial setup and after
completed actions. Passing `{.allow_illegal_history = true}` retains such history
for inspection. Both modes reject malformed syntax, invalid or ambiguous moves,
illegal promotions, and actions that fail to advance the present. The option
applies to loading and is not retained as a permanent editing policy.

Royal-safety detection is nonthrowing with respect to validation failures.
`state::find_royal_exposure(true)` checks both attacker colors for an initial
setup. After submission from a legal position, `find_royal_exposure(false)`
checks only the new side to move, the opponent of the player who just moved.
The result includes a witnessing capture and its notation. Strict loaders throw
from that result; permissive loading records the
[illegal-history flag](client.md#illegal-history). These checks use existing endpoints,
including inactive timelines; ordinary check on the defender's turn is allowed.

### Game history construction

PGN loading resolves duplicate canonical actions with **last occurrence wins**.
The incoming subtree is fully constructed and validated before the loader uses
`find_child`, `delete_child`, and `add_child` to replace the earlier branch.
Replacement discards the earlier comments and all descendants, and appends the
survivor at the later occurrence's position. Other siblings retain their order.
This applies recursively, including actions whose independently ordered moves
canonicalize to the same sequence. No merging is performed.

The recursive loader returns the last surviving endpoint to select after import,
avoiding navigation pointers into replaced subtrees. Failed loading leaves the
caller's existing game intact.

## Error types

The three exception classes are declared in
[`pgnerror.h`](../../src/core/pgnerror.h), with message generation and source
handling in [`pgnerror.cpp`](../../src/core/pgnerror.cpp). All derive from
`std::invalid_argument`; each reason is an alternative of `error_type_t`, a
`std::variant`. The error module does not depend on board or state definitions.

| Class | Responsibility | Reason alternatives |
|---|---|---|
| `board_fen_error` | Square contents and board shape | `rank_too_short`, `rank_too_long`, `too_few_ranks`, `too_many_ranks`, `invalid_piece`, `invalid_unmoved_marker` |
| `parse_error` | PGN syntax | `unexpected_character`, `unexpected_input`, `unterminated_comment`, `unterminated_metadata`, `unterminated_header_value`, `integer_out_of_range`, `turn_out_of_range`, `timeline_out_of_range`, `nonconsecutive_turn`, `duplicate_header` |
| `pgn_validation_error` | Interpretation and validation of parsed PGN | `invalid_move`, `ambiguous_move`, `illegal_promotion`, `illegal_move`, `present_not_advanced`, `royal_exposed`, `invalid_size`, `unknown_variant`, `invalid_timeline`, `invalid_promotions`, `missing_setup` |

`unexpected_input` is shared by lexer and parser expectation failures. An empty
range identifies missing input; it means EOF only at the source length.
Board-metadata parsing on its own reports local offsets; `parse_game()` rebases
those errors into the complete PGN.

Validation errors quote the original move or action when source data is
available. Ambiguity payloads retain all distinct candidates and their PGN
notation, while the message displays two and the total count:

```text
Ambiguous move: Nf3; possible candidates: (0T1)Nd2f3, (0T1)Nh2f3 (totally 2)
Ambiguous move: Nf3; possible candidates: (0T1)Nd2f3, (0T1)Nh2f3, ... (totally 3)
```

`royal_exposed` retains the witnessing `full_move`, its full notation, and
`is_from_initial_setup` to select initial-setup versus action wording.
`present_not_advanced` identifies a failed turn submission separately.
`illegal_promotion` carries the requested piece and allowed promotion options;
`illegal_move` retains the resolved move that failed application.

Setup resolution uses a private `invalid_variant` carrying the offending header
key and a structured reason. The internal setup loader catches this type and
translates it to `pgn_validation_error`, attaching the header value range when
available. The `Board` header takes precedence over `Variant`, and diagnostics
refer to the header actually used. Missing setup has no source location;
programmatic ASTs without header locations use the header value as a description.
The size reason distinguishes format, conversion, and bounds failures and carries
the maximum dimension. Promotion reasons retain an invalid piece or a wildcard
conflict. `parse_promotions()` throws its own typed `invalid_promotions` exception
(derived from `std::invalid_argument`); the setup resolver translates only that
exception, without inspecting message text. Both loaders propagate the translated
validation error, including when illegal history is permitted.

Python maps these invalid-argument exceptions to `ValueError`. The WebAssembly
loader preserves its result-object convention and distinguishes `parse_error`
and `pgn_validation_error` through the `type` field. Other invalid-argument
exceptions, including board-FEN errors, use `invalid_argument`; unrelated
runtime errors use `runtime_error`. Failures also carry `diagnostic` with
`category`, a stable variant-derived `code`, `message`, an optional exclusive
`range`, and `canAllowIllegalHistory`. Only `royal_exposed` enables the permission
hint. The binding converts source byte ranges to UTF-16 offsets for JavaScript;
unlocated runtime errors have a null range. Legacy failure fields remain intact.

## Board FEN

The `board` FEN constructor parses square contents directly into its bitboards
and unmoved mask. Padding outside the declared dimensions starts as walls;
every rank inside the rectangle must specify the full width. Explicit `*`
markers populate the unmoved mask after
placing the corresponding pieces; ordinary `set_piece()` edits continue to
clear that square's unmoved flag. Board construction does not depend on PGN
parsing: `pgnparser` extracts FEN strings and timeline/turn metadata, while the
board constructor interprets the square notation.

### Internal wall extension

This implementation uses `#` in a FEN rank to represent one `WALL_PIECE` square.
This is a project-specific extension, not part of the Branched 5DPGN standard.
Both the C++ board parser and the analyzer's JavaScript FEN parser accept it.
Each `#` advances the file position by one; consecutive walls are printed as
individual `#` characters. Digits compress empty squares only, and `*` is not
valid after a wall.

`board::get_fen(width, height)` prints
exactly the declared rectangle, from the highest rank to the lowest. Walls
inside that rectangle must be printed, including entire ranks of walls. Squares
outside the rectangle are omitted regardless of their contents. Standalone
boards do not store dimensions: the printer defaults to 8 × 8, and multiverse
serialization passes its stored dimensions explicitly. The bitboard printer's
`SHOW_UMOVE` template option preserves unmoved markers; its default omits them.

The constructor rejects short or overflowing ranks and incorrect rank counts.
Walls inside the declared rectangle must be explicit `#` characters. Game and
PGN serialization use the same size-aware printer, so interior walls and fully
walled ranks survive export and reloading.

### FEN diagnostics

FEN failures throw `board_fen_error`, derived from `std::invalid_argument`.
It owns the original FEN, an algebraic error reason, and an exclusive-end byte
range. Shape-error alternatives carry expected/actual counts through their
`invalid_shape` base; piece and marker errors carry no counts. Short ranks highlight the entire rank, excluding
its separating slashes; overflow highlights the first token exceeding the width
(including `*` when present). Missing ranks use an empty range at end of input;
extra ranks highlight the first excess slash. Invalid pieces highlight one
character, and invalid unmoved markers highlight the piece/marker pair or a
standalone `*`. At overflow, the actual count is the width reached by the
offending token. Invalid dimension arguments use ordinary `std::invalid_argument`
because they do not refer to a location within the FEN. FEN-relative getters
remain unchanged. During PGN loading, the internal setup loader
attaches the enclosing source and rebased range through `get_pgn_source()` and
`get_pgn_range()`. Parsed ASTs carry parallel `board_sources` entries for each
board declaration and FEN substring. Initial royal-exposure errors use the
attacker's endpoint declaration as their source location; programmatic ASTs
without source metadata retain description-only diagnostics.

The WebAssembly binding sends FEN strings for current and phantom boards. The
worker builds `board.parsed` using the JavaScript FEN parser; there is no separate
binding-generated square array. The renderer recognizes `#` as unavailable
board space and retains the board-border fill at those squares.

## Printing

Printing currently converts objects directly to text; it does not construct an
AST first. A `full_move` stores only its source and destination coordinates.
Its `to_string()` output is context-free; `lan(state)` and `pgn(state, options)`
use the position for piece and timeline information. Move and action notation
is implemented in [`action.cpp`](../../src/core/action.cpp).

`game::show_pgn()` exports metadata, the initial FEN, comments, and either the
complete variation tree or the selected path. Its options control notation and
metadata selection. The AST's stream operators in
[`ast.cpp`](../../src/core/ast.cpp) produce diagnostic descriptions, rather than
PGN serialization. Optional source metadata does not prevent constructing an
AST for a future object-to-AST printing path.
