# Content hashes

This is a very first design, intended for the project's currently minimal
use of hashing. It computes hashes from the current contents on each call,
keeping the implementation simple and avoiding cache lifetime and mutation
tracking. These development notes describe an implementation that is not
yet considered stable; the improvements below are possibilities for heavier
use, not implemented features.

Include `core/hash.h` to use `hash_board(const board &)`,
`hash_multiverse(const multiverse &)`, or `hash_state(const state &)`. Each
returns a deterministic `std::uint64_t` computed from the current contents.
The header contains declarations; implementation details live in
`src/core/hash.cpp`. It combines integer fields in a fixed order with the
[SplitMix64 mixing function](https://prng.di.unimi.it/splitmix64.c). It uses
fixed constants and unsigned arithmetic, without hashing raw object memory,
pointer addresses, or allocating FEN strings.

Board hashes include every piece bitboard and the unmoved-piece mask.
Multiverse hashes additionally include the odd/even timeline layout, obtained
from `get_initial_lines_range()` through virtual dispatch without RTTI, and
every board at its logical timeline, time, and color coordinates. Derived
timeline ranges and start/end turns only bound traversal; they are not
separate hash inputs. Nominal dimensions are also omitted: a 4x4 board and
an 8x8 board with matching wall and piece contents have equal hashes.
Historical boards are included even when timeline endpoints are identical.
Insertion order and shared board ownership do not affect the result.
State hashes also include the stored present, current player, and promotion
rules; PGN metadata, comments, and
recorded results do not affect them.

These are non-cryptographic hashes for finding candidate matches. Confirm
exact content equality before treating matching hashes as duplicates. Hashes
are recomputed without caching, including for boards shared between states.
The integer encoding is independent of platform
byte order and pointer size, but persisted hashes should identify the hash
schema version: the board tag uses version 1, and the multiverse and state
tags use version 2.

If repeated hashing becomes significant, profiling should first establish
its cost and how often board objects are reused. Caching each board's content
hash could save work across states that share historical boards, but boards
remain mutable, so every mutation and assignment would need to invalidate
the cache. A cache stored inside the board would follow its lifetime; an
external pointer cache would also need to handle destruction and address
reuse. Cache memory, lookup costs, and any synchronization should be measured
against recomputing the board's twelve bitboard fields. The current multiverse
combiner feeds those fields directly into a running accumulator; combining
cached board hashes instead would change the hash schema and require new
multiverse and state version tags.

Caching board hashes would still require visiting every board to construct
a multiverse hash. If that traversal becomes a bottleneck during search, a
further design could maintain the multiverse hash as boards are added or
changed, using separately computed contributions that include each board's
logical coordinates. This would require a combiner that supports such
updates and a clear policy for mutations of boards shared by multiple states.
The update scheme should be checked against full recomputation, including
branching timelines and changes to the state's present and promotion rules.

If the number of stored hashes grows substantially, collision requirements
should also be revisited. A wider hash could reduce accidental candidate
matches at the cost of more storage and computation. Exact content comparison
would remain necessary before removing a duplicate, and changes to the
encoding or combiner would need explicit schema versioning.
