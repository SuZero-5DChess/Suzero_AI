#ifndef CORE_HASH_H
#define CORE_HASH_H

#include <cstdint>

class board;
class multiverse;
class state;

// Deterministic, non-cryptographic hashes of current contents. These do not
// cache values: boards may be mutated and may be shared by several states.
// Equal hashes identify candidates; verify exact equality before deduplicating.
std::uint64_t hash_board(const board &b) noexcept;

// Includes odd/even timeline layout and every board's coordinates and contents.
// Derived shape ranges and nominal dimensions are not separate hash inputs.
std::uint64_t hash_multiverse(const multiverse &m);

// Includes the multiverse plus the stored present, current player, and
// promotion rules. PGN metadata, comments, and recorded results are excluded.
std::uint64_t hash_state(const state &s);

#endif /* CORE_HASH_H */
