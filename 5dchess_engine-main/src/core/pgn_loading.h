#ifndef PGN_LOADING_H
#define PGN_LOADING_H

#include "state.h"

// Internal setup loading shared by the state and client game loaders.
namespace pgn_loading {

template<bool UNSAFE>
std::pair<state, std::optional<pgn_validation_error>>
load_initial_position(const pgnparser_ast::game &g);

} /* namespace pgn_loading */
#endif /* PGN_LOADING_H */
