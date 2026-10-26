// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-5DChess contributors
//
// Thin pybind11 adapter over Chess5DGameState and the C++ self-play bridge.
//
// The rules/encoding API is deliberately stateless.  `run_selfplay` is the
// exception: it exposes the C++ PUCT + ONNX Runtime path to the Python trainer
// so that self-play can run on CUDA without a Python round trip per position.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "console_interrupt.h"
#include "suzero/5dchess/game_state.h"
#include "suzero/5dchess/selfplay_bridge.h"

namespace py = pybind11;
void registerMatchBindings(py::module_& m);

using suzero::chess5d::Action;
using suzero::chess5d::BridgeGame;
using suzero::chess5d::EncodedPosition;
using suzero::chess5d::Chess5DGameState;
using suzero::chess5d::GameConfig;
using suzero::chess5d::RecordSnapshot;
using suzero::chess5d::SelfPlayBridgeConfig;

namespace {

std::unique_ptr<Chess5DGameState> makeState(const std::string& variant_pgn,
                                               int board_side, int line_shift,
                                               int board_limit,
                                               double material_scale) {
  GameConfig config;
  config.variant_pgn = variant_pgn;
  config.board_side = board_side;
  config.board_squares = board_side * board_side;
  config.line_shift = line_shift;
  config.board_limit = board_limit;
  config.material_scale = material_scale;
  return std::make_unique<Chess5DGameState>(config);
}

py::list actionList(const Chess5DGameState& state) {
  const auto& actions = state.actions();
  py::list out(actions.size());
  for (std::size_t i = 0; i < actions.size(); ++i) {
    const Action& action = actions[i];
    py::tuple entry(2);
    entry[0] = py::bool_(action.is_submit);
    entry[1] = action.is_submit ? py::none() : py::cast(action.move);
    out[i] = entry;
  }
  return out;
}

py::dict recordSnapshot(const Chess5DGameState& state) {
  const RecordSnapshot snapshot = state.recordSnapshot();

  py::dict size;
  size["x"] = snapshot.sizeX;
  size["y"] = snapshot.sizeY;

  py::list boards;
  for (const auto& board : snapshot.boards) {
    py::dict entry;
    entry["l"] = board.l;
    entry["t"] = board.t;
    entry["c"] = board.c;
    entry["fen"] = board.fen;
    boards.append(std::move(entry));
  }

  py::dict present;
  present["t"] = snapshot.presentT;
  present["c"] = snapshot.presentC;

  py::dict timeline_status;
  timeline_status["mandatory_timelines"] = snapshot.mandatoryTimelines;
  timeline_status["optional_timelines"] = snapshot.optionalTimelines;
  timeline_status["unplayable_timelines"] = snapshot.unplayableTimelines;

  py::dict out;
  out["size"] = std::move(size);
  out["fen"] = snapshot.fen;
  out["boards"] = std::move(boards);
  out["present"] = std::move(present);
  out["timeline_status"] = std::move(timeline_status);
  return out;
}

py::list recordedMoves(const BridgeGame& game) {
  py::list out;
  for (const auto& move : game.moves) {
    py::dict entry;
    entry["player"] = move.player;
    if (move.action.is_submit) {
      entry["type"] = "submit";
    } else {
      py::list from;
      py::list to;
      for (int i = 0; i < 4; ++i) {
        from.append(move.action.move[static_cast<std::size_t>(i)]);
        to.append(move.action.move[static_cast<std::size_t>(i + 4)]);
      }
      entry["type"] = "move";
      entry["from"] = std::move(from);
      entry["to"] = std::move(to);
      entry["promotion"] = "queen";
    }
    out.append(std::move(entry));
  }
  return out;
}

py::dict gameEntry(const BridgeGame& game) {
  py::dict entry;
  entry["index"] = game.index;
  entry["white_outcome"] = game.whiteOutcome;
  entry["terminal_reason"] = game.terminalReason;
  entry["semimoves"] = game.semimoves;
  entry["turns"] = game.turns;
  entry["board_limit"] = game.boardLimit;
  entry["board_count"] = game.boardCount;
  entry["final_player"] = game.finalPlayer;
  entry["moves_played"] = game.movesPlayed;
  entry["tree_samples"] = game.treeSamples;
  entry["samples"] = static_cast<std::int64_t>(game.samples.size());
  entry["moves"] = recordedMoves(game);
  entry["final_fen"] = game.finalFen;
  return entry;
}

constexpr int kPieceChannels = 12;

/// Serialise the samples of one or more games into flat, ragged numpy arrays.
///
/// Padding is only introduced later, inside minibatches.  The same helper
/// serves the whole-batch return value and the per-game streaming callback, so
/// both paths always describe samples identically.
void fillSampleArrays(const std::vector<const BridgeGame*>& games,
                      int board_side, py::dict& out) {
  std::size_t total_samples = 0;
  std::size_t total_boards = 0;
  std::size_t total_actions = 0;
  for (const BridgeGame* game : games) {
    for (const auto& sample : game->samples) {
      ++total_samples;
      total_boards += static_cast<std::size_t>(sample.encoded.num_boards);
      total_actions += sample.actions.size();
    }
  }

  const py::ssize_t samples = static_cast<py::ssize_t>(total_samples);
  const py::ssize_t boards = static_cast<py::ssize_t>(total_boards);
  const py::ssize_t actions = static_cast<py::ssize_t>(total_actions);
  const py::ssize_t squares = static_cast<py::ssize_t>(board_side * board_side);

  py::array_t<float> board_planes(
      std::vector<py::ssize_t>{boards, static_cast<py::ssize_t>(kPieceChannels),
                               squares});
  py::array_t<std::int64_t> l_coords(boards);
  py::array_t<std::int64_t> t_coords(boards);
  py::array_t<std::int64_t> board_offsets(samples + 1);
  py::array_t<std::int64_t> action_offsets(samples + 1);
  py::array_t<std::int64_t> used_board_counts(samples);
  py::array_t<float> urgency(samples);
  py::array_t<std::int64_t> side_to_move(samples);
  py::array_t<std::int64_t> action_data(
      std::vector<py::ssize_t>{actions, static_cast<py::ssize_t>(8)});
  py::array_t<std::int64_t> action_counts(samples);
  py::array_t<float> policy(actions);
  py::array_t<bool> has_submit(samples);
  py::array_t<std::int64_t> player(samples);
  py::array_t<std::int64_t> move_count(samples);
  py::array_t<float> value(samples);
  py::array_t<std::int64_t> game_index(samples);
  py::array_t<std::int64_t> sample_source(samples);
  py::array_t<std::int64_t> search_visits(samples);

  const std::size_t plane_board_size =
      static_cast<std::size_t>(kPieceChannels) *
      static_cast<std::size_t>(squares);
  float* planes_ptr = board_planes.mutable_data();
  std::int64_t* l_ptr = l_coords.mutable_data();
  std::int64_t* t_ptr = t_coords.mutable_data();
  std::int64_t* board_offsets_ptr = board_offsets.mutable_data();
  std::int64_t* action_offsets_ptr = action_offsets.mutable_data();
  board_offsets_ptr[0] = 0;
  action_offsets_ptr[0] = 0;
  std::int64_t* action_ptr = action_data.mutable_data();
  float* policy_ptr = policy.mutable_data();

  std::int64_t* used_ptr = used_board_counts.mutable_data();
  float* urgency_ptr = urgency.mutable_data();
  std::int64_t* side_ptr = side_to_move.mutable_data();
  std::int64_t* action_counts_ptr = action_counts.mutable_data();
  bool* submit_ptr = has_submit.mutable_data();
  std::int64_t* player_ptr = player.mutable_data();
  std::int64_t* move_count_ptr = move_count.mutable_data();
  float* value_ptr = value.mutable_data();
  std::int64_t* game_index_ptr = game_index.mutable_data();
  std::int64_t* source_ptr = sample_source.mutable_data();
  std::int64_t* visits_ptr = search_visits.mutable_data();

  std::size_t row = 0;
  std::size_t board_offset = 0;
  std::size_t action_offset = 0;
  for (const BridgeGame* game : games) {
    for (const auto& sample : game->samples) {
      const std::size_t n =
          static_cast<std::size_t>(sample.encoded.num_boards);
      const std::size_t count = sample.actions.size();
      const bool submit =
          !sample.actions.empty() && sample.actions.back().is_submit;

      std::copy(sample.encoded.board_planes.begin(),
                sample.encoded.board_planes.end(),
                planes_ptr + board_offset * plane_board_size);
      std::copy(sample.encoded.l_coords.begin(), sample.encoded.l_coords.end(),
                l_ptr + board_offset);
      std::copy(sample.encoded.t_coords.begin(), sample.encoded.t_coords.end(),
                t_ptr + board_offset);

      used_ptr[row] = static_cast<std::int64_t>(n);
      urgency_ptr[row] = sample.encoded.urgency;
      side_ptr[row] = sample.encoded.side_to_move;
      action_counts_ptr[row] = static_cast<std::int64_t>(count);
      submit_ptr[row] = submit;
      player_ptr[row] = sample.player;
      move_count_ptr[row] = sample.moveCount;
      value_ptr[row] = sample.value;
      game_index_ptr[row] = game->index;
      source_ptr[row] = sample.source;
      visits_ptr[row] = sample.searchVisits;

      for (std::size_t j = 0; j < count; ++j) {
        policy_ptr[action_offset + j] = sample.policy[j];
        std::int64_t* destination = action_ptr + (action_offset + j) * 8;
        if (sample.actions[j].is_submit) {
          std::fill_n(destination, 8, std::int64_t{0});
          continue;
        }
        const auto& move = sample.actions[j].move;
        for (std::size_t k = 0; k < move.size(); ++k) {
          destination[k] = move[k];
        }
      }
      board_offset += n;
      action_offset += count;
      ++row;
      board_offsets_ptr[row] = static_cast<std::int64_t>(board_offset);
      action_offsets_ptr[row] = static_cast<std::int64_t>(action_offset);
    }
  }

  out["board_offsets"] = std::move(board_offsets);
  out["action_offsets"] = std::move(action_offsets);
  out["board_planes"] = std::move(board_planes);
  out["l_coords"] = std::move(l_coords);
  out["t_coords"] = std::move(t_coords);
  out["used_board_counts"] = std::move(used_board_counts);
  out["urgency"] = std::move(urgency);
  out["side_to_move"] = std::move(side_to_move);
  out["actions"] = std::move(action_data);
  out["action_counts"] = std::move(action_counts);
  out["policy"] = std::move(policy);
  out["has_submit"] = std::move(has_submit);
  out["player"] = std::move(player);
  out["move_count"] = std::move(move_count);
  out["value"] = std::move(value);
  out["game_index"] = std::move(game_index);
  out["sample_source"] = std::move(sample_source);
  out["search_visits"] = std::move(search_visits);
}

py::dict streamPayload(const BridgeGame& game, int board_side) {
  py::dict out;
  fillSampleArrays({&game}, board_side, out);
  py::list game_list;
  game_list.append(gameEntry(game));
  out["games"] = std::move(game_list);
  return out;
}

py::dict encodePosition(const Chess5DGameState& state) {
  const EncodedPosition encoded = state.encodePosition();
  const std::size_t boards = static_cast<std::size_t>(encoded.num_boards);
  const std::size_t channels = static_cast<std::size_t>(encoded.piece_channels);
  const std::size_t squares = static_cast<std::size_t>(encoded.board_squares);

  py::array_t<float> planes(
      {static_cast<py::ssize_t>(boards), static_cast<py::ssize_t>(channels),
       static_cast<py::ssize_t>(squares)});
  if (!encoded.board_planes.empty()) {
    std::memcpy(planes.mutable_data(), encoded.board_planes.data(),
                encoded.board_planes.size() * sizeof(float));
  }

  py::array_t<std::int64_t> l_coords(static_cast<py::ssize_t>(boards));
  py::array_t<std::int64_t> t_coords(static_cast<py::ssize_t>(boards));
  if (boards > 0) {
    std::memcpy(l_coords.mutable_data(), encoded.l_coords.data(),
                boards * sizeof(std::int64_t));
    std::memcpy(t_coords.mutable_data(), encoded.t_coords.data(),
                boards * sizeof(std::int64_t));
  }

  py::dict out;
  out["num_boards"] = encoded.num_boards;
  out["board_planes"] = std::move(planes);
  out["l_coords"] = std::move(l_coords);
  out["t_coords"] = std::move(t_coords);
  out["urgency"] = encoded.urgency;
  out["side_to_move"] = encoded.side_to_move;
  out["board_side"] = encoded.board_side;
  out["board_squares"] = encoded.board_squares;
  out["piece_channels"] = encoded.piece_channels;
  return out;
}

py::dict runSelfPlayBinding(
    const std::string& model_path, const std::string& metadata_path,
    const std::string& provider, int cuda_device_id,
    const std::string& variant_pgn, int board_side, int line_shift,
    int num_games, int num_threads, int num_simulations, int fast_simulations,
    bool playout_cap_randomization, double full_search_probability,
    double c_puct, double dirichlet_alpha,
    double dirichlet_epsilon, int max_depth, double temperature,
    double temperature_final, int temperature_threshold, int min_board_limit,
    int max_board_limit, double material_scale, int max_batch_size,
    std::uint64_t seed, py::object progress, int num_server_threads = 1,
    bool record_games = false, bool record_tree_positions = true,
    int record_tree_min_visits = 64, int record_tree_max_depth = 5,
    int record_tree_max_per_turn = 4,
    double record_tree_sample_probability = 0.25, bool stream_games = false,
    py::object on_game = py::none()) {
  SelfPlayBridgeConfig config;
  config.evaluator.model_path = model_path;
  config.evaluator.metadata_path = metadata_path;
  config.evaluator.provider = provider;
  config.evaluator.cuda_device_id = cuda_device_id;
  config.evaluator.board_side = board_side;
  config.evaluator.board_squares = board_side * board_side;
  config.evaluator.piece_channels = kPieceChannels;

  config.game.variant_pgn = variant_pgn;
  config.game.board_side = board_side;
  config.game.board_squares = board_side * board_side;
  config.game.line_shift = line_shift;
  config.game.board_limit = max_board_limit;
  config.game.material_scale = material_scale;

  config.numGames = num_games;
  config.numGameThreads = num_threads;
  config.numSimulations = num_simulations;
  config.fastSimulations = fast_simulations;
  config.playoutCapRandomization = playout_cap_randomization;
  config.fullSearchProbability = full_search_probability;
  config.cpuctExploration = c_puct;
  config.dirichletAlpha = dirichlet_alpha;
  config.dirichletEpsilon = dirichlet_epsilon;
  config.maxDepth = max_depth;
  config.temperature = temperature;
  config.temperatureFinal = temperature_final;
  config.temperatureThreshold = temperature_threshold;
  config.minBoardLimit = min_board_limit;
  config.maxBoardLimit = max_board_limit;
  config.materialScale = material_scale;
  config.maxBatchSize = max_batch_size;
  config.numServerThreads = num_server_threads;
  config.recordGames = record_games;
  config.recordTreePositions = record_tree_positions;
  config.recordTreeMinVisits = record_tree_min_visits;
  config.recordTreeMaxDepth = record_tree_max_depth;
  config.recordTreeMaxPerTurn = record_tree_max_per_turn;
  config.recordTreeSampleProbability = record_tree_sample_probability;
  config.seed = seed;

  // ``progress`` only reports how many games are done; ``on_game`` receives
  // the samples of each finished game so a file-backed replay can persist it
  // immediately instead of holding the whole round in RAM.
  suzero::chess5d::BridgeGameCallback callback;
  const bool have_progress = !progress.is_none();
  const bool have_stream = stream_games && !on_game.is_none();
  if (have_progress || have_stream) {
    py::function progress_function;
    py::function stream_function;
    if (have_progress) {
      progress_function = py::cast<py::function>(progress);
    }
    if (have_stream) {
      stream_function = py::cast<py::function>(on_game);
    }
    callback = [progress_function, stream_function, have_progress, have_stream,
                board_side](int done, const BridgeGame& game) {
      py::gil_scoped_acquire gil;
      if (have_progress) {
        progress_function(done, game.index);
      }
      if (have_stream) {
        stream_function(done, game.index, streamPayload(game, board_side));
      }
    };
  }

  std::vector<BridgeGame> games;
  suzero::chess5d::SelfPlayStats stats;
  bool stopped = false;
  {
    // The GIL is released for the whole batch, so Python's SIGINT handler
    // cannot run until this call returns. The guard acknowledges Ctrl+C
    // natively, sets its stop flag, and hands the signal back to Python; the
    // scheduler stops starting new games while in-flight games finish.
    suzero::chess5d::internal::ConsoleInterruptGuard interrupt_guard;
    py::gil_scoped_release release;
    games = suzero::chess5d::runSelfPlay(
        config, callback, &stats,
        []() { return suzero::chess5d::internal::stopRequested(); });
    stopped = suzero::chess5d::internal::stopRequested();
  }

  py::list game_list;
  for (const BridgeGame& game : games) {
    game_list.append(gameEntry(game));
  }

  py::dict stats_out;
  stats_out["server_threads"] = stats.numServerThreads;
  stats_out["max_batch_size"] = stats.maxBatchSize;
  stats_out["requests"] = stats.requests;
  stats_out["batches"] = stats.batches;
  stats_out["rows"] = stats.rows;
  stats_out["max_observed_batch"] = stats.maxObservedBatch;
  stats_out["total_seconds"] = stats.totalSeconds;
  stats_out["setup_seconds"] = stats.setupSeconds;

  py::dict out;
  out["stats"] = std::move(stats_out);
  if (!have_stream) {
    // Streamed samples already went to the replay store from the callback;
    // returning them again would load the whole round into RAM.
    std::vector<const BridgeGame*> game_refs;
    game_refs.reserve(games.size());
    for (const BridgeGame& game : games) {
      game_refs.push_back(&game);
    }
    fillSampleArrays(game_refs, board_side, out);
  }
  out["games"] = std::move(game_list);
  out["stopped"] = stopped;
  return out;
}

}  // namespace

PYBIND11_MODULE(_5dchess, m) {
  registerMatchBindings(m);
  m.attr("RECORD_FORMAT_VERSION") = 1;
  m.attr("TREE_POSITION_FORMAT_VERSION") = 1;
  m.attr("GAME_STREAM_FORMAT_VERSION") = 1;
  m.doc() =
      "5D chess rules/encoding adapter for suzero_5dchess, plus the C++ "
      "PUCT/ONNX self-play bridge.";

  py::class_<Chess5DGameState>(m, "GameState")
      .def(py::init(&makeState), py::arg("variant_pgn"),
           py::arg("board_side") = 4, py::arg("line_shift") = 5,
           py::arg("board_limit") = 25, py::arg("material_scale") = 2.0)
      .def("clone", &Chess5DGameState::cloneState,
           "Return an independent copy of this position.")
      .def("actions", &actionList,
           "Candidate actions as [(is_submit, move_key_or_None), ...].")
      .def("action_count", &Chess5DGameState::actionCount)
      .def("apply_action", &Chess5DGameState::applyAction,
           py::arg("action_id"))
      .def_property_readonly("current_player",
                             &Chess5DGameState::currentPlayer)
      .def_property_readonly("is_terminal", &Chess5DGameState::isTerminal)
      .def_property_readonly("terminal_utility",
                             &Chess5DGameState::terminalUtility)
      .def_property_readonly("state_key", &Chess5DGameState::stateKey)
      .def_property_readonly("move_count", &Chess5DGameState::moveCount)
      .def_property_readonly("board_count", &Chess5DGameState::boardCount)
      .def_property_readonly("board_limit", &Chess5DGameState::boardLimit)
      .def_property_readonly("urgency", &Chess5DGameState::urgency)
      .def_property_readonly("terminal_reason",
                             &Chess5DGameState::terminalReason)
      .def_property_readonly("native_hash", &Chess5DGameState::nativeHash)
      .def("record_snapshot", &recordSnapshot,
           "Return complete native state data for .5dck archival.")
      .def("encode", &encodePosition,
           "Encode the position for the smallstep_movekey_v1 network.");

  m.def(
      "run_selfplay", &runSelfPlayBinding,
      py::arg("model_path"), py::arg("metadata_path"), py::arg("provider"),
      py::arg("cuda_device_id"), py::arg("variant_pgn"), py::arg("board_side"),
      py::arg("line_shift"), py::arg("num_games"), py::arg("num_threads"),
      py::arg("num_simulations"), py::arg("fast_simulations"),
      py::arg("playout_cap_randomization"), py::arg("full_search_probability"),
      py::arg("c_puct"),
      py::arg("dirichlet_alpha"), py::arg("dirichlet_epsilon"),
      py::arg("max_depth"), py::arg("temperature"),
      py::arg("temperature_final"), py::arg("temperature_threshold"),
      py::arg("min_board_limit"), py::arg("max_board_limit"),
      py::arg("material_scale"), py::arg("max_batch_size"), py::arg("seed"),
      py::arg("progress") = py::none(),
      py::arg("num_server_threads") = 1,
      py::arg("record_games") = false,
      py::arg("record_tree_positions") = true,
      py::arg("record_tree_min_visits") = 64,
      py::arg("record_tree_max_depth") = 5,
      py::arg("record_tree_max_per_turn") = 4,
      py::arg("record_tree_sample_probability") = 0.25,
      py::arg("stream_games") = false, py::arg("on_game") = py::none(),
      "Run C++ PUCT + ONNX Runtime self-play on several game threads.\n\n"
      "Returns padded sample arrays plus one metadata dict per game. "
      "'provider' is \"cpu\" or \"cuda\"; a CUDA request fails if the "
      "provider is unavailable. 'num_server_threads' is the number of ONNX "
      "evaluator server threads (each owns a model copy). With "
      "'stream_games', 'on_game(done, index, payload)' receives one game's "
      "samples as soon as that game finishes; 'payload' has the same keys as "
      "the return value, restricted to that game.");
}
