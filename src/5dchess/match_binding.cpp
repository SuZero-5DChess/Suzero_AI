// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-02 contributors

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utility>

#include "suzero/5dchess/match.h"

namespace py = pybind11;

namespace {

py::dict scoreDict(const suzero::mcts::MatchScore& score) {
  py::dict out;
  out["wins"] = score.wins;
  out["losses"] = score.losses;
  out["draws"] = score.draws;
  out["truncated"] = score.truncated;
  return out;
}

py::dict gameDict(const suzero::mcts::MatchResult& game, int index) {
  py::dict out;
  out["index"] = index;
  out["a_is_white"] = index % 2 == 0;
  out["terminal"] = game.terminal;
  out["truncated"] = game.truncated;
  out["final_player"] = game.finalPlayer;
  out["final_utility"] = game.finalUtility;
  out["participant_utilities"] = game.participantUtilities;
  out["terminal_reason"] = game.terminalReason;
  py::list moves;
  for (const auto& move : game.moves) {
    py::dict item;
    item["player"] = move.player;
    item["participant_index"] = move.participantIndex;
    item["action"] = move.action;
    item["local_actions"] = move.localActions;
    moves.append(std::move(item));
  }
  out["moves"] = std::move(moves);
  return out;
}

py::dict runMatchBinding(
    const suzero::chess5d::OnnxMatchParticipant& a,
    const suzero::chess5d::OnnxMatchParticipant& b,
    int pairs, int max_moves, std::uint64_t seed, py::object progress,
    int num_threads) {
  if (!progress.is_none() && !PyCallable_Check(progress.ptr())) {
    throw py::type_error("progress must be callable or None");
  }
  suzero::mcts::MatchSeriesConfig config;
  config.pairs = pairs;
  config.game.maxMoves = max_moves;
  config.game.seed = seed;
  config.numGameThreads = num_threads;

  // Copy configs before releasing the GIL: another Python thread may modify
  // the bound objects, but cannot change a match that is already in progress.
  const std::array<suzero::chess5d::OnnxMatchParticipant, 2> bots{{a, b}};
  suzero::mcts::MatchSeriesResult result;
  int completed = 0;
  {
    py::gil_scoped_release release;
    result = suzero::chess5d::runOnnxMatchSeries(
        config, bots, [&progress, &completed](int index, const auto& game) {
          py::gil_scoped_acquire gil;
          if (PyErr_CheckSignals() != 0) {
            throw py::error_already_set();
          }
          ++completed;
          if (!progress.is_none()) {
            progress(completed, gameDict(game, index));
          }
        });
  }

  py::dict out;
  out["participants"] = py::make_tuple(bots[0].name, bots[1].name);
  out["totals"] = py::make_tuple(scoreDict(result.totals[0]),
                                  scoreDict(result.totals[1]));
  py::list by_seat;
  for (const auto& seats : result.bySeat) {
    py::dict item;
    item["white"] = scoreDict(seats[0]);
    item["black"] = scoreDict(seats[1]);
    by_seat.append(std::move(item));
  }
  out["by_color"] = std::move(by_seat);
  py::list games;
  for (std::size_t i = 0; i < result.games.size(); ++i) {
    games.append(gameDict(result.games[i], static_cast<int>(i)));
  }
  out["games"] = std::move(games);
  return out;
}

}  // namespace

void registerMatchBindings(py::module_& m) {
  using suzero::chess5d::GameConfig;
  using suzero::chess5d::OnnxEvaluatorConfig;
  using suzero::chess5d::OnnxMatchParticipant;
  using suzero::mcts::SearchParams;

  py::class_<GameConfig>(m, "MatchGameConfig")
      .def(py::init<>())
      .def_readwrite("variant_pgn", &GameConfig::variant_pgn)
      .def_readwrite("board_side", &GameConfig::board_side)
      .def_readwrite("board_squares", &GameConfig::board_squares)
      .def_readwrite("piece_channels", &GameConfig::piece_channels)
      .def_readwrite("line_shift", &GameConfig::line_shift)
      .def_readwrite("board_limit", &GameConfig::board_limit)
      .def_readwrite("material_scale", &GameConfig::material_scale);
  py::class_<OnnxEvaluatorConfig>(m, "MatchEvaluatorConfig")
      .def(py::init<>())
      .def_readwrite("model_path", &OnnxEvaluatorConfig::model_path)
      .def_readwrite("metadata_path", &OnnxEvaluatorConfig::metadata_path)
      .def_readwrite("provider", &OnnxEvaluatorConfig::provider)
      .def_readwrite("cuda_device_id", &OnnxEvaluatorConfig::cuda_device_id)
      .def_readwrite("intra_op_threads", &OnnxEvaluatorConfig::intra_op_threads)
      .def_readwrite("board_side", &OnnxEvaluatorConfig::board_side)
      .def_readwrite("board_squares", &OnnxEvaluatorConfig::board_squares)
      .def_readwrite("piece_channels", &OnnxEvaluatorConfig::piece_channels);
  py::class_<SearchParams>(m, "MatchSearchParams")
      .def(py::init<>())
      .def_readwrite("max_playouts", &SearchParams::maxPlayouts)
      .def_readwrite("max_visits", &SearchParams::maxVisits)
      .def_readwrite("max_time_seconds", &SearchParams::maxTimeSeconds)
      .def_readwrite("max_depth", &SearchParams::maxDepth)
      .def_readwrite("c_puct", &SearchParams::cpuctExploration)
      .def_readwrite("temperature", &SearchParams::chosenMoveTemperature)
      .def_readwrite("temperature_early", &SearchParams::chosenMoveTemperatureEarly);
  py::class_<OnnxMatchParticipant>(m, "MatchBot")
      .def(py::init<>())
      .def_readwrite("name", &OnnxMatchParticipant::name)
      .def_readwrite("game", &OnnxMatchParticipant::game)
      .def_readwrite("evaluator", &OnnxMatchParticipant::evaluator)
      .def_readwrite("search", &OnnxMatchParticipant::search)
      .def_readwrite("reverse_move_order", &OnnxMatchParticipant::reverseMoveOrder);
  m.def("run_match", &runMatchBinding, py::arg("a"), py::arg("b"),
        py::arg("pairs") = 1, py::arg("max_moves") = 0,
        py::arg("seed") = suzero::mcts::kDefaultSeed,
        py::arg("progress") = py::none(), py::arg("num_threads") = 1,
        "Run a color-balanced C++ match with independent ONNX models. "
        "Move limits count every action, including SUBMIT; truncations have "
        "None utilities and are not draws. num_threads caps concurrent games; "
        "progress runs on the calling thread after each completed game.");
}
