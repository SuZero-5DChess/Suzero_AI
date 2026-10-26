cmake_minimum_required(VERSION 3.16)

project(suzero_mcts
  VERSION 0.5.0
  DESCRIPTION "SuZero AI 5D chess AlphaZero; search core derived from KataGo"
  LANGUAGES CXX
)

option(SUZERO_MCTS_BUILD_TESTS "Build the unit tests" ON)

# ---------------------------------------------------------------- 5D chess --
option(SUZERO_BUILD_5DCHESS
  "Build the 5D chess (Very Small - Open) adapter" OFF)
option(SUZERO_BUILD_PYTHON_BINDINGS
  "Build the pybind11 rules adapter used by python/suzero_5dchess" OFF)

set(SUZERO_5DCHESS_ENGINE_ROOT "" CACHE PATH
  "Path to the pinned 5dchess_engine source tree (contains src/core/state.h)")
set(SUZERO_ONNXRUNTIME_ROOT "" CACHE PATH
  "Path to a prebuilt ONNX Runtime SDK (include/, lib/onnxruntime.lib)")
set(SUZERO_5DCHESS_MODEL
  "${CMAKE_CURRENT_SOURCE_DIR}/models/5dchess/very_small_latest_fp32.onnx"
  CACHE FILEPATH "Path to the smallstep_movekey_v1 ONNX model")

find_package(Threads REQUIRED)

add_library(suzero_mcts STATIC
  src/batch_evaluator.cpp
  src/match.cpp
  src/puct_search.cpp
  src/random.cpp
  src/search_node.cpp
  src/selfplay.cpp
)

target_include_directories(suzero_mcts PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(suzero_mcts PUBLIC Threads::Threads)

target_compile_features(suzero_mcts PUBLIC cxx_std_17)

if(MSVC)
  target_compile_options(suzero_mcts PRIVATE /W4 /permissive-)
else()
  target_compile_options(suzero_mcts PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(SUZERO_MCTS_BUILD_TESTS)
  enable_testing()

  add_executable(suzero_mcts_tests
    tests/test_main.cpp
  )
  target_link_libraries(suzero_mcts_tests PRIVATE suzero_mcts)
  target_include_directories(suzero_mcts_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )

  if(MSVC)
    target_compile_options(suzero_mcts_tests PRIVATE /W4 /permissive-)
  else()
    target_compile_options(suzero_mcts_tests PRIVATE -Wall -Wextra -Wpedantic)
  endif()

  add_test(NAME suzero_mcts_tests COMMAND suzero_mcts_tests)

  add_executable(suzero_match_tests tests/test_match.cpp)
  target_link_libraries(suzero_match_tests PRIVATE suzero_mcts)
  if(MSVC)
    target_compile_options(suzero_match_tests PRIVATE /W4 /permissive-)
  else()
    target_compile_options(suzero_match_tests PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  add_test(NAME suzero_match_tests COMMAND suzero_match_tests)
endif()

if(SUZERO_BUILD_5DCHESS)
  if(NOT SUZERO_5DCHESS_ENGINE_ROOT)
    set(_suzero_engine_default
      "${CMAKE_CURRENT_SOURCE_DIR}/5dchess_engine-main")
    if(EXISTS "${_suzero_engine_default}/src/core/state.h")
      set(SUZERO_5DCHESS_ENGINE_ROOT "${_suzero_engine_default}")
    else()
      message(FATAL_ERROR
        "SUZERO_5DCHESS_ENGINE_ROOT is not set and "
        "${_suzero_engine_default} does not exist. Point it at the pinned "
        "5dchess_engine source tree.")
    endif()
  endif()

  if(NOT SUZERO_ONNXRUNTIME_ROOT)
    set(_suzero_onnx_default
      "${CMAKE_CURRENT_SOURCE_DIR}/extern/onnxruntime-win-x64-1.24.3")
    if(EXISTS "${_suzero_onnx_default}/include/onnxruntime_cxx_api.h")
      set(SUZERO_ONNXRUNTIME_ROOT "${_suzero_onnx_default}")
    else()
      message(FATAL_ERROR
        "SUZERO_ONNXRUNTIME_ROOT is not set and no default SDK was found. "
        "Point it at a prebuilt ONNX Runtime SDK.")
    endif()
  endif()

  set(SUZERO_ONNXRUNTIME_INCLUDE_DIR "${SUZERO_ONNXRUNTIME_ROOT}/include")
  set(SUZERO_ONNXRUNTIME_LIB "${SUZERO_ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")

  foreach(_required
      "${SUZERO_5DCHESS_ENGINE_ROOT}/src/core/state.h"
      "${SUZERO_ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h"
      "${SUZERO_ONNXRUNTIME_LIB}")
    if(NOT EXISTS "${_required}")
      message(FATAL_ERROR "Required 5D chess dependency not found: ${_required}")
    endif()
  endforeach()

  file(GLOB_RECURSE SUZERO_5DCHESS_ENGINE_SOURCES CONFIGURE_DEPENDS
    "${SUZERO_5DCHESS_ENGINE_ROOT}/src/core/*.cpp"
    "${SUZERO_5DCHESS_ENGINE_ROOT}/src/engine/*.cpp"
    "${SUZERO_5DCHESS_ENGINE_ROOT}/src/misc/*.cpp"
    "${SUZERO_5DCHESS_ENGINE_ROOT}/src/client/*.cpp"
  )

  # The pinned upstream engine requires C++20/23. Keep the standard on these
  # targets only; the public adapter headers remain C++17.
  add_library(suzero_5dchess_engine STATIC ${SUZERO_5DCHESS_ENGINE_SOURCES})
  set_target_properties(suzero_5dchess_engine PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(suzero_5dchess_engine PUBLIC
    "${SUZERO_5DCHESS_ENGINE_ROOT}/src"
    "${SUZERO_5DCHESS_ENGINE_ROOT}/src/core"
    "${SUZERO_5DCHESS_ENGINE_ROOT}/src/client"
    "${SUZERO_5DCHESS_ENGINE_ROOT}/src/engine"
    "${SUZERO_5DCHESS_ENGINE_ROOT}/src/misc"
  )
  target_compile_definitions(suzero_5dchess_engine PUBLIC
    PROJECT_VERSION_STRING="0.0.0"
  )
  if(MSVC)
    target_compile_options(suzero_5dchess_engine PRIVATE
      /utf-8 /constexpr:steps100000000)
  endif()

  add_library(suzero_5dchess STATIC
    src/5dchess/game_state.cpp
    src/5dchess/match.cpp
    src/5dchess/match_game.cpp
    src/5dchess/onnx_batch_backend.cpp
    src/5dchess/onnx_evaluator.cpp
    src/5dchess/onnx_model.cpp
    src/5dchess/agent.cpp
    src/5dchess/selfplay_bridge.cpp
    src/5dchess/tree_positions.cpp
  )
  target_link_libraries(suzero_5dchess
    PUBLIC suzero_mcts
    PRIVATE suzero_5dchess_engine "${SUZERO_ONNXRUNTIME_LIB}"
  )
  target_include_directories(suzero_5dchess
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE "${SUZERO_ONNXRUNTIME_INCLUDE_DIR}"
  )
  set_target_properties(suzero_5dchess PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
  )
  if(MSVC)
    target_compile_options(suzero_5dchess PRIVATE /utf-8)
    target_compile_definitions(suzero_5dchess PRIVATE NOMINMAX
      WIN32_LEAN_AND_MEAN)
  endif()

  # ------------------------------------------------- ONNX Runtime runtime --
  # The import library above only provides the stable C ABI, so any matching
  # onnxruntime.dll works at runtime. CUDAExecutionProvider exists only in a
  # GPU build of that DLL, so prefer the onnxruntime-gpu installation of the
  # Python environment (or an explicit SDK).
  #
  # Nothing is copied into the Python package: the extension registers the
  # runtime directory with os.add_dll_directory at import time (see
  # python/suzero_5dchess/_ort_runtime.py). C++ executables get the two small
  # DLLs by hardlink instead of copy, so several build configurations share one
  # physical file. SUZERO_ORT_STAGE_CUDA additionally stages the ~299 MB CUDA
  # provider for self-contained C++ test directories.
  set(SUZERO_ONNXRUNTIME_RUNTIME_DIR "" CACHE PATH
    "Directory holding the runtime onnxruntime.dll (a GPU build enables CUDA)")
  if(NOT SUZERO_ONNXRUNTIME_RUNTIME_DIR)
    find_package(Python 3.9 COMPONENTS Interpreter QUIET)
    if(Python_Interpreter_FOUND)
      execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
          "import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi'))"
        OUTPUT_VARIABLE _suzero_ort_python_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _suzero_ort_python_result
        ERROR_QUIET)
      if(_suzero_ort_python_result EQUAL 0
          AND EXISTS "${_suzero_ort_python_dir}/onnxruntime.dll")
        set(SUZERO_ONNXRUNTIME_RUNTIME_DIR "${_suzero_ort_python_dir}")
      endif()
    endif()
  endif()
  if(NOT SUZERO_ONNXRUNTIME_RUNTIME_DIR)
    set(SUZERO_ONNXRUNTIME_RUNTIME_DIR "${SUZERO_ONNXRUNTIME_ROOT}/lib")
  endif()
  message(STATUS
    "5D chess ONNX Runtime DLLs: ${SUZERO_ONNXRUNTIME_RUNTIME_DIR}")

  # The C++ tests default to the CPU provider, so the CUDA provider stays out
  # of every output directory unless explicitly requested.
  option(SUZERO_ORT_STAGE_CUDA
    "Stage onnxruntime_providers_cuda.dll next to C++ executables" OFF)

  set(_suzero_ort_runtime_files
    onnxruntime.dll
    onnxruntime_providers_shared.dll)
  if(SUZERO_ORT_STAGE_CUDA)
    list(APPEND _suzero_ort_runtime_files onnxruntime_providers_cuda.dll)
  endif()

  function(suzero_stage_ort_runtime target)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND}
        "-DSUZERO_RUNTIME_SOURCE_DIR=${SUZERO_ONNXRUNTIME_RUNTIME_DIR}"
        "-DSUZERO_RUNTIME_TARGET_DIR=$<TARGET_FILE_DIR:${target}>"
        "-DSUZERO_RUNTIME_FILES=${_suzero_ort_runtime_files}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/link_runtime_files.cmake"
      VERBATIM)
  endfunction()

  if(SUZERO_BUILD_PYTHON_BINDINGS)
    find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
      OUTPUT_VARIABLE SUZERO_PYBIND11_CMAKE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE SUZERO_PYBIND11_QUERY_RESULT)
    if(NOT SUZERO_PYBIND11_QUERY_RESULT EQUAL 0)
      message(FATAL_ERROR
        "pybind11 is not installed for ${Python_EXECUTABLE}. "
        "Install it with: \"${Python_EXECUTABLE}\" -m pip install pybind11")
    endif()
    list(APPEND CMAKE_PREFIX_PATH "${SUZERO_PYBIND11_CMAKE_DIR}")
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(suzero_5dchess_py
      src/5dchess/python_binding.cpp
      src/5dchess/match_binding.cpp)
    target_link_libraries(suzero_5dchess_py PRIVATE suzero_5dchess)
    set_target_properties(suzero_5dchess_py PROPERTIES
      OUTPUT_NAME "_5dchess"
      CXX_STANDARD 23
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
    )
    if(MSVC)
      target_compile_options(suzero_5dchess_py PRIVATE /utf-8)
      target_compile_definitions(suzero_5dchess_py PRIVATE NOMINMAX
        WIN32_LEAN_AND_MEAN)
    endif()

    # Keep the extension importable directly from the source tree. ONNX
    # Runtime DLLs are resolved by suzero_5dchess/_ort_runtime.py at import
    # time, so no runtime files are staged here.
    set(SUZERO_PY_PACKAGE_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/python/suzero_5dchess")
    foreach(_cfg "" _DEBUG _RELEASE _RELWITHDEBINFO _MINSIZEREL)
      set_target_properties(suzero_5dchess_py PROPERTIES
        "LIBRARY_OUTPUT_DIRECTORY${_cfg}" "${SUZERO_PY_PACKAGE_DIR}"
        "RUNTIME_OUTPUT_DIRECTORY${_cfg}" "${SUZERO_PY_PACKAGE_DIR}")
    endforeach()
  endif()

  if(SUZERO_MCTS_BUILD_TESTS)
    add_executable(suzero_5dchess_tests
      tests/test_5dchess.cpp
    )
    target_link_libraries(suzero_5dchess_tests PRIVATE suzero_5dchess)
    target_compile_definitions(suzero_5dchess_tests PRIVATE
      SUZERO_5DCHESS_MODEL_PATH="${SUZERO_5DCHESS_MODEL}"
    )
    if(MSVC)
      target_compile_options(suzero_5dchess_tests PRIVATE /utf-8)
    endif()

    add_test(NAME suzero_5dchess_tests
      COMMAND suzero_5dchess_tests)

    suzero_stage_ort_runtime(suzero_5dchess_tests)

    add_executable(suzero_5dchess_match_tests tests/test_5dchess_match.cpp)
    target_link_libraries(suzero_5dchess_match_tests PRIVATE suzero_5dchess)
    target_compile_definitions(suzero_5dchess_match_tests PRIVATE
      SUZERO_5DCHESS_MODEL_PATH="${SUZERO_5DCHESS_MODEL}")
    if(MSVC)
      target_compile_options(suzero_5dchess_match_tests PRIVATE /utf-8 /W4)
    endif()
    add_test(NAME suzero_5dchess_match_tests COMMAND suzero_5dchess_match_tests)
    suzero_stage_ort_runtime(suzero_5dchess_match_tests)

    add_executable(suzero_5dchess_tree_position_tests
      tests/test_5dchess_tree_positions.cpp)
    target_link_libraries(suzero_5dchess_tree_position_tests PRIVATE suzero_5dchess)
    target_compile_definitions(suzero_5dchess_tree_position_tests PRIVATE
      SUZERO_5DCHESS_MODEL_PATH="${SUZERO_5DCHESS_MODEL}")
    target_include_directories(suzero_5dchess_tree_position_tests PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src)
    if(MSVC)
      target_compile_options(suzero_5dchess_tree_position_tests PRIVATE /utf-8 /W4)
    endif()
    add_test(NAME suzero_5dchess_tree_position_tests
      COMMAND suzero_5dchess_tree_position_tests)
    suzero_stage_ort_runtime(suzero_5dchess_tree_position_tests)
  endif()
endif()

include(GNUInstallDirs)
install(TARGETS suzero_mcts
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
