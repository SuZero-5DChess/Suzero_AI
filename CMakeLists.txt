cmake_minimum_required(VERSION 3.16)

project(suzero_mcts
  VERSION 0.2.0
  DESCRIPTION "Game-agnostic PUCT graph search core derived from KataGo"
  LANGUAGES CXX
)

option(SUZERO_MCTS_BUILD_TESTS "Build the unit tests" ON)

add_library(suzero_mcts STATIC
  src/puct_search.cpp
  src/random.cpp
  src/search_node.cpp
)

target_include_directories(suzero_mcts PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

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
  )

  if(MSVC)
    target_compile_options(suzero_mcts_tests PRIVATE /W4 /permissive-)
  else()
    target_compile_options(suzero_mcts_tests PRIVATE -Wall -Wextra -Wpedantic)
  endif()

  add_test(NAME suzero_mcts_tests COMMAND suzero_mcts_tests)
endif()

include(GNUInstallDirs)
install(TARGETS suzero_mcts
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
