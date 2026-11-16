find_program(Julia_EXECUTABLE julia)

if(NOT Julia_EXECUTABLE)
  message(STATUS "Julia executable not found; skipping Julia tests.")
  return()
endif()

add_test(NAME test_julia_api_smoke
  COMMAND "${Julia_EXECUTABLE}" "--project=${CMAKE_CURRENT_LIST_DIR}/.." "${CMAKE_CURRENT_LIST_DIR}/runtests.jl"
)

add_test(NAME test_julia_example_soliton
  COMMAND "${Julia_EXECUTABLE}" "--project=${CMAKE_CURRENT_LIST_DIR}/../examples"
          "-e"
          "using Pkg; Pkg.instantiate(); include(raw\"${CMAKE_CURRENT_LIST_DIR}/../examples/second_order_soliton_rk4ip.jl\"); main(String[])"
)

set(_NLOLIB_TEST_PATH "$ENV{PATH}")
string(REPLACE ";" "\\;" _NLOLIB_TEST_PATH "${_NLOLIB_TEST_PATH}")

set_tests_properties(test_julia_api_smoke PROPERTIES
  ENVIRONMENT "NLOLIB_LIBRARY=$<TARGET_FILE:nlolib>;PATH=${_NLOLIB_TEST_PATH}"
)

set_tests_properties(test_julia_example_soliton PROPERTIES
  ENVIRONMENT "NLOLIB_LIBRARY=$<TARGET_FILE:nlolib>;PATH=${_NLOLIB_TEST_PATH}"
)
