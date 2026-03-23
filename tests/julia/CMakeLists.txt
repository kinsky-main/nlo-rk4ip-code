find_program(Julia_EXECUTABLE julia)

if(NOT Julia_EXECUTABLE)
  message(STATUS "Julia executable not found; skipping Julia tests.")
  return()
endif()

add_test(NAME test_julia_api_smoke
  COMMAND "${Julia_EXECUTABLE}" "--project=${CMAKE_SOURCE_DIR}/julia" "${CMAKE_SOURCE_DIR}/julia/test/runtests.jl"
)

add_test(NAME test_julia_example_soliton
  COMMAND "${Julia_EXECUTABLE}" "--project=${CMAKE_SOURCE_DIR}/examples/julia"
          "-e"
          "using Pkg; Pkg.instantiate(); include(raw\"${CMAKE_SOURCE_DIR}/examples/julia/second_order_soliton_rk4ip.jl\"); main(String[])"
)

add_test(NAME test_julia_example_storage_init
  COMMAND "${Julia_EXECUTABLE}" "--project=${CMAKE_SOURCE_DIR}/examples/julia"
          "${CMAKE_SOURCE_DIR}/tests/julia/test_example_storage_init.jl"
)

set(_NLOLIB_TEST_PATH "$ENV{PATH}")
string(REPLACE ";" "\\;" _NLOLIB_TEST_PATH "${_NLOLIB_TEST_PATH}")

set_tests_properties(test_julia_api_smoke PROPERTIES
  ENVIRONMENT "NLOLIB_LIBRARY=$<TARGET_FILE:nlolib>;PATH=${_NLOLIB_TEST_PATH}"
)

set_tests_properties(test_julia_example_soliton PROPERTIES
  ENVIRONMENT "NLOLIB_LIBRARY=$<TARGET_FILE:nlolib>;PATH=${_NLOLIB_TEST_PATH}"
)

set_tests_properties(test_julia_example_storage_init PROPERTIES
  ENVIRONMENT "NLOLIB_LIBRARY=$<TARGET_FILE:nlolib>;PATH=${_NLOLIB_TEST_PATH}"
)
