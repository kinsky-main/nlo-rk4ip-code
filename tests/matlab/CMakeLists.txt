find_program(MATLAB_EXECUTABLE NAMES matlab)
if(NOT MATLAB_EXECUTABLE)
  message(STATUS "MATLAB executable not found; skipping MATLAB tests.")
  return()
endif()

add_test(NAME test_matlab_runtime_handle_parser
  COMMAND "${MATLAB_EXECUTABLE}" -batch
          "run('${CMAKE_SOURCE_DIR}/tests/matlab/run_matlab_tests.m')"
)

set(_NLOLIB_TEST_PATH "$ENV{PATH}")
string(REPLACE ";" "\\;" _NLOLIB_TEST_PATH "${_NLOLIB_TEST_PATH}")

set_tests_properties(test_matlab_runtime_handle_parser PROPERTIES
  ENVIRONMENT "NLOLIB_LIBRARY=$<TARGET_FILE:nlolib>;PATH=${_NLOLIB_TEST_PATH}"
)
