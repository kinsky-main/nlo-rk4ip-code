# Compile-time ABI parity check between src/nlolib_matlab.h and the canonical
# public headers. All assertions are static; running the binary is a no-op, so
# a drift regression fails the build rather than the test run.

add_executable(test_matlab_header_abi
  ${CMAKE_CURRENT_LIST_DIR}/test_matlab_header_abi.c
)

target_include_directories(test_matlab_header_abi PRIVATE
  ${CMAKE_SOURCE_DIR}/src
)

# Inherit the Vulkan (and other) include paths the canonical headers need.
target_link_libraries(test_matlab_header_abi PRIVATE nlolib)

add_test(NAME test_matlab_header_abi COMMAND test_matlab_header_abi)
