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

# The shipped Linux library must load on glibc 2.35 (Ubuntu 22.04), even when
# it is built on a newer glibc; src/utility/glibc_compat.c keeps it there.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_OBJDUMP)
  add_test(NAME test_glibc_floor
    COMMAND ${CMAKE_COMMAND}
      -DOBJDUMP=${CMAKE_OBJDUMP}
      -DLIBRARY=$<TARGET_FILE:nlolib>
      -DFLOOR=2.35
      -P ${CMAKE_CURRENT_LIST_DIR}/check_glibc_floor.cmake
  )
endif()
