# Tests CMakeLists.txt

add_subdirectory(fft)
add_subdirectory(core)
add_subdirectory(numerics)
add_subdirectory(python)
add_subdirectory(julia)
add_subdirectory(abi)

# Static check of the MATLAB binding's C type-name strings. Needs no MATLAB
# installation, so it always runs.
add_test(NAME test_matlab_binding_types
  COMMAND "${CMAKE_COMMAND}"
          "-DNLOLIB_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
          -P "${CMAKE_SOURCE_DIR}/cmake/check_matlab_binding_types.cmake"
)

if(NLOLIB_BUILD_MATLAB_TESTS)
  add_subdirectory(matlab)
endif()
