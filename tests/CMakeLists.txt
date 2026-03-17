# Tests CMakeLists.txt

add_subdirectory(fft)
add_subdirectory(core)
add_subdirectory(numerics)
add_subdirectory(python)
add_subdirectory(julia)
if(NLOLIB_BUILD_MATLAB_TESTS)
  add_subdirectory(matlab)
endif()
