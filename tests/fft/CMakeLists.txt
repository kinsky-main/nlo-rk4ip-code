add_executable(test_nlo_complex
  ${CMAKE_CURRENT_LIST_DIR}/test_nlo_complex.c
)

target_include_directories(test_nlo_complex PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_nlo_complex PRIVATE nlolib)

add_test(NAME test_nlo_complex COMMAND test_nlo_complex)

add_executable(test_fft
  ${CMAKE_CURRENT_LIST_DIR}/test_fft.c
  ${CMAKE_SOURCE_DIR}/src/fft/fft.c
)

target_include_directories(test_fft PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_fft PRIVATE nlo_fft_backend)

add_test(NAME test_fft COMMAND test_fft)

set(_NLOLIB_TEST_PATH "$ENV{PATH}")
string(REPLACE ";" "\\;" _NLOLIB_TEST_PATH "${_NLOLIB_TEST_PATH}")
set(_NLOLIB_TEST_PATH "${_NLOLIB_TEST_PATH}\\;C:/libs/fftw")

set_tests_properties(test_fft PROPERTIES
  ENVIRONMENT "PATH=${_NLOLIB_TEST_PATH}"
)
