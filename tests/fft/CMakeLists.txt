add_executable(test_nlo_complex
  ${CMAKE_CURRENT_LIST_DIR}/test_nlo_complex.c
)

target_include_directories(test_nlo_complex PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_nlo_complex PRIVATE nlolib)

add_test(NAME test_nlo_complex COMMAND test_nlo_complex)
