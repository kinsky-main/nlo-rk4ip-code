add_executable(test_nlo_numerics
  ${CMAKE_CURRENT_LIST_DIR}/test_nlo_numerics.c
)

target_include_directories(test_nlo_numerics PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_nlo_numerics PRIVATE nlolib)

add_test(NAME test_nlo_numerics COMMAND test_nlo_numerics)
