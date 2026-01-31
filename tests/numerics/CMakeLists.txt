add_executable(test_nlo_numerics
  ${CMAKE_CURRENT_LIST_DIR}/test_nlo_numerics.c
  ${CMAKE_SOURCE_DIR}/src/numerics/math_ops.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vector_ops.c
)

target_include_directories(test_nlo_numerics PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_test(NAME test_nlo_numerics COMMAND test_nlo_numerics)
