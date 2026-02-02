add_executable(test_nlo_numerics
  ${CMAKE_CURRENT_LIST_DIR}/test_nlo_numerics.c
  ${CMAKE_SOURCE_DIR}/src/numerics/math_ops.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vector_ops.c
)

target_include_directories(test_nlo_numerics PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_nlo_numerics PRIVATE simd::simd)
if(UNIX)
  target_link_libraries(test_nlo_numerics PRIVATE m)
endif()

add_test(NAME test_nlo_numerics COMMAND test_nlo_numerics)

add_executable(test_nlo_vector_backend
  ${CMAKE_CURRENT_LIST_DIR}/test_nlo_vector_backend.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vector_backend.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vector_ops.c
)

target_include_directories(test_nlo_vector_backend PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_nlo_vector_backend PRIVATE simd::simd)
if(UNIX)
  target_link_libraries(test_nlo_vector_backend PRIVATE m)
endif()

add_test(NAME test_nlo_vector_backend COMMAND test_nlo_vector_backend)
