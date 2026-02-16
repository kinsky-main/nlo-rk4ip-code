add_executable(test_nlo_complex
  ${CMAKE_CURRENT_LIST_DIR}/test_nlo_complex.c
)

target_include_directories(test_nlo_complex PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_nlo_complex PRIVATE nlolib)

add_test(NAME test_nlo_complex COMMAND test_nlo_complex)

add_executable(test_fft
  ${CMAKE_CURRENT_LIST_DIR}/test_fft.c
  ${CMAKE_SOURCE_DIR}/src/fft/fft.c
  ${CMAKE_SOURCE_DIR}/src/backend/vector_backend.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vector_ops.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vk_vector_ops.c
)

include(ResolveVulkan)
nlo_resolve_vulkan(_nlo_vk_headers_available _nlo_vk_loader_available)

target_include_directories(test_fft PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_BINARY_DIR}/src/generated
)
target_link_libraries(test_fft PRIVATE nlo_fft_backend simd::simd Vulkan::Vulkan)
if(TARGET nlo_vk_shaders)
  add_dependencies(test_fft nlo_vk_shaders)
endif()
if(UNIX)
  target_link_libraries(test_fft PRIVATE m)
endif()

add_test(NAME test_fft COMMAND test_fft)

set(_NLOLIB_TEST_PATH "$ENV{PATH}")
string(REPLACE ";" "\\;" _NLOLIB_TEST_PATH "${_NLOLIB_TEST_PATH}")

set_tests_properties(test_fft PROPERTIES
  ENVIRONMENT "PATH=${_NLOLIB_TEST_PATH}"
)
