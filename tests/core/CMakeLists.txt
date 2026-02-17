add_executable(test_core_state_alloc
  ${CMAKE_CURRENT_LIST_DIR}/test_state_alloc.c
  ${CMAKE_SOURCE_DIR}/src/core/state.c
  ${CMAKE_SOURCE_DIR}/src/core/init.c
  ${CMAKE_SOURCE_DIR}/src/backend/vector_backend.c
  ${CMAKE_SOURCE_DIR}/src/backend/vk_auto_context.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vector_ops.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vk_vector_ops.c
  ${CMAKE_SOURCE_DIR}/src/numerics/math_ops.c
  ${CMAKE_SOURCE_DIR}/src/physics/operators.c
  ${CMAKE_SOURCE_DIR}/src/numerics/rk4_kernel.c
  ${CMAKE_SOURCE_DIR}/src/utility/rk4_debug.c
  ${CMAKE_SOURCE_DIR}/src/fft/fft.c
)

include(ResolveVulkan)
nlo_resolve_vulkan(_nlo_vk_headers_available _nlo_vk_loader_available)

target_include_directories(test_core_state_alloc PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_BINARY_DIR}/src/generated
)
target_link_libraries(test_core_state_alloc PRIVATE nlo_fft_backend simd::simd Vulkan::Vulkan)
if(TARGET nlo_vk_shaders)
  add_dependencies(test_core_state_alloc nlo_vk_shaders)
endif()
if(WIN32)
  target_compile_definitions(test_core_state_alloc PRIVATE NLOLIB_EXPORTS)
endif()
if(UNIX)
  target_link_libraries(test_core_state_alloc PRIVATE m)
endif()

add_test(NAME test_core_state_alloc COMMAND test_core_state_alloc)

set(_NLOLIB_TEST_PATH "$ENV{PATH}")
string(REPLACE ";" "\\;" _NLOLIB_TEST_PATH "${_NLOLIB_TEST_PATH}")

set_tests_properties(test_core_state_alloc PROPERTIES
  ENVIRONMENT "PATH=${_NLOLIB_TEST_PATH}"
)
