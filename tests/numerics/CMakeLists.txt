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

include(ResolveVulkan)
nlo_resolve_vulkan(_nlo_vk_headers_available _nlo_vk_loader_available)

add_executable(test_nlo_vector_backend
  ${CMAKE_CURRENT_LIST_DIR}/test_nlo_vector_backend.c
  ${CMAKE_SOURCE_DIR}/src/backend/vector_backend.c
  ${CMAKE_SOURCE_DIR}/src/backend/vk_auto_context.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vector_ops.c
  ${CMAKE_SOURCE_DIR}/src/utility/perf_profile.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vk_vector_ops.c
)

target_include_directories(test_nlo_vector_backend PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_BINARY_DIR}/src/generated
)
target_link_libraries(test_nlo_vector_backend PRIVATE simd::simd Vulkan::Vulkan)
if(TARGET nlo_vk_shaders)
  add_dependencies(test_nlo_vector_backend nlo_vk_shaders)
endif()
if(UNIX)
  target_link_libraries(test_nlo_vector_backend PRIVATE m)
endif()

add_test(NAME test_nlo_vector_backend COMMAND test_nlo_vector_backend)

add_executable(test_nlo_simd_wrapper
  ${CMAKE_CURRENT_LIST_DIR}/test_nlo_simd_wrapper.c
)

target_include_directories(test_nlo_simd_wrapper PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_nlo_simd_wrapper PRIVATE simd::simd)
if(UNIX)
  target_link_libraries(test_nlo_simd_wrapper PRIVATE m)
endif()

add_test(NAME test_nlo_simd_wrapper COMMAND test_nlo_simd_wrapper)

add_executable(test_nlo_vector_backend_vulkan
  ${CMAKE_CURRENT_LIST_DIR}/test_nlo_vector_backend_vulkan.c
  ${CMAKE_SOURCE_DIR}/src/backend/vector_backend.c
  ${CMAKE_SOURCE_DIR}/src/backend/vk_auto_context.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vector_ops.c
  ${CMAKE_SOURCE_DIR}/src/utility/perf_profile.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vk_vector_ops.c
)

target_include_directories(test_nlo_vector_backend_vulkan PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_BINARY_DIR}/src/generated
)
target_link_libraries(test_nlo_vector_backend_vulkan PRIVATE simd::simd Vulkan::Vulkan)
if(TARGET nlo_vk_shaders)
  add_dependencies(test_nlo_vector_backend_vulkan nlo_vk_shaders)
endif()
if(UNIX)
  target_link_libraries(test_nlo_vector_backend_vulkan PRIVATE m)
endif()

add_test(NAME test_nlo_vector_backend_vulkan COMMAND test_nlo_vector_backend_vulkan)
