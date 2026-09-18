add_executable(test_nlo_numerics
  ${CMAKE_CURRENT_LIST_DIR}/test_nlo_numerics.c
  ${CMAKE_SOURCE_DIR}/src/numerics/math_ops.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vector_ops.c
)

target_include_directories(test_nlo_numerics PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(test_nlo_numerics PRIVATE
  ENABLE_VULKAN_BACKEND=$<IF:$<BOOL:${ENABLE_VULKAN_BACKEND}>,1,0>
  ENABLE_VKFFT=$<IF:$<BOOL:${ENABLE_VKFFT}>,1,0>
)
if(UNIX)
  target_link_libraries(test_nlo_numerics PRIVATE m)
endif()
if(WIN32)
  add_custom_command(TARGET test_nlo_numerics POST_BUILD
    COMMAND ${CMAKE_COMMAND}
            -DNLO_RUNTIME_SOURCE="$<TARGET_FILE:test_nlo_numerics>"
            -DNLO_RUNTIME_DEST="$<TARGET_FILE_DIR:test_nlo_numerics>"
            -P "${CMAKE_SOURCE_DIR}/cmake/copy_runtime_deps.cmake"
  )
endif()

add_test(NAME test_nlo_numerics COMMAND test_nlo_numerics)

include(ResolveVulkan)
resolve_vulkan(vk_headers_available vk_loader_available)

set(vector_backend_test_sources
  ${CMAKE_SOURCE_DIR}/src/backend/vector_backend.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vector_ops.c
  ${CMAKE_SOURCE_DIR}/src/utility/perf_profile.c
)
if(ENABLE_VULKAN_BACKEND)
  list(APPEND vector_backend_test_sources
    ${CMAKE_SOURCE_DIR}/src/backend/vk_auto_context.c
    ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_resources.c
    ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_phase.c
    ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_dispatch.c
    ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_transfer.c
    ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_ops.c
  )
else()
  list(APPEND vector_backend_test_sources
    ${CMAKE_SOURCE_DIR}/src/backend/vulkan_stubs.c
  )
endif()

add_executable(test_nlo_vector_backend
  ${CMAKE_CURRENT_LIST_DIR}/test_nlo_vector_backend.c
  ${vector_backend_test_sources}
)

target_include_directories(test_nlo_vector_backend PRIVATE
  ${CMAKE_SOURCE_DIR}/src
)
if(ENABLE_VULKAN_BACKEND)
  target_include_directories(test_nlo_vector_backend PRIVATE ${CMAKE_BINARY_DIR}/src/generated)
endif()
target_compile_definitions(test_nlo_vector_backend PRIVATE
  ENABLE_VULKAN_BACKEND=$<IF:$<BOOL:${ENABLE_VULKAN_BACKEND}>,1,0>
  ENABLE_VKFFT=$<IF:$<BOOL:${ENABLE_VKFFT}>,1,0>
)
if(ENABLE_VULKAN_BACKEND)
  target_link_libraries(test_nlo_vector_backend PRIVATE Vulkan::Vulkan)
endif()
include(NLOVulkanBackend)
nlolib_use_embedded_spirv(test_nlo_vector_backend)
if(UNIX)
  target_link_libraries(test_nlo_vector_backend PRIVATE m)
endif()
if(WIN32)
  add_custom_command(TARGET test_nlo_vector_backend POST_BUILD
    COMMAND ${CMAKE_COMMAND}
            -DNLO_RUNTIME_SOURCE="$<TARGET_FILE:test_nlo_vector_backend>"
            -DNLO_RUNTIME_DEST="$<TARGET_FILE_DIR:test_nlo_vector_backend>"
            -P "${CMAKE_SOURCE_DIR}/cmake/copy_runtime_deps.cmake"
  )
endif()

add_test(NAME test_nlo_vector_backend COMMAND test_nlo_vector_backend)

if(ENABLE_VULKAN_BACKEND)
  add_executable(test_nlo_vector_backend_vulkan
    ${CMAKE_CURRENT_LIST_DIR}/test_nlo_vector_backend_vulkan.c
    ${vector_backend_test_sources}
  )

  target_include_directories(test_nlo_vector_backend_vulkan PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}/src/generated
  )
  target_link_libraries(test_nlo_vector_backend_vulkan PRIVATE Vulkan::Vulkan)
  target_compile_definitions(test_nlo_vector_backend_vulkan PRIVATE
    ENABLE_VULKAN_BACKEND=$<IF:$<BOOL:${ENABLE_VULKAN_BACKEND}>,1,0>
    ENABLE_VKFFT=$<IF:$<BOOL:${ENABLE_VKFFT}>,1,0>
  )
  nlolib_use_embedded_spirv(test_nlo_vector_backend_vulkan)
  if(UNIX)
    target_link_libraries(test_nlo_vector_backend_vulkan PRIVATE m)
  endif()
  if(WIN32)
    add_custom_command(TARGET test_nlo_vector_backend_vulkan POST_BUILD
      COMMAND ${CMAKE_COMMAND}
              -DNLO_RUNTIME_SOURCE="$<TARGET_FILE:test_nlo_vector_backend_vulkan>"
              -DNLO_RUNTIME_DEST="$<TARGET_FILE_DIR:test_nlo_vector_backend_vulkan>"
              -P "${CMAKE_SOURCE_DIR}/cmake/copy_runtime_deps.cmake"
    )
  endif()

  add_test(NAME test_nlo_vector_backend_vulkan COMMAND test_nlo_vector_backend_vulkan)
endif()
