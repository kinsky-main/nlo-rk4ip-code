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
  ${CMAKE_SOURCE_DIR}/src/utility/perf_profile.c
)
if(ENABLE_VKFFT)
  set_target_properties(test_fft PROPERTIES LINKER_LANGUAGE CXX)
endif()
if(ENABLE_VULKAN_BACKEND)
  target_sources(test_fft PRIVATE
    ${CMAKE_SOURCE_DIR}/src/backend/vk_auto_context.c
    ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_resources.c
    ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_phase.c
    ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_dispatch.c
    ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_transfer.c
    ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_ops.c
  )
else()
  target_sources(test_fft PRIVATE
    ${CMAKE_SOURCE_DIR}/src/backend/vulkan_stubs.c
  )
endif()

include(ResolveVulkan)
resolve_vulkan(vk_headers_available vk_loader_available)
include(NLOCblas)
configure_cblas(cblas_target cblas_include_dirs)

target_include_directories(test_fft PRIVATE
  ${CMAKE_SOURCE_DIR}/src
)
if(ENABLE_VULKAN_BACKEND)
  target_include_directories(test_fft PRIVATE ${CMAKE_BINARY_DIR}/src/generated)
endif()
target_link_libraries(test_fft PRIVATE fft_backend ${cblas_target})
target_compile_definitions(test_fft PRIVATE
  ENABLE_VULKAN_BACKEND=$<IF:$<BOOL:${ENABLE_VULKAN_BACKEND}>,1,0>
  ENABLE_VKFFT=$<IF:$<BOOL:${ENABLE_VKFFT}>,1,0>
)
if(ENABLE_VULKAN_BACKEND)
  target_link_libraries(test_fft PRIVATE Vulkan::Vulkan)
endif()
if(TARGET vk_shaders)
  add_dependencies(test_fft vk_shaders)
endif()
if(UNIX)
  target_link_libraries(test_fft PRIVATE m)
endif()
if(WIN32)
  add_custom_command(TARGET test_fft POST_BUILD
    COMMAND ${CMAKE_COMMAND}
            -DNLO_RUNTIME_SOURCE="$<TARGET_FILE:test_fft>"
            -DNLO_RUNTIME_DEST="$<TARGET_FILE_DIR:test_fft>"
            -DNLO_RUNTIME_HINTS="${CBLAS_RUNTIME_HINTS}"
            -P "${CMAKE_SOURCE_DIR}/cmake/copy_runtime_deps.cmake"
  )
endif()

add_test(NAME test_fft COMMAND test_fft)

set(_NLOLIB_TEST_PATH "$ENV{PATH}")
string(REPLACE ";" "\\;" _NLOLIB_TEST_PATH "${_NLOLIB_TEST_PATH}")

set_tests_properties(test_fft PROPERTIES
  ENVIRONMENT "PATH=${_NLOLIB_TEST_PATH}"
)
