add_executable(test_core_state_alloc
  ${CMAKE_CURRENT_LIST_DIR}/test_state_alloc.c
  ${CMAKE_SOURCE_DIR}/src/core/init.c
  ${CMAKE_SOURCE_DIR}/src/core/init_defaults.c
  ${CMAKE_SOURCE_DIR}/src/core/init_limits.c
  ${CMAKE_SOURCE_DIR}/src/core/init_state_builder.c
  ${CMAKE_SOURCE_DIR}/src/core/state_lifecycle.c
  ${CMAKE_SOURCE_DIR}/src/core/state_snapshots.c
  ${CMAKE_SOURCE_DIR}/src/backend/vector_backend.c
  ${CMAKE_SOURCE_DIR}/src/backend/vk_auto_context.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vector_ops.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_resources.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_phase.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_dispatch.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_transfer.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_ops.c
  ${CMAKE_SOURCE_DIR}/src/numerics/math_ops.c
  ${CMAKE_SOURCE_DIR}/src/physics/operators.c
  ${CMAKE_SOURCE_DIR}/src/physics/operator_program_compile.c
  ${CMAKE_SOURCE_DIR}/src/physics/operator_program_execute.c
  ${CMAKE_SOURCE_DIR}/src/numerics/rk4_kernel.c
  ${CMAKE_SOURCE_DIR}/src/io/log_format.c
  ${CMAKE_SOURCE_DIR}/src/io/log_sink.c
  ${CMAKE_SOURCE_DIR}/src/io/snapshot_store.c
  ${CMAKE_SOURCE_DIR}/src/utility/perf_profile.c
  ${CMAKE_SOURCE_DIR}/src/utility/rk4_debug.c
  ${CMAKE_SOURCE_DIR}/src/utility/state_debug.c
  ${CMAKE_SOURCE_DIR}/src/fft/fft.c
)

include(ResolveVulkan)
nlo_resolve_vulkan(_nlo_vk_headers_available _nlo_vk_loader_available)

target_include_directories(test_core_state_alloc PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_BINARY_DIR}/src/generated
)
target_link_libraries(test_core_state_alloc PRIVATE nlo_fft_backend simd::simd Vulkan::Vulkan)
include(NLOSQLite)
nlo_configure_sqlite(test_core_state_alloc)
if(TARGET nlo_vk_shaders)
  add_dependencies(test_core_state_alloc nlo_vk_shaders)
endif()
if(WIN32)
  target_compile_definitions(test_core_state_alloc PRIVATE NLOLIB_EXPORTS)
  add_custom_command(TARGET test_core_state_alloc POST_BUILD
    COMMAND ${CMAKE_COMMAND}
            -DNLO_RUNTIME_SOURCE="$<TARGET_FILE:test_core_state_alloc>"
            -DNLO_RUNTIME_DEST="$<TARGET_FILE_DIR:test_core_state_alloc>"
            -DNLO_RUNTIME_HINTS="${NLO_SQLITE_RUNTIME_HINTS}"
            -P "${CMAKE_SOURCE_DIR}/cmake/copy_runtime_deps.cmake"
  )
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

add_executable(test_core_log_sink
  ${CMAKE_CURRENT_LIST_DIR}/test_log_sink.c
  ${CMAKE_SOURCE_DIR}/src/io/log_format.c
  ${CMAKE_SOURCE_DIR}/src/io/log_sink.c
)

target_include_directories(test_core_log_sink PRIVATE ${CMAKE_SOURCE_DIR}/src)
if(UNIX)
  target_link_libraries(test_core_log_sink PRIVATE m)
endif()

add_test(NAME test_core_log_sink COMMAND test_core_log_sink)
