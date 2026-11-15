# Main library CMakeLists.txt

set(NLOLIB_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/nlolib.c
  ${CMAKE_CURRENT_LIST_DIR}/core/init.c
  ${CMAKE_CURRENT_LIST_DIR}/core/init_defaults.c
  ${CMAKE_CURRENT_LIST_DIR}/core/init_limits.c
  ${CMAKE_CURRENT_LIST_DIR}/core/init_state_builder.c
  ${CMAKE_CURRENT_LIST_DIR}/core/state_lifecycle.c
  ${CMAKE_CURRENT_LIST_DIR}/core/state_snapshots.c
  ${CMAKE_CURRENT_LIST_DIR}/physics/operators.c
  ${CMAKE_CURRENT_LIST_DIR}/physics/operator_program_compile.c
  ${CMAKE_CURRENT_LIST_DIR}/physics/operator_program_execute.c
  ${CMAKE_CURRENT_LIST_DIR}/numerics/math_ops.c
  ${CMAKE_CURRENT_LIST_DIR}/numerics/vector_ops.c
  ${CMAKE_CURRENT_LIST_DIR}/numerics/rk4_kernel.c
  ${CMAKE_CURRENT_LIST_DIR}/io/log_format.c
  ${CMAKE_CURRENT_LIST_DIR}/io/propagate_log.c
  ${CMAKE_CURRENT_LIST_DIR}/io/log_sink.c
  ${CMAKE_CURRENT_LIST_DIR}/io/snapshot_store.c
  ${CMAKE_CURRENT_LIST_DIR}/io/run_store.c
  ${CMAKE_CURRENT_LIST_DIR}/io/sha256.c
  ${CMAKE_CURRENT_LIST_DIR}/utility/perf_profile.c
  ${CMAKE_CURRENT_LIST_DIR}/utility/rk4_debug.c
  ${CMAKE_CURRENT_LIST_DIR}/utility/state_debug.c
  ${CMAKE_CURRENT_LIST_DIR}/backend/vector_backend.c
  ${CMAKE_CURRENT_LIST_DIR}/fft/fft.c
)

if(ENABLE_VULKAN_BACKEND)
  list(APPEND NLOLIB_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/backend/vk_auto_context.c
    ${CMAKE_CURRENT_LIST_DIR}/numerics/vk_backend_resources.c
    ${CMAKE_CURRENT_LIST_DIR}/numerics/vk_backend_phase.c
    ${CMAKE_CURRENT_LIST_DIR}/numerics/vk_backend_dispatch.c
    ${CMAKE_CURRENT_LIST_DIR}/numerics/vk_backend_transfer.c
    ${CMAKE_CURRENT_LIST_DIR}/numerics/vk_backend_ops.c
    ${CMAKE_CURRENT_LIST_DIR}/numerics/vk_operator_jit.c
  )
else()
  list(APPEND NLOLIB_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/backend/vulkan_stubs.c
  )
endif()

add_library(nlolib SHARED ${NLOLIB_SOURCES})
set_target_properties(nlolib PROPERTIES OUTPUT_NAME nlolib)
if(ENABLE_VKFFT)
  set_target_properties(nlolib PROPERTIES LINKER_LANGUAGE CXX)
endif()

target_include_directories(nlolib PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# Revision recorded in the `git_commit` column of stored runs, refreshed on every build.
set(NLOLIB_GIT_REVISION_DIR "${CMAKE_BINARY_DIR}/generated_revision" CACHE INTERNAL "")
add_custom_target(nlolib_git_revision
  COMMAND ${CMAKE_COMMAND}
          -DOUTPUT=${NLOLIB_GIT_REVISION_DIR}/nlolib_git_revision.h
          -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
          -DVERSION=${PROJECT_VERSION}
          -P ${CMAKE_SOURCE_DIR}/cmake/write_git_revision.cmake
  BYPRODUCTS ${NLOLIB_GIT_REVISION_DIR}/nlolib_git_revision.h
  COMMENT "Recording the nlolib git revision"
)
add_dependencies(nlolib nlolib_git_revision)
target_include_directories(nlolib PRIVATE ${NLOLIB_GIT_REVISION_DIR})
target_compile_definitions(nlolib PRIVATE
  NLOLIB_HAVE_GIT_REVISION=1
  NLOLIB_VERSION_STRING="${PROJECT_VERSION}"
)

if(WIN32)
  target_compile_definitions(nlolib PRIVATE NLOLIB_EXPORTS)
endif()

if(ENABLE_RK4_DEBUG_DIAGNOSTICS)
  target_compile_definitions(nlolib PRIVATE
    $<$<CONFIG:Debug>:ENABLE_RK4_DEBUG_DIAGNOSTICS=1>
  )
endif()

add_subdirectory(fft)
target_link_libraries(nlolib PUBLIC fft_backend)
include(NLOSQLite)
configure_sqlite(nlolib)

if(UNIX)
  target_link_libraries(nlolib PUBLIC m)
endif()

# Embed the C++ runtime (pulled in by glslang/VkFFT) so hosts that preload an
# older libstdc++ -- notably MATLAB's sys/os/glnxa64 -- can still load nlolib.
# --exclude-libs keeps the embedded runtime symbols private to nlolib.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_options(nlolib PRIVATE
    -static-libstdc++
    -static-libgcc
    -Wl,--exclude-libs,ALL
  )
endif()

if(ENABLE_VULKAN_BACKEND)
  include(NLOVulkanBackend)
  configure_vulkan_backend(
    nlolib
    "${CMAKE_CURRENT_LIST_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}"
  )
endif()

if(WIN32)
  set(runtime_hints "${SQLITE_RUNTIME_HINTS}")
  add_custom_command(TARGET nlolib POST_BUILD
    COMMAND ${CMAKE_COMMAND}
            -DRUNTIME_SOURCE="$<TARGET_FILE:nlolib>"
            -DRUNTIME_DEST="$<TARGET_FILE_DIR:nlolib>"
            -DRUNTIME_HINTS="${runtime_hints}"
            -P "${CMAKE_SOURCE_DIR}/cmake/copy_runtime_deps.cmake"
  )
endif()

target_compile_definitions(nlolib PUBLIC
  ENABLE_VULKAN_BACKEND=$<IF:$<BOOL:${ENABLE_VULKAN_BACKEND}>,1,0>
  ENABLE_VKFFT=$<IF:$<BOOL:${ENABLE_VKFFT}>,1,0>
)


# Command-line merge of completed staging files, for the host that merges runs.
add_executable(nlolib_store_merge ${CMAKE_CURRENT_LIST_DIR}/tools/store_merge.c)
target_link_libraries(nlolib_store_merge PRIVATE nlolib)
if(WIN32)
  add_custom_command(TARGET nlolib_store_merge POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:nlolib> $<TARGET_FILE_DIR:nlolib_store_merge>
  )
endif()
