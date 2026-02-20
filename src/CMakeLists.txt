# Main library CMakeLists.txt

set(NLOLIB_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/nlolib.c
  ${CMAKE_CURRENT_LIST_DIR}/core/state.c
  ${CMAKE_CURRENT_LIST_DIR}/core/init.c
  ${CMAKE_CURRENT_LIST_DIR}/physics/operators.c
  ${CMAKE_CURRENT_LIST_DIR}/physics/operator_program.c
  ${CMAKE_CURRENT_LIST_DIR}/numerics/math_ops.c
  ${CMAKE_CURRENT_LIST_DIR}/numerics/vector_ops.c
  ${CMAKE_CURRENT_LIST_DIR}/numerics/rk4_kernel.c
  ${CMAKE_CURRENT_LIST_DIR}/io/snapshot_store.c
  ${CMAKE_CURRENT_LIST_DIR}/utility/perf_profile.c
  ${CMAKE_CURRENT_LIST_DIR}/utility/rk4_debug.c
  ${CMAKE_CURRENT_LIST_DIR}/utility/state_debug.c
  ${CMAKE_CURRENT_LIST_DIR}/backend/vector_backend.c
  ${CMAKE_CURRENT_LIST_DIR}/backend/vk_auto_context.c
  ${CMAKE_CURRENT_LIST_DIR}/fft/fft.c
  ${CMAKE_CURRENT_LIST_DIR}/numerics/vk_vector_ops.c
)

add_library(nlolib SHARED ${NLOLIB_SOURCES})
set_target_properties(nlolib PROPERTIES OUTPUT_NAME nlolib)

target_include_directories(nlolib PUBLIC ${CMAKE_CURRENT_LIST_DIR})

if(WIN32)
  target_compile_definitions(nlolib PRIVATE NLOLIB_EXPORTS)
endif()

if(NLO_ENABLE_RK4_DEBUG_DIAGNOSTICS)
  target_compile_definitions(nlolib PRIVATE
    $<$<CONFIG:Debug>:NLO_ENABLE_RK4_DEBUG_DIAGNOSTICS=1>
  )
endif()

add_subdirectory(fft)
target_link_libraries(nlolib PUBLIC nlo_fft_backend simd::simd)
include(NLOSQLite)
nlo_configure_sqlite(nlolib)

if(UNIX)
  target_link_libraries(nlolib PUBLIC m)
endif()

include(NLOVulkanBackend)
nlo_configure_vulkan_backend(
  nlolib
  "${CMAKE_CURRENT_LIST_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}"
)


