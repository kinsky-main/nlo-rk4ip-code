# Benchmarks CMakeLists.txt

add_executable(bench_solver_backend
  ${CMAKE_CURRENT_LIST_DIR}/bench_solver_backend.c
  ${CMAKE_CURRENT_LIST_DIR}/bench_tensor_scaling.c
  ${CMAKE_CURRENT_LIST_DIR}/tensor_scaling_plan.c
  ${CMAKE_SOURCE_DIR}/src/core/init.c
  ${CMAKE_SOURCE_DIR}/src/core/init_defaults.c
  ${CMAKE_SOURCE_DIR}/src/core/init_limits.c
  ${CMAKE_SOURCE_DIR}/src/core/init_state_builder.c
  ${CMAKE_SOURCE_DIR}/src/core/state_lifecycle.c
  ${CMAKE_SOURCE_DIR}/src/core/state_snapshots.c
  ${CMAKE_SOURCE_DIR}/src/io/log_format.c
  ${CMAKE_SOURCE_DIR}/src/io/log_sink.c
  ${CMAKE_SOURCE_DIR}/src/io/snapshot_store.c
  ${CMAKE_SOURCE_DIR}/src/backend/vector_backend.c
  ${CMAKE_SOURCE_DIR}/src/backend/vk_auto_context.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vector_ops.c
  ${CMAKE_SOURCE_DIR}/src/numerics/math_ops.c
  ${CMAKE_SOURCE_DIR}/src/physics/operators.c
  ${CMAKE_SOURCE_DIR}/src/physics/operator_program_compile.c
  ${CMAKE_SOURCE_DIR}/src/physics/operator_program_execute.c
  ${CMAKE_SOURCE_DIR}/src/numerics/rk4_kernel.c
  ${CMAKE_SOURCE_DIR}/src/utility/perf_profile.c
  ${CMAKE_SOURCE_DIR}/src/utility/rk4_debug.c
  ${CMAKE_SOURCE_DIR}/src/utility/state_debug.c
  ${CMAKE_SOURCE_DIR}/src/fft/fft.c
  ${CMAKE_CURRENT_LIST_DIR}/vulkan_bench_context.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_resources.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_phase.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_dispatch.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_transfer.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_ops.c
)
if(ENABLE_VKFFT)
  set_target_properties(bench_solver_backend PROPERTIES LINKER_LANGUAGE CXX)
endif()

include(ResolveVulkan)
resolve_vulkan(vk_headers_available vk_loader_available)
include(NLOCblas)
configure_cblas(cblas_target cblas_include_dirs)

target_include_directories(bench_solver_backend PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(bench_solver_backend PRIVATE ${cblas_target})
include(NLOSQLite)
configure_sqlite(bench_solver_backend)

include(NLOFftw)
configure_fftw(fftw_target fftw_include_dirs)
target_link_libraries(bench_solver_backend PRIVATE ${fftw_target})
if(fftw_include_dirs)
  target_include_directories(bench_solver_backend PRIVATE ${fftw_include_dirs})
endif()

set(VKFFT_INCLUDE_DIR "${CMAKE_BINARY_DIR}/_deps/vkfft-src")
if(EXISTS "${VKFFT_INCLUDE_DIR}/vkFFT/vkFFT.h")
  target_include_directories(bench_solver_backend PRIVATE
    "${VKFFT_INCLUDE_DIR}"
    "${VKFFT_INCLUDE_DIR}/vkFFT"
  )
  target_compile_definitions(bench_solver_backend PRIVATE VKFFT_BACKEND=0)
else()
  message(FATAL_ERROR
    "VkFFT headers were not found at ${VKFFT_INCLUDE_DIR}.")
endif()

find_path(GLSLANG_C_INCLUDE_DIR
  NAMES glslang_c_interface.h
  HINTS
    ENV VULKAN_SDK
  PATH_SUFFIXES
    Include/glslang/Include
    glslang/Include
)
if(GLSLANG_C_INCLUDE_DIR)
  target_include_directories(bench_solver_backend PRIVATE ${GLSLANG_C_INCLUDE_DIR})
endif()

include(ResolveGlslang)
resolve_glslang_for_vkfft(glslang_target glslang_found)
target_link_libraries(bench_solver_backend PRIVATE ${glslang_target})

if(WIN32 AND MSVC)
  target_link_options(bench_solver_backend PRIVATE /IGNORE:4099)
endif()

if(WIN32)
  target_compile_definitions(bench_solver_backend PRIVATE NLOLIB_EXPORTS)
  set(runtime_hints "${SQLITE_RUNTIME_HINTS}")
  if(CBLAS_RUNTIME_HINTS)
    list(APPEND runtime_hints "${CBLAS_RUNTIME_HINTS}")
  endif()
  add_custom_command(TARGET bench_solver_backend POST_BUILD
    COMMAND ${CMAKE_COMMAND}
            -DNLO_RUNTIME_SOURCE="$<TARGET_FILE:bench_solver_backend>"
            -DNLO_RUNTIME_DEST="$<TARGET_FILE_DIR:bench_solver_backend>"
            -DNLO_RUNTIME_HINTS="${runtime_hints}"
            -P "${CMAKE_SOURCE_DIR}/cmake/copy_runtime_deps.cmake"
  )
endif()

if(UNIX)
  target_link_libraries(bench_solver_backend PRIVATE m)
endif()

target_include_directories(bench_solver_backend PRIVATE ${CMAKE_BINARY_DIR}/src/generated)
target_link_libraries(bench_solver_backend PRIVATE Vulkan::Vulkan)
if(TARGET vk_shaders)
  add_dependencies(bench_solver_backend vk_shaders)
endif()

if(BUILD_TESTING)
  add_test(NAME test_bench_solver_backend_tensor_dry_run
    COMMAND ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:bench_solver_backend>
            -DTEST_SOURCE_DIR=${CMAKE_SOURCE_DIR}
            -P ${CMAKE_SOURCE_DIR}/tests/core/check_tensor_bench_dry_run.cmake
  )
endif()
