add_executable(test_core_state_alloc
  ${CMAKE_CURRENT_LIST_DIR}/test_state_alloc.c
  ${CMAKE_SOURCE_DIR}/src/core/init.c
  ${CMAKE_SOURCE_DIR}/src/core/init_defaults.c
  ${CMAKE_SOURCE_DIR}/src/core/init_limits.c
  ${CMAKE_SOURCE_DIR}/src/core/init_state_builder.c
  ${CMAKE_SOURCE_DIR}/src/core/state_lifecycle.c
  ${CMAKE_SOURCE_DIR}/src/core/state_snapshots.c
  ${CMAKE_SOURCE_DIR}/src/backend/vector_backend.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vector_ops.c
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
if(NLO_ENABLE_CUDA_BACKEND)
  target_sources(test_core_state_alloc PRIVATE
    ${CMAKE_SOURCE_DIR}/src/backend/cuda_backend.cu
  )
else()
  target_sources(test_core_state_alloc PRIVATE
    ${CMAKE_SOURCE_DIR}/src/backend/cuda_backend_stubs.c
  )
endif()
if(NLO_ENABLE_VULKAN_BACKEND)
  target_sources(test_core_state_alloc PRIVATE
    ${CMAKE_SOURCE_DIR}/src/backend/vk_auto_context.c
    ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_resources.c
    ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_phase.c
    ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_dispatch.c
    ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_transfer.c
    ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_ops.c
    ${CMAKE_SOURCE_DIR}/src/numerics/vk_operator_jit.c
  )
else()
  target_sources(test_core_state_alloc PRIVATE
    ${CMAKE_SOURCE_DIR}/src/backend/vulkan_stubs.c
  )
endif()

include(ResolveVulkan)
nlo_resolve_vulkan(_nlo_vk_headers_available _nlo_vk_loader_available)
if(NLO_ENABLE_VULKAN_BACKEND)
  find_package(Vulkan QUIET COMPONENTS glslang)
endif()
include(NLOCblas)
nlo_configure_cblas(_nlo_cblas_target _nlo_cblas_include_dirs)

target_include_directories(test_core_state_alloc PRIVATE
  ${CMAKE_SOURCE_DIR}/src
)
if(NLO_ENABLE_VULKAN_BACKEND)
  target_include_directories(test_core_state_alloc PRIVATE ${CMAKE_BINARY_DIR}/src/generated)
  find_path(NLO_TEST_GLSLANG_C_INCLUDE_DIR
    NAMES glslang_c_interface.h
    HINTS
      ENV VULKAN_SDK
    PATH_SUFFIXES
      Include/glslang/Include
      glslang/Include
  )
  if(NLO_TEST_GLSLANG_C_INCLUDE_DIR)
    target_include_directories(test_core_state_alloc PRIVATE ${NLO_TEST_GLSLANG_C_INCLUDE_DIR})
  endif()
  find_path(NLO_TEST_GLSLANG_RESOURCE_INCLUDE_DIR
    NAMES resource_limits_c.h
    HINTS
      ENV VULKAN_SDK
    PATH_SUFFIXES
      Include/glslang/Public
      glslang/Public
  )
  if(NLO_TEST_GLSLANG_RESOURCE_INCLUDE_DIR)
    target_include_directories(test_core_state_alloc PRIVATE ${NLO_TEST_GLSLANG_RESOURCE_INCLUDE_DIR})
  endif()
endif()
target_link_libraries(test_core_state_alloc PRIVATE nlo_fft_backend ${_nlo_cblas_target})
target_compile_definitions(test_core_state_alloc PRIVATE
  NLO_ENABLE_VULKAN_BACKEND=$<IF:$<BOOL:${NLO_ENABLE_VULKAN_BACKEND}>,1,0>
  NLO_ENABLE_VKFFT=$<IF:$<BOOL:${NLO_ENABLE_VKFFT}>,1,0>
  NLO_ENABLE_CUDA_BACKEND=$<IF:$<BOOL:${NLO_ENABLE_CUDA_BACKEND}>,1,0>
  NLO_ENABLE_CUFFT=$<IF:$<BOOL:${NLO_ENABLE_CUFFT}>,1,0>
  NLO_ENABLE_RUNTIME_PROFILING=1
)
if(NLO_ENABLE_CUDA_BACKEND)
  target_link_libraries(test_core_state_alloc PRIVATE CUDA::cudart)
endif()
if(NLO_ENABLE_VULKAN_BACKEND)
  target_link_libraries(test_core_state_alloc PRIVATE Vulkan::Vulkan Vulkan::glslang)
  if(WIN32 AND DEFINED ENV{VULKAN_SDK})
    target_link_directories(test_core_state_alloc PRIVATE "$ENV{VULKAN_SDK}/Lib")
    target_link_libraries(test_core_state_alloc PRIVATE
      $<$<CONFIG:Debug>:MachineIndependentd;OSDependentd;GenericCodeGend;SPIRVd;SPIRV-Toolsd;SPIRV-Tools-optd;glslang-default-resource-limitsd>
      $<$<NOT:$<CONFIG:Debug>>:MachineIndependent;OSDependent;GenericCodeGen;SPIRV;SPIRV-Tools;SPIRV-Tools-opt;glslang-default-resource-limits>
    )
    if(MSVC)
      target_link_options(test_core_state_alloc PRIVATE /IGNORE:4099)
    endif()
  endif()
endif()
include(NLOSQLite)
nlo_configure_sqlite(test_core_state_alloc)
if(TARGET nlo_vk_shaders)
  add_dependencies(test_core_state_alloc nlo_vk_shaders)
endif()
if(WIN32)
  target_compile_definitions(test_core_state_alloc PRIVATE NLOLIB_EXPORTS)
  set(_nlo_runtime_hints "${NLO_SQLITE_RUNTIME_HINTS}")
  if(NLO_CBLAS_RUNTIME_HINTS)
    list(APPEND _nlo_runtime_hints "${NLO_CBLAS_RUNTIME_HINTS}")
  endif()
  add_custom_command(TARGET test_core_state_alloc POST_BUILD
    COMMAND ${CMAKE_COMMAND}
            -DNLO_RUNTIME_SOURCE="$<TARGET_FILE:test_core_state_alloc>"
            -DNLO_RUNTIME_DEST="$<TARGET_FILE_DIR:test_core_state_alloc>"
            -DNLO_RUNTIME_HINTS="${_nlo_runtime_hints}"
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
  ${CMAKE_SOURCE_DIR}/src/io/propagate_log.c
  ${CMAKE_SOURCE_DIR}/src/io/log_sink.c
)

target_include_directories(test_core_log_sink PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(test_core_log_sink PRIVATE
  NLO_ENABLE_VULKAN_BACKEND=0
  NLO_ENABLE_VKFFT=0
  NLO_ENABLE_CUDA_BACKEND=0
)
if(UNIX)
  target_link_libraries(test_core_log_sink PRIVATE m)
endif()

add_test(NAME test_core_log_sink COMMAND test_core_log_sink)

add_executable(test_core_perf_profile
  ${CMAKE_CURRENT_LIST_DIR}/test_perf_profile.c
  ${CMAKE_SOURCE_DIR}/src/utility/perf_profile.c
  ${CMAKE_SOURCE_DIR}/src/backend/vulkan_stubs.c
)
target_include_directories(test_core_perf_profile PRIVATE
  ${CMAKE_SOURCE_DIR}/src
)
target_compile_definitions(test_core_perf_profile PRIVATE
  NLO_ENABLE_VULKAN_BACKEND=0
  NLO_ENABLE_VKFFT=0
  NLO_ENABLE_CUDA_BACKEND=0
  NLO_ENABLE_RUNTIME_PROFILING=1
)
if(UNIX)
  target_link_libraries(test_core_perf_profile PRIVATE m)
endif()
add_test(NAME test_core_perf_profile COMMAND test_core_perf_profile)

add_executable(test_tensor_bench_plan
  ${CMAKE_CURRENT_LIST_DIR}/test_tensor_bench_plan.c
  ${CMAKE_SOURCE_DIR}/benchmarks/tensor_scaling_plan.c
)

target_include_directories(test_tensor_bench_plan PRIVATE
  ${CMAKE_SOURCE_DIR}/benchmarks
)

if(UNIX)
  target_link_libraries(test_tensor_bench_plan PRIVATE m)
endif()

add_test(NAME test_tensor_bench_plan COMMAND test_tensor_bench_plan)

find_program(NLO_TEST_BASH_EXECUTABLE bash)
if(NLO_TEST_BASH_EXECUTABLE)
  add_test(NAME test_docker_validate_cpu_dry_run
    COMMAND "${NLO_TEST_BASH_EXECUTABLE}" -lc "export NLO_DOCKER_VALIDATE_DRY_RUN=1; ./scripts/docker_validate_cpu.sh")
  set_tests_properties(test_docker_validate_cpu_dry_run PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  )

  add_test(NAME test_docker_validate_vulkan_dry_run
    COMMAND "${NLO_TEST_BASH_EXECUTABLE}" -lc "export NLO_DOCKER_VALIDATE_DRY_RUN=1; ./scripts/docker_validate_vulkan.sh")
  set_tests_properties(test_docker_validate_vulkan_dry_run PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  )

  add_test(NAME test_docker_validate_cuda_dry_run
    COMMAND "${NLO_TEST_BASH_EXECUTABLE}" -lc "export NLO_DOCKER_VALIDATE_DRY_RUN=1; ./scripts/docker_validate_cuda.sh")
  set_tests_properties(test_docker_validate_cuda_dry_run PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  )

  add_test(NAME test_docker_validate_cuda_multigpu_dry_run
    COMMAND "${NLO_TEST_BASH_EXECUTABLE}" -lc "export NLO_DOCKER_VALIDATE_DRY_RUN=1; ./scripts/docker_validate_cuda_multigpu.sh")
  set_tests_properties(test_docker_validate_cuda_multigpu_dry_run PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  )
endif()
