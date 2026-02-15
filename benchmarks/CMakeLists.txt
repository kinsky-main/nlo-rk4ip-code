# Benchmarks CMakeLists.txt

add_executable(bench_solver_backend
  ${CMAKE_CURRENT_LIST_DIR}/bench_solver_backend.c
  ${CMAKE_SOURCE_DIR}/src/core/state.c
  ${CMAKE_SOURCE_DIR}/src/core/init.c
  ${CMAKE_SOURCE_DIR}/src/backend/vector_backend.c
  ${CMAKE_SOURCE_DIR}/src/numerics/vector_ops.c
  ${CMAKE_SOURCE_DIR}/src/numerics/math_ops.c
  ${CMAKE_SOURCE_DIR}/src/physics/operators.c
  ${CMAKE_SOURCE_DIR}/src/numerics/rk4_kernel.c
  ${CMAKE_SOURCE_DIR}/src/utility/rk4_debug.c
  ${CMAKE_SOURCE_DIR}/src/fft/fft.c
)

if(NLO_VECTOR_BACKEND_VULKAN)
  include(ResolveVulkan)
  nlo_resolve_vulkan(_nlo_vk_headers_available _nlo_vk_loader_available)

  target_sources(bench_solver_backend PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/vulkan_bench_context.c
    ${CMAKE_SOURCE_DIR}/src/numerics/vk_vector_ops.c
  )
endif()

target_include_directories(bench_solver_backend PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(bench_solver_backend PRIVATE simd::simd)

if(NLO_ENABLE_FFTW)
  find_package(FFTW3 REQUIRED)
  target_link_libraries(bench_solver_backend PRIVATE FFTW3::fftw3)
  target_compile_definitions(bench_solver_backend PRIVATE NLO_ENABLE_FFTW_BACKEND=1)
endif()

if(NLO_ENABLE_VKFFT AND NLO_VECTOR_BACKEND_VULKAN)
  set(_NLO_VKFFT_INCLUDE_DIR "${CMAKE_BINARY_DIR}/_deps/vkfft-src")
  if(EXISTS "${_NLO_VKFFT_INCLUDE_DIR}/vkFFT/vkFFT.h")
    target_include_directories(bench_solver_backend PRIVATE "${_NLO_VKFFT_INCLUDE_DIR}")
    target_compile_definitions(bench_solver_backend PRIVATE NLO_ENABLE_VKFFT_BACKEND=1 VKFFT_BACKEND=0)
  else()
    message(FATAL_ERROR
      "NLO_ENABLE_VKFFT is ON but VkFFT headers were not found at ${_NLO_VKFFT_INCLUDE_DIR}.")
  endif()

  find_path(_NLO_GLSLANG_C_INCLUDE_DIR
    NAMES glslang_c_interface.h
    HINTS
      ENV VULKAN_SDK
    PATH_SUFFIXES
      Include/glslang/Include
      glslang/Include
  )
  if(_NLO_GLSLANG_C_INCLUDE_DIR)
    target_include_directories(bench_solver_backend PRIVATE ${_NLO_GLSLANG_C_INCLUDE_DIR})
  endif()

  find_package(Vulkan QUIET COMPONENTS glslang)
  if(TARGET Vulkan::glslang)
    target_link_libraries(bench_solver_backend PRIVATE Vulkan::glslang)
  endif()

  if(WIN32 AND DEFINED ENV{VULKAN_SDK})
    target_link_directories(bench_solver_backend PRIVATE "$ENV{VULKAN_SDK}/Lib")
    target_link_libraries(bench_solver_backend PRIVATE
      $<$<CONFIG:Debug>:MachineIndependentd;OSDependentd;GenericCodeGend;SPIRVd;SPIRV-Toolsd;SPIRV-Tools-optd;glslang-default-resource-limitsd>
      $<$<NOT:$<CONFIG:Debug>>:MachineIndependent;OSDependent;GenericCodeGen;SPIRV;SPIRV-Tools;SPIRV-Tools-opt;glslang-default-resource-limits>
    )
  endif()
endif()

if(WIN32)
  target_compile_definitions(bench_solver_backend PRIVATE NLOLIB_EXPORTS)
endif()

if(UNIX)
  target_link_libraries(bench_solver_backend PRIVATE m)
endif()

if(NLO_VECTOR_BACKEND_VULKAN)
  target_include_directories(bench_solver_backend PRIVATE ${CMAKE_BINARY_DIR}/src/generated)
  target_compile_definitions(bench_solver_backend PRIVATE NLO_ENABLE_VECTOR_BACKEND_VULKAN=1)
  target_link_libraries(bench_solver_backend PRIVATE Vulkan::Vulkan)
  if(TARGET nlo_vk_shaders)
    add_dependencies(bench_solver_backend nlo_vk_shaders)
  endif()
endif()
