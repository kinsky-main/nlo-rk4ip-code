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
  ${CMAKE_SOURCE_DIR}/src/io/log_format.c
  ${CMAKE_SOURCE_DIR}/src/io/log_sink.c
)
if(NLO_ENABLE_CUDA_BACKEND)
  target_sources(test_fft PRIVATE
    ${CMAKE_SOURCE_DIR}/src/backend/cuda_backend.cu
  )
else()
  target_sources(test_fft PRIVATE
    ${CMAKE_SOURCE_DIR}/src/backend/cuda_backend_stubs.c
  )
endif()
if(NLO_ENABLE_VULKAN_BACKEND)
  target_sources(test_fft PRIVATE
    ${CMAKE_SOURCE_DIR}/src/backend/vk_auto_context.c
    ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_resources.c
    ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_phase.c
    ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_dispatch.c
    ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_transfer.c
    ${CMAKE_SOURCE_DIR}/src/numerics/vk_backend_ops.c
    ${CMAKE_SOURCE_DIR}/src/numerics/vk_operator_jit.c
  )
else()
  target_sources(test_fft PRIVATE
    ${CMAKE_SOURCE_DIR}/src/backend/vulkan_stubs.c
  )
endif()

include(ResolveVulkan)
nlo_resolve_vulkan(_nlo_vk_headers_available _nlo_vk_loader_available)
include(NLOCblas)
nlo_configure_cblas(_nlo_cblas_target _nlo_cblas_include_dirs)

target_include_directories(test_fft PRIVATE
  ${CMAKE_SOURCE_DIR}/src
)
if(NLO_ENABLE_VULKAN_BACKEND)
  target_include_directories(test_fft PRIVATE ${CMAKE_BINARY_DIR}/src/generated)
  find_path(NLO_TEST_FFT_GLSLANG_C_INCLUDE_DIR
    NAMES glslang_c_interface.h
    HINTS
      ENV VULKAN_SDK
    PATH_SUFFIXES
      Include/glslang/Include
      glslang/Include
  )
  if(NLO_TEST_FFT_GLSLANG_C_INCLUDE_DIR)
    target_include_directories(test_fft PRIVATE ${NLO_TEST_FFT_GLSLANG_C_INCLUDE_DIR})
  endif()
  find_path(NLO_TEST_FFT_GLSLANG_RESOURCE_INCLUDE_DIR
    NAMES resource_limits_c.h
    HINTS
      ENV VULKAN_SDK
    PATH_SUFFIXES
      Include/glslang/Public
      glslang/Public
  )
  if(NLO_TEST_FFT_GLSLANG_RESOURCE_INCLUDE_DIR)
    target_include_directories(test_fft PRIVATE ${NLO_TEST_FFT_GLSLANG_RESOURCE_INCLUDE_DIR})
  endif()
endif()
target_link_libraries(test_fft PRIVATE nlo_fft_backend ${_nlo_cblas_target})
target_compile_definitions(test_fft PRIVATE
  NLO_ENABLE_VULKAN_BACKEND=$<IF:$<BOOL:${NLO_ENABLE_VULKAN_BACKEND}>,1,0>
  NLO_ENABLE_VKFFT=$<IF:$<BOOL:${NLO_ENABLE_VKFFT}>,1,0>
  NLO_ENABLE_CUDA_BACKEND=$<IF:$<BOOL:${NLO_ENABLE_CUDA_BACKEND}>,1,0>
  NLO_ENABLE_CUFFT=$<IF:$<BOOL:${NLO_ENABLE_CUFFT}>,1,0>
)
if(NLO_ENABLE_CUDA_BACKEND)
  target_link_libraries(test_fft PRIVATE CUDA::cudart)
endif()
if(NLO_ENABLE_VULKAN_BACKEND)
  find_package(Vulkan QUIET COMPONENTS glslang)
  target_link_libraries(test_fft PRIVATE Vulkan::Vulkan)
  if(TARGET Vulkan::glslang)
    target_link_libraries(test_fft PRIVATE Vulkan::glslang)
  endif()
  if(WIN32 AND DEFINED ENV{VULKAN_SDK})
    target_link_directories(test_fft PRIVATE "$ENV{VULKAN_SDK}/Lib")
    target_link_libraries(test_fft PRIVATE
      $<$<CONFIG:Debug>:MachineIndependentd;OSDependentd;GenericCodeGend;SPIRVd;SPIRV-Toolsd;SPIRV-Tools-optd;glslang-default-resource-limitsd>
      $<$<NOT:$<CONFIG:Debug>>:MachineIndependent;OSDependent;GenericCodeGen;SPIRV;SPIRV-Tools;SPIRV-Tools-opt;glslang-default-resource-limits>
    )
    if(MSVC)
      target_link_options(test_fft PRIVATE /IGNORE:4099)
    endif()
  endif()
endif()
if(TARGET nlo_vk_shaders)
  add_dependencies(test_fft nlo_vk_shaders)
endif()
if(UNIX)
  target_link_libraries(test_fft PRIVATE m)
endif()
if(WIN32)
  add_custom_command(TARGET test_fft POST_BUILD
    COMMAND ${CMAKE_COMMAND}
            -DNLO_RUNTIME_SOURCE="$<TARGET_FILE:test_fft>"
            -DNLO_RUNTIME_DEST="$<TARGET_FILE_DIR:test_fft>"
            -DNLO_RUNTIME_HINTS="${NLO_CBLAS_RUNTIME_HINTS}"
            -P "${CMAKE_SOURCE_DIR}/cmake/copy_runtime_deps.cmake"
  )
endif()

add_test(NAME test_fft COMMAND test_fft)

set(_NLOLIB_TEST_PATH "$ENV{PATH}")
string(REPLACE ";" "\\;" _NLOLIB_TEST_PATH "${_NLOLIB_TEST_PATH}")

set_tests_properties(test_fft PROPERTIES
  ENVIRONMENT "PATH=${_NLOLIB_TEST_PATH}"
)
