# Main library CMakeLists.txt

set(NLOLIB_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/nlolib.c
  ${CMAKE_CURRENT_LIST_DIR}/core/state.c
  ${CMAKE_CURRENT_LIST_DIR}/core/init.c
  ${CMAKE_CURRENT_LIST_DIR}/physics/operators.c
  ${CMAKE_CURRENT_LIST_DIR}/numerics/math_ops.c
  ${CMAKE_CURRENT_LIST_DIR}/numerics/vector_ops.c
  ${CMAKE_CURRENT_LIST_DIR}/backend/vector_backend.c
  ${CMAKE_CURRENT_LIST_DIR}/fft/fft.c
)

if(NLO_VECTOR_BACKEND_VULKAN)
  list(APPEND NLOLIB_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/numerics/vk_vector_ops.c
  )
endif()

add_library(nlolib SHARED ${NLOLIB_SOURCES})
set_target_properties(nlolib PROPERTIES OUTPUT_NAME nlolib)

target_include_directories(nlolib PUBLIC ${CMAKE_CURRENT_LIST_DIR})

if(WIN32)
  target_compile_definitions(nlolib PRIVATE NLOLIB_EXPORTS)
endif()

add_subdirectory(fft)
target_link_libraries(nlolib PUBLIC nlo_fft_backend simd::simd)

if(UNIX)
  target_link_libraries(nlolib PUBLIC m)
endif()

if(NLO_VECTOR_BACKEND_VULKAN)
  include(ResolveVulkan)
  nlo_resolve_vulkan(_nlo_vk_headers_available _nlo_vk_loader_available)
  if(NOT _nlo_vk_headers_available)
    message(FATAL_ERROR
      "Vulkan headers were not found and could not be fetched. "
      "Set NLO_VULKAN_FETCH_HEADERS=ON or provide VULKAN_SDK/include.")
  endif()
  if(NOT _nlo_vk_loader_available)
    message(FATAL_ERROR
      "Vulkan loader library was not found. "
      "Install Vulkan loader/SDK (e.g. libvulkan-dev on Linux or LunarG Vulkan SDK on Windows).")
  endif()

  find_program(NLO_GLSLANG_VALIDATOR
    NAMES glslangValidator glslangValidator.exe
    HINTS
      ENV VULKAN_SDK
    PATH_SUFFIXES
      Bin
      bin
  )
  if(NOT NLO_GLSLANG_VALIDATOR)
    message(FATAL_ERROR
      "glslangValidator was not found. Install Vulkan SDK and ensure "
      "its Bin directory is on PATH (or VULKAN_SDK is set).")
  endif()

  set(NLO_VK_KERNEL_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/backend/vulkan/kernels")
  set(NLO_VK_KERNEL_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/backend/vulkan/kernels")
  file(MAKE_DIRECTORY "${NLO_VK_KERNEL_BINARY_DIR}")

  set(_nlo_vk_kernel_names
    nlo_real_fill
    nlo_real_mul_inplace
    nlo_complex_fill
    nlo_complex_scalar_mul_inplace
    nlo_complex_add_inplace
    nlo_complex_mul_inplace
    nlo_complex_magnitude_squared
  )

  set(_nlo_vk_spv_outputs "")
  foreach(_nlo_vk_kernel IN LISTS _nlo_vk_kernel_names)
    set(_nlo_vk_kernel_src "${NLO_VK_KERNEL_SOURCE_DIR}/${_nlo_vk_kernel}.comp")
    set(_nlo_vk_kernel_spv "${NLO_VK_KERNEL_BINARY_DIR}/${_nlo_vk_kernel}.spv")
    add_custom_command(
      OUTPUT "${_nlo_vk_kernel_spv}"
      COMMAND "${NLO_GLSLANG_VALIDATOR}"
        -V
        --target-env vulkan1.2
        -I"${NLO_VK_KERNEL_SOURCE_DIR}"
        -o "${_nlo_vk_kernel_spv}"
        "${_nlo_vk_kernel_src}"
      DEPENDS
        "${_nlo_vk_kernel_src}"
        "${NLO_VK_KERNEL_SOURCE_DIR}/nlo_complex_device.glslinc"
      COMMENT "Compiling Vulkan compute shader ${_nlo_vk_kernel}.comp"
      VERBATIM
    )
    list(APPEND _nlo_vk_spv_outputs "${_nlo_vk_kernel_spv}")
  endforeach()

  add_custom_target(nlo_vk_shaders DEPENDS ${_nlo_vk_spv_outputs})
  add_dependencies(nlolib nlo_vk_shaders)

  set(NLO_VK_SHADER_REAL_FILL_PATH "${NLO_VK_KERNEL_BINARY_DIR}/nlo_real_fill.spv")
  set(NLO_VK_SHADER_REAL_MUL_INPLACE_PATH "${NLO_VK_KERNEL_BINARY_DIR}/nlo_real_mul_inplace.spv")
  set(NLO_VK_SHADER_COMPLEX_FILL_PATH "${NLO_VK_KERNEL_BINARY_DIR}/nlo_complex_fill.spv")
  set(NLO_VK_SHADER_COMPLEX_SCALAR_MUL_INPLACE_PATH "${NLO_VK_KERNEL_BINARY_DIR}/nlo_complex_scalar_mul_inplace.spv")
  set(NLO_VK_SHADER_COMPLEX_ADD_INPLACE_PATH "${NLO_VK_KERNEL_BINARY_DIR}/nlo_complex_add_inplace.spv")
  set(NLO_VK_SHADER_COMPLEX_MUL_INPLACE_PATH "${NLO_VK_KERNEL_BINARY_DIR}/nlo_complex_mul_inplace.spv")
  set(NLO_VK_SHADER_COMPLEX_MAGNITUDE_SQUARED_PATH "${NLO_VK_KERNEL_BINARY_DIR}/nlo_complex_magnitude_squared.spv")

  configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/backend/vulkan/nlo_vk_shader_paths.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/nlo_vk_shader_paths.h"
    @ONLY
  )

  target_include_directories(nlolib PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
  target_compile_definitions(nlolib PUBLIC NLO_ENABLE_VECTOR_BACKEND_VULKAN=1)
  target_link_libraries(nlolib PUBLIC Vulkan::Headers Vulkan::Vulkan)
endif()

if(WIN32 AND NLO_FFT_BACKEND_FFTW AND FFTW3_LIBRARY)
  get_filename_component(_FFTW3_DIR "${FFTW3_LIBRARY}" DIRECTORY)
  get_filename_component(_FFTW3_NAME "${FFTW3_LIBRARY}" NAME_WE)
  set(_FFTW3_DLL "${_FFTW3_DIR}/${_FFTW3_NAME}.dll")

  if(EXISTS "${_FFTW3_DLL}")
    add_custom_command(TARGET nlolib POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${_FFTW3_DLL}"
      "$<TARGET_FILE_DIR:nlolib>"
    )
  endif()
endif()

