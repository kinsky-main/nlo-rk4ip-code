add_library(nlo_fft_backend INTERFACE)

if(NOT NLO_ENABLE_FFTW AND NOT NLO_ENABLE_VKFFT)
  message(FATAL_ERROR "No FFT implementations enabled. Set NLO_ENABLE_FFTW or NLO_ENABLE_VKFFT.")
endif()

if(NLO_ENABLE_FFTW)
  find_package(FFTW3 REQUIRED)
  target_link_libraries(nlo_fft_backend INTERFACE FFTW3::fftw3)
  target_compile_definitions(nlo_fft_backend INTERFACE NLO_ENABLE_FFTW_BACKEND=1)
endif()

if(NLO_ENABLE_VKFFT)
  if(NOT NLO_VECTOR_BACKEND_VULKAN)
    message(FATAL_ERROR
      "NLO_ENABLE_VKFFT=ON requires NLO_VECTOR_BACKEND_VULKAN=ON.")
  endif()

  include(FetchContent)
  find_package(Vulkan QUIET COMPONENTS glslang)
  if(NOT TARGET Vulkan::glslang)
    message(FATAL_ERROR
      "NLO_ENABLE_VKFFT=ON requested, but Vulkan::glslang target is unavailable.")
  endif()

  FetchContent_Declare(
    vkfft
    URL https://github.com/DTolm/VkFFT/archive/refs/heads/master.zip
  )
  FetchContent_MakeAvailable(vkfft)

  target_include_directories(nlo_fft_backend INTERFACE ${vkfft_SOURCE_DIR})
  target_compile_definitions(nlo_fft_backend INTERFACE NLO_ENABLE_VKFFT_BACKEND=1 VKFFT_BACKEND=0)
  target_link_libraries(nlo_fft_backend INTERFACE Vulkan::glslang)

  find_path(NLO_GLSLANG_C_INCLUDE_DIR
    NAMES glslang_c_interface.h
    HINTS
      ENV VULKAN_SDK
    PATH_SUFFIXES
      Include/glslang/Include
      glslang/Include
  )
  if(NLO_GLSLANG_C_INCLUDE_DIR)
    target_include_directories(nlo_fft_backend INTERFACE ${NLO_GLSLANG_C_INCLUDE_DIR})
  endif()

  if(WIN32 AND DEFINED ENV{VULKAN_SDK})
    target_link_directories(nlo_fft_backend INTERFACE "$ENV{VULKAN_SDK}/Lib")
    target_link_libraries(nlo_fft_backend INTERFACE
      $<$<CONFIG:Debug>:MachineIndependentd;OSDependentd;GenericCodeGend;SPIRVd;SPIRV-Toolsd;SPIRV-Tools-optd;glslang-default-resource-limitsd>
      $<$<NOT:$<CONFIG:Debug>>:MachineIndependent;OSDependent;GenericCodeGen;SPIRV;SPIRV-Tools;SPIRV-Tools-opt;glslang-default-resource-limits>
    )
  endif()
endif()
