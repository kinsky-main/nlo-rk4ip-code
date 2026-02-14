add_library(nlo_fft_backend INTERFACE)

option(NLO_ENABLE_VKFFT "Enable VkFFT integration for Vulkan FFT execution" ON)

if(NLO_FFT_BACKEND_FFTW AND NLO_FFT_BACKEND_CUFFT)
  message(FATAL_ERROR "Only one FFT backend can be enabled at a time.")
endif()

if(NLO_FFT_BACKEND_FFTW)
  find_package(FFTW3 REQUIRED)
  target_link_libraries(nlo_fft_backend INTERFACE FFTW3::fftw3)
  target_compile_definitions(nlo_fft_backend INTERFACE NLO_FFT_BACKEND_FFTW)
elseif(NLO_FFT_BACKEND_CUFFT)
  target_compile_definitions(nlo_fft_backend INTERFACE NLO_FFT_BACKEND_CUFFT)
else()
  message(FATAL_ERROR "No FFT backend enabled. Set NLO_FFT_BACKEND_FFTW or NLO_FFT_BACKEND_CUFFT.")
endif()

if(NLO_VECTOR_BACKEND_VULKAN AND NLO_ENABLE_VKFFT)
  include(FetchContent)
  find_package(Vulkan QUIET COMPONENTS glslang)

  if(TARGET Vulkan::glslang)
    FetchContent_Declare(
      vkfft
      URL https://github.com/DTolm/VkFFT/archive/refs/heads/master.zip
    )
    FetchContent_MakeAvailable(vkfft)

    target_include_directories(nlo_fft_backend INTERFACE ${vkfft_SOURCE_DIR})
    target_compile_definitions(nlo_fft_backend INTERFACE NLO_ENABLE_VKFFT=1 VKFFT_BACKEND=0)
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
  else()
    message(WARNING
      "NLO_ENABLE_VKFFT=ON requested, but Vulkan::glslang target is unavailable. "
      "VkFFT integration is disabled for this build.")
  endif()
endif()
