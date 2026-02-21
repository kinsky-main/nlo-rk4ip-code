add_library(nlo_fft_backend INTERFACE)

include(NLOFftw)
nlo_configure_fftw(_nlo_fftw_target _nlo_fftw_include_dirs)
target_link_libraries(nlo_fft_backend INTERFACE ${_nlo_fftw_target})
if(_nlo_fftw_include_dirs)
  target_include_directories(nlo_fft_backend INTERFACE ${_nlo_fftw_include_dirs})
endif()

if(NLO_ENABLE_VKFFT)
  include(FetchContent)
  find_package(Vulkan QUIET COMPONENTS glslang)
  if(NOT TARGET Vulkan::glslang)
    message(FATAL_ERROR
      "Required target Vulkan::glslang is unavailable.")
  endif()

  FetchContent_Declare(
    vkfft
    URL https://github.com/DTolm/VkFFT/archive/refs/heads/master.zip
  )
  FetchContent_MakeAvailable(vkfft)

  target_include_directories(nlo_fft_backend INTERFACE ${vkfft_SOURCE_DIR})
  target_compile_definitions(nlo_fft_backend INTERFACE VKFFT_BACKEND=0)
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
    if(MSVC)
      # Vulkan/glslang static libs frequently ship without redistributable PDBs.
      target_link_options(nlo_fft_backend INTERFACE /IGNORE:4099)
    endif()
  endif()
endif()

target_compile_definitions(nlo_fft_backend INTERFACE
  NLO_ENABLE_VKFFT=$<IF:$<BOOL:${NLO_ENABLE_VKFFT}>,1,0>
)
