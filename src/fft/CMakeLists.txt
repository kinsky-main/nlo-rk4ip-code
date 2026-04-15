add_library(fft_backend INTERFACE)

include(NLOFftw)
configure_fftw(fftw_target fftw_include_dirs)
target_link_libraries(fft_backend INTERFACE ${fftw_target})
if(fftw_include_dirs)
  target_include_directories(fft_backend INTERFACE ${fftw_include_dirs})
endif()

if(ENABLE_VKFFT)
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
  # VkFFT is used as headers-only; avoid adding its CMake sub-build because it
  # can inject fragile transitive glslang library paths on Windows generators.
  FetchContent_GetProperties(vkfft)
  if(NOT vkfft_POPULATED)
    if(POLICY CMP0169)
      cmake_policy(PUSH)
      cmake_policy(SET CMP0169 OLD)
      FetchContent_Populate(vkfft)
      cmake_policy(POP)
    else()
      FetchContent_Populate(vkfft)
    endif()
  endif()

  target_include_directories(fft_backend INTERFACE ${vkfft_SOURCE_DIR})
  target_compile_definitions(fft_backend INTERFACE VKFFT_BACKEND=0)
  target_link_libraries(fft_backend INTERFACE Vulkan::glslang)

  find_path(GLSLANG_C_INCLUDE_DIR
    NAMES glslang_c_interface.h
    HINTS
      ENV VULKAN_SDK
    PATH_SUFFIXES
      Include/glslang/Include
      glslang/Include
  )
  if(GLSLANG_C_INCLUDE_DIR)
    target_include_directories(fft_backend INTERFACE ${GLSLANG_C_INCLUDE_DIR})
  endif()

  if(WIN32 AND DEFINED ENV{VULKAN_SDK})
    target_link_directories(fft_backend INTERFACE "$ENV{VULKAN_SDK}/Lib")
    target_link_libraries(fft_backend INTERFACE
      $<$<CONFIG:Debug>:MachineIndependentd;OSDependentd;GenericCodeGend;SPIRVd;SPIRV-Toolsd;SPIRV-Tools-optd;glslang-default-resource-limitsd>
      $<$<NOT:$<CONFIG:Debug>>:MachineIndependent;OSDependent;GenericCodeGen;SPIRV;SPIRV-Tools;SPIRV-Tools-opt;glslang-default-resource-limits>
    )
    if(MSVC)
      # Vulkan/glslang static libs frequently ship without redistributable PDBs.
      target_link_options(fft_backend INTERFACE /IGNORE:4099)
    endif()
  endif()
endif()

target_compile_definitions(fft_backend INTERFACE
  ENABLE_VKFFT=$<IF:$<BOOL:${ENABLE_VKFFT}>,1,0>
)
