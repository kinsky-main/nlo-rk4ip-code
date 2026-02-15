cmake_minimum_required(VERSION 3.28.3)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(NLOBuildConfig)
nlo_configure_build_configurations()

project(nlolib VERSION 0.0.1 LANGUAGES C)

# Set C standard
set(CMAKE_C_STANDARD 99)

option(NLO_ENABLE_FFTW "Enable FFTW integration for CPU FFT execution" ON)
option(NLO_ENABLE_VKFFT "Enable VkFFT integration for Vulkan FFT execution" ON)
option(NLO_VECTOR_BACKEND_VULKAN "Enable Vulkan backend for vector operations" ON)
option(NLO_VULKAN_FETCH_HEADERS "Fetch Vulkan headers when not found locally" ON)
option(NLOLIB_BUILD_DOCS "Build API docs with Doxygen" ON)
set(NLO_VULKAN_HEADERS_URL
  "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/heads/main.zip"
  CACHE STRING "URL used to fetch Vulkan headers when local headers are unavailable")
set(NLO_CPU_SIMD_LEVEL "AUTO" CACHE STRING "CPU SIMD level: AUTO, AVX2, AVX, or SCALAR")
set_property(CACHE NLO_CPU_SIMD_LEVEL PROPERTY STRINGS AUTO AVX2 AVX SCALAR)

include(NLOVersioning)
include(NLODocs)
nlo_install_minor_version_hook(VERSION_FILE "${CMAKE_CURRENT_LIST_FILE}")

include(simd)

add_subdirectory(src)
add_subdirectory(python)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
  nlo_add_build_test_patch_target(VERSION_FILE "${CMAKE_CURRENT_LIST_FILE}")
endif()

nlo_configure_docs_target()
