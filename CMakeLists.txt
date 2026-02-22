cmake_minimum_required(VERSION 3.28.3)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(NLOBuildConfig)
nlo_configure_build_configurations()

project(nlolib VERSION 0.15.0 LANGUAGES C)

# Set C standard
set(CMAKE_C_STANDARD 99)

option(NLO_ENABLE_RK4_DEBUG_DIAGNOSTICS "Enable debug-only RK4/dispersion diagnostics in Debug builds" ON)
option(NLO_INSTALL_GIT_HOOKS "Install repository git hooks during CMake configure" ON)
option(NLOLIB_BUILD_DOCS "Build API docs with Doxygen" ON)
option(NLOLIB_BUILD_BENCHMARKS "Build benchmark executables" ON)
option(NLOLIB_BUILD_EXAMPLES "Build example executables" ON)
option(NLO_ENABLE_VULKAN_BACKEND "Enable Vulkan vector backend and shader compilation path" ON)
option(NLO_ENABLE_VKFFT "Enable VkFFT backend integration for Vulkan FFT execution" ON)
option(NLO_BUMP_PATCH_ON_BUILD "Bump patch version in CMakeLists.txt after successful builds" ON)
option(NLO_SQLITE_USE_FETCHCONTENT "Fetch SQLite amalgamation via CMake instead of system SQLite discovery" OFF)

if(NOT NLO_ENABLE_VULKAN_BACKEND)
  set(NLO_ENABLE_VKFFT OFF CACHE BOOL "Enable VkFFT backend integration for Vulkan FFT execution" FORCE)
  if(NLOLIB_BUILD_BENCHMARKS)
    message(WARNING "Disabling NLOLIB_BUILD_BENCHMARKS because NLO_ENABLE_VULKAN_BACKEND=OFF.")
    set(NLOLIB_BUILD_BENCHMARKS OFF CACHE BOOL "Build benchmark executables" FORCE)
  endif()
endif()

set(NLO_VULKAN_HEADERS_URL
  "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/heads/main.zip"
  CACHE STRING "URL used to fetch Vulkan headers when local headers are unavailable")
set(NLO_SQLITE_AMALGAMATION_URL
  "https://www.sqlite.org/2025/sqlite-amalgamation-3490200.zip"
  CACHE STRING "URL used to fetch SQLite amalgamation when SQLite3 is unavailable")
set(NLO_FFTW_GIT_TAG
  "fftw-3.3.10"
  CACHE STRING "FFTW git tag used for in-tree static FFTW builds")
set(NLO_CPU_SIMD_LEVEL "AUTO" CACHE STRING "CPU SIMD level: AUTO, AVX2, AVX, or SCALAR")
set_property(CACHE NLO_CPU_SIMD_LEVEL PROPERTY STRINGS AUTO AVX2 AVX SCALAR)

include(NLOVersioning)
include(NLODocs)
if(NLO_INSTALL_GIT_HOOKS)
  nlo_install_minor_version_hook(VERSION_FILE "${CMAKE_CURRENT_LIST_FILE}")
endif()

include(simd)

add_subdirectory(src)
add_subdirectory(python)
add_subdirectory(matlab)

include(CTest)
if(NLO_BUMP_PATCH_ON_BUILD)
  nlo_add_patch_bump_on_build(
    VERSION_FILE "${CMAKE_CURRENT_LIST_FILE}"
    DEPENDS_ON_TARGETS nlolib
  )
endif()
if(BUILD_TESTING)
  add_subdirectory(tests)
  nlo_add_build_test_patch_target(VERSION_FILE "${CMAKE_CURRENT_LIST_FILE}")
endif()

if(NLOLIB_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
if(NLOLIB_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

nlo_configure_docs_target()

