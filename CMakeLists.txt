cmake_minimum_required(VERSION 3.22.1)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(NLOBuildConfig)
configure_build_configurations()

project(nlolib VERSION 1.10.32 LANGUAGES C)

# Set C standard
set(CMAKE_C_STANDARD 99)

option(ENABLE_RK4_DEBUG_DIAGNOSTICS "Enable debug-only RK4/dispersion diagnostics in Debug builds" ON)
option(INSTALL_GIT_HOOKS "Install repository git hooks during CMake configure" OFF)
option(NLOLIB_BUILD_DOCS "Build API docs with Doxygen" ON)
option(NLOLIB_BUILD_BENCHMARKS "Build benchmark executables" ON)
option(NLOLIB_BUILD_EXAMPLES "Build example executables" ON)
option(NLOLIB_BUILD_MATLAB_TESTS "Enable MATLAB runtime-handle parser tests when MATLAB is available" OFF)
option(ENABLE_VULKAN_BACKEND "Enable Vulkan vector backend and shader compilation path" ON)
option(ENABLE_VKFFT "Enable VkFFT backend integration for Vulkan FFT execution" ON)
option(BUMP_PATCH_ON_BUILD "Bump patch version in CMakeLists.txt after successful builds" OFF)
option(SQLITE_USE_FETCHCONTENT "Deprecated: SQLite amalgamation is always used" ON)
set(NLOLIB_GLSLANG_PROVIDER "FETCH" CACHE STRING
  "glslang provider for VkFFT: AUTO, SYSTEM, or FETCH")
set_property(CACHE NLOLIB_GLSLANG_PROVIDER PROPERTY STRINGS AUTO SYSTEM FETCH)

if(NOT ENABLE_VULKAN_BACKEND)
  set(ENABLE_VKFFT OFF CACHE BOOL "Enable VkFFT backend integration for Vulkan FFT execution" FORCE)
  if(NLOLIB_BUILD_BENCHMARKS)
    message(WARNING "Disabling NLOLIB_BUILD_BENCHMARKS because ENABLE_VULKAN_BACKEND=OFF.")
    set(NLOLIB_BUILD_BENCHMARKS OFF CACHE BOOL "Build benchmark executables" FORCE)
  endif()
endif()

set(VULKAN_HEADERS_URL
  "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/heads/main.zip"
  CACHE STRING "URL used to fetch Vulkan headers when local headers are unavailable")
set(GLSLANG_GIT_TAG
  "12.3.1"
  CACHE STRING "glslang git tag used for VkFFT when NLOLIB_GLSLANG_PROVIDER=FETCH")
set(SQLITE_AMALGAMATION_URL
  "https://www.sqlite.org/2025/sqlite-amalgamation-3490200.zip"
  CACHE STRING "URL used to fetch SQLite amalgamation when SQLite3 is unavailable")
set(FFTW_GIT_TAG
  "fftw-3.3.10"
  CACHE STRING "FFTW git tag used for in-tree static FFTW builds")
option(NLOLIB_STATIC_MSVC_RUNTIME
  "Link the MSVC runtime statically so redistributables are not required"
  ${WIN32})
if(WIN32 AND MSVC AND NLOLIB_STATIC_MSVC_RUNTIME)
  # Removes the MSVCP140/VCRUNTIME140 imports, so a client machine needs no
  # Visual C++ redistributable.  Safe here because the public API is
  # caller-allocates throughout: nothing malloc'd inside the library is ever
  # freed by the caller, so a private CRT heap cannot be crossed.
  # Set before add_subdirectory so fetched dependencies match; mismatched CRTs
  # do not link.
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if(ENABLE_VKFFT)
  enable_language(CXX)
endif()

include(NLOVersioning)
include(NLODocs)
if(INSTALL_GIT_HOOKS)
  install_minor_version_hook(VERSION_FILE "${CMAKE_CURRENT_LIST_FILE}")
endif()

add_subdirectory(src)
add_subdirectory(python)
add_subdirectory(matlab)
add_subdirectory(julia)

include(CTest)
if(BUMP_PATCH_ON_BUILD)
  add_patch_bump_on_build(
    VERSION_FILE "${CMAKE_CURRENT_LIST_FILE}"
    DEPENDS_ON_TARGETS nlolib
  )
endif()
if(BUILD_TESTING)
  add_subdirectory(tests)
  add_build_test_patch_target(VERSION_FILE "${CMAKE_CURRENT_LIST_FILE}")
endif()

if(NLOLIB_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
if(NLOLIB_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

configure_docs_target()
