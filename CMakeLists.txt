cmake_minimum_required(VERSION 3.22.1)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(NLOBuildConfig)
configure_build_configurations()

project(nlolib VERSION 0.2.1 LANGUAGES C)

# Set C standard
set(CMAKE_C_STANDARD 99)

option(ENABLE_RK4_DEBUG_DIAGNOSTICS "Enable debug-only RK4/dispersion diagnostics in Debug builds" ON)
option(INSTALL_GIT_HOOKS "Install the pre-commit hook that checks sbom.cdx.json matches the version and pins" OFF)
option(NLOLIB_BUILD_DOCS "Build API docs with Doxygen" ON)
option(NLOLIB_BUILD_BENCHMARKS "Build benchmark executables" ON)
option(NLOLIB_BUILD_EXAMPLES "Build example executables" ON)
option(NLOLIB_BUILD_MATLAB_TESTS "Enable MATLAB runtime-handle parser tests when MATLAB is available" OFF)
option(ENABLE_VULKAN_BACKEND "Enable Vulkan vector backend and shader compilation path" ON)
option(ENABLE_VKFFT "Enable VkFFT backend integration for Vulkan FFT execution" ON)
# Prototype: store Vulkan device buffers and run every kernel, the operator JIT
# and VkFFT in single precision, so the backend runs on devices without
# shaderFloat64 (Apple GPUs through MoltenVK). Host-side data stays double.
option(NLOLIB_VULKAN_FP32 "Run the Vulkan backend in single precision (prototype)" OFF)
if(NLOLIB_VULKAN_FP32)
  add_compile_definitions(NLO_VK_FP32=1)
endif()
# Keep this OFF for a release build: the packaged version is read back from
# CMakeLists.txt after the build, so a bump mid-build publishes a version that
# does not match its tag. Bump deliberately with
# tools/release/check_versions.py --bump instead.
option(BUMP_PATCH_ON_BUILD "Bump patch version in CMakeLists.txt after successful builds" OFF)
option(SQLITE_USE_FETCHCONTENT "Deprecated: SQLite amalgamation is always used" ON)
set(NLOLIB_GLSLANG_PROVIDER "FETCH" CACHE STRING
  "glslang provider for VkFFT: AUTO, SYSTEM, or FETCH")
set_property(CACHE NLOLIB_GLSLANG_PROVIDER PROPERTY STRINGS AUTO SYSTEM FETCH)
set(NLOLIB_FFTW_PROVIDER "FETCH" CACHE STRING
  "FFTW provider: FETCH (pinned release, built in-tree) or SYSTEM (installed libfftw3)")
set_property(CACHE NLOLIB_FFTW_PROVIDER PROPERTY STRINGS SYSTEM FETCH)
set(NLOLIB_SQLITE_PROVIDER "FETCH" CACHE STRING
  "SQLite provider: FETCH (pinned amalgamation, compiled in) or SYSTEM (installed libsqlite3)")
set_property(CACHE NLOLIB_SQLITE_PROVIDER PROPERTY STRINGS SYSTEM FETCH)
option(NLOLIB_STATIC_LIBSTDCXX
  "On Linux with GCC, embed libstdc++/libgcc in nlolib and hide their symbols"
  ON)

if(NOT ENABLE_VULKAN_BACKEND)
  set(ENABLE_VKFFT OFF CACHE BOOL "Enable VkFFT backend integration for Vulkan FFT execution" FORCE)
  if(NLOLIB_BUILD_BENCHMARKS)
    message(WARNING "Disabling NLOLIB_BUILD_BENCHMARKS because ENABLE_VULKAN_BACKEND=OFF.")
    set(NLOLIB_BUILD_BENCHMARKS OFF CACHE BOOL "Build benchmark executables" FORCE)
  endif()
endif()

# Every third-party source pin (version, URL, SHA-256) lives in one place.
include(NLODependencies)
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
  install_version_check_hook()
endif()

add_subdirectory(src)

include(CTest)

# Each language binding lives in its own repository and is mounted here as a
# git submodule (git submodule update --init). A binding that is not checked
# out is skipped, so the core library builds on its own. NLOLIB_<LANG>_DIR
# points the build at a checkout elsewhere, which is how a binding repository
# builds and tests itself against the core revision it pins.
foreach(_binding python matlab julia)
  string(TOUPPER "${_binding}" _BINDING)
  set(NLOLIB_${_BINDING}_DIR "${CMAKE_CURRENT_LIST_DIR}/${_binding}" CACHE PATH
      "Checkout of the ${_binding} binding repository")
  # A normal variable over the cache entry, so a Windows path given with
  # backslashes on the command line is usable in CMake strings.
  file(TO_CMAKE_PATH "${NLOLIB_${_BINDING}_DIR}" NLOLIB_${_BINDING}_DIR)
  if(EXISTS "${NLOLIB_${_BINDING}_DIR}/CMakeLists.txt")
    add_subdirectory("${NLOLIB_${_BINDING}_DIR}" "${CMAKE_BINARY_DIR}/${_binding}")
  else()
    message(STATUS "No ${_binding} binding at ${NLOLIB_${_BINDING}_DIR}; skipping it "
                   "(git submodule update --init ${_binding}).")
  endif()
endforeach()

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
