cmake_minimum_required(VERSION 3.28.3)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(NLOBuildConfig)
nlo_configure_build_configurations()

project(nlolib VERSION 1.10.28 LANGUAGES C)

# Set C standard
set(CMAKE_C_STANDARD 99)

option(NLO_ENABLE_RK4_DEBUG_DIAGNOSTICS "Enable debug-only RK4/dispersion diagnostics in Debug builds" ON)
option(NLO_INSTALL_GIT_HOOKS "Install repository git hooks during CMake configure" ON)
option(NLOLIB_BUILD_DOCS "Build API docs with Doxygen" ON)
option(NLOLIB_BUILD_BENCHMARKS "Build benchmark executables" ON)
option(NLOLIB_BUILD_EXAMPLES "Build example executables" ON)
option(NLOLIB_BUILD_MATLAB_TESTS "Enable MATLAB runtime-handle parser tests when MATLAB is available" OFF)
option(NLO_ENABLE_VULKAN_BACKEND "Enable Vulkan vector backend and shader compilation path" ON)
option(NLO_ENABLE_VKFFT "Enable VkFFT backend integration for Vulkan FFT execution" ON)
option(NLO_ENABLE_CUDA_BACKEND "Enable CUDA backend API/build scaffolding" OFF)
option(NLO_ENABLE_CUFFT "Enable cuFFT/cuFFT Xt integration when CUDA backend support is present" OFF)
option(NLO_ENABLE_NVRTC "Enable NVRTC runtime operator JIT integration when CUDA backend support is present" OFF)
option(NLO_ENABLE_NCCL "Enable NCCL integration for single-node CUDA multi-GPU runs" OFF)
option(NLO_ENABLE_DOCKER_VALIDATION "Enable Docker validation helper targets when Docker is available" OFF)
option(NLO_BUMP_PATCH_ON_BUILD "Bump patch version in CMakeLists.txt after successful builds" ON)
option(NLO_SQLITE_USE_FETCHCONTENT "Deprecated: SQLite amalgamation is always used" ON)

if(NLO_ENABLE_CUDA_BACKEND)
  find_package(CUDAToolkit QUIET)
  if(NOT CUDAToolkit_FOUND)
    message(WARNING "Disabling NLO_ENABLE_CUDA_BACKEND because no CUDA toolkit was found.")
    set(NLO_ENABLE_CUDA_BACKEND OFF CACHE BOOL "Enable CUDA backend API/build scaffolding" FORCE)
  else()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures for nlolib" FORCE)
    endif()
  endif()
endif()

if(NOT NLO_ENABLE_CUDA_BACKEND)
  set(NLO_ENABLE_CUFFT OFF CACHE BOOL "Enable cuFFT/cuFFT Xt integration when CUDA backend support is present" FORCE)
  set(NLO_ENABLE_NVRTC OFF CACHE BOOL "Enable NVRTC runtime operator JIT integration when CUDA backend support is present" FORCE)
  set(NLO_ENABLE_NCCL OFF CACHE BOOL "Enable NCCL integration for single-node CUDA multi-GPU runs" FORCE)
endif()

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
if(WIN32)
  set(_NLO_CBLAS_PREFER_SYSTEM_DEFAULT OFF)
else()
  set(_NLO_CBLAS_PREFER_SYSTEM_DEFAULT ON)
endif()
option(NLO_CBLAS_PREFER_SYSTEM
  "Prefer a system-provided OpenBLAS/CBLAS before fetching OpenBLAS"
  ${_NLO_CBLAS_PREFER_SYSTEM_DEFAULT})
set(NLO_OPENBLAS_URL
  "https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.30/OpenBLAS-0.3.30.tar.gz"
  CACHE STRING "URL used to fetch OpenBLAS source for in-tree static CBLAS builds")
set(NLO_OPENBLAS_WINDOWS_URL
  "https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.30/OpenBLAS-0.3.30-x64.zip"
  CACHE STRING "URL used to fetch the prebuilt OpenBLAS package used on Windows")

include(NLOVersioning)
include(NLODocs)
if(NLO_INSTALL_GIT_HOOKS)
  nlo_install_minor_version_hook(VERSION_FILE "${CMAKE_CURRENT_LIST_FILE}")
endif()

add_subdirectory(src)
add_subdirectory(python)
add_subdirectory(matlab)
add_subdirectory(julia)

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

if(NLO_ENABLE_DOCKER_VALIDATION)
  find_program(NLO_DOCKER_EXECUTABLE docker)
  find_program(NLO_BASH_EXECUTABLE bash)
  if(NLO_DOCKER_EXECUTABLE AND NLO_BASH_EXECUTABLE)
    add_custom_target(docker_validate_cpu
      COMMAND "${NLO_BASH_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/docker_validate_cpu.sh"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      USES_TERMINAL
    )
    add_custom_target(docker_validate_vulkan
      COMMAND "${NLO_BASH_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/docker_validate_vulkan.sh"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      USES_TERMINAL
    )
    add_custom_target(docker_validate_cuda
      COMMAND "${NLO_BASH_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/docker_validate_cuda.sh"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      USES_TERMINAL
    )
    add_custom_target(docker_validate_cuda_multigpu
      COMMAND "${NLO_BASH_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/docker_validate_cuda_multigpu.sh"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      USES_TERMINAL
    )
  else()
    message(WARNING "NLO_ENABLE_DOCKER_VALIDATION is ON, but docker or bash was not found on PATH.")
  endif()
endif()

nlo_configure_docs_target()

