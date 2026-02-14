cmake_minimum_required(VERSION 3.28.3)
project(nlolib VERSION 0.0.1 LANGUAGES C)

# Set C standard
set(CMAKE_C_STANDARD 99)

option(NLO_FFT_BACKEND_FFTW "Use FFTW backend" ON)
option(NLO_FFT_BACKEND_CUFFT "Use CUFFT backend" OFF)
option(NLO_VECTOR_BACKEND_VULKAN "Enable Vulkan backend for vector operations" OFF)
option(NLO_VULKAN_FETCH_HEADERS "Fetch Vulkan headers when not found locally" ON)
option(NLOLIB_BUILD_DOCS "Build API docs with Doxygen" ON)
set(NLO_VULKAN_HEADERS_URL
  "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/heads/main.zip"
  CACHE STRING "URL used to fetch Vulkan headers when local headers are unavailable")
set(NLO_CPU_SIMD_LEVEL "AUTO" CACHE STRING "CPU SIMD level: AUTO, AVX2, AVX, or SCALAR")
set_property(CACHE NLO_CPU_SIMD_LEVEL PROPERTY STRINGS AUTO AVX2 AVX SCALAR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(NLOVersioning)
nlo_install_minor_version_hook(VERSION_FILE "${CMAKE_CURRENT_LIST_FILE}")

include(simd)

add_subdirectory(src)
add_subdirectory(python)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
  nlo_add_build_test_patch_target(VERSION_FILE "${CMAKE_CURRENT_LIST_FILE}")
endif()

if(NLOLIB_BUILD_DOCS)
  find_package(Doxygen QUIET)

  if(DOXYGEN_FOUND)

    include(FetchContent)
    FetchContent_Declare(
      doxygen-awesome-css
      URL https://github.com/jothepro/doxygen-awesome-css/archive/refs/heads/main.zip
    )
    FetchContent_MakeAvailable(doxygen-awesome-css)
    FetchContent_GetProperties(doxygen-awesome-css SOURCE_DIR AWESOME_CSS_DIR)

    set(DOXYGEN_IN "${CMAKE_CURRENT_LIST_DIR}/cmake/Doxyfile.in")
    set(DOXYGEN_OUT "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")
    set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/docs")
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    add_custom_target(docs
      COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating API documentation with Doxygen"
      VERBATIM
    )
  else()
    message(STATUS "Doxygen not found, docs target will not be available.")
  endif()
endif()
