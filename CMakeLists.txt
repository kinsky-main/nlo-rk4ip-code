cmake_minimum_required(VERSION 3.18)
project(NonLinearOpticsSolver VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(USE_CUDA "Enable CUDA support" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build tests" ON)

if (WIN32)
    set(SQLite3_LIBRARY "C:/Program Files/SQLite3/lib/sqlite3.lib")
    set(SQLite3_INCLUDE_DIR "C:/Program Files/SQLite3/include")
    add_definitions(-D_USE_MATH_DEFINES)
endif()

find_package(SQLite3 REQUIRED)

# Enable vcpkg support
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "")
    message(STATUS "Vcpkg toolchain file set to ${CMAKE_TOOLCHAIN_FILE}")
else()
    message(STATUS "Vcpkg support not enabled")
endif()

# Find CUDA if enabled
if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CUDA_AVAILABLE TRUE)
        find_package(CUDAToolkit)
        message(STATUS "CUDA support enabled")
    else()
        message(WARNING "CUDA requested but not found. Building CPU-only version.")
        set(CUDA_AVAILABLE FALSE)
    endif()
else()
    set(CUDA_AVAILABLE FALSE)
    message(STATUS "CUDA support disabled")
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/rk45_cpu.cpp
)

# CUDA sources (if available)
if(CUDA_AVAILABLE)
    set(CUDA_SOURCES
        src/rk45_gpu.cu
        src/nlo_equations.cu
    )
endif()

# Create library
if(CUDA_AVAILABLE)
    add_library(nlo_solver ${SOURCES} ${CUDA_SOURCES})
    set_target_properties(nlo_solver PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    add_library(nlo_solver ${SOURCES})
endif()

message(STATUS "CUDA include directories: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

target_include_directories(nlo_solver PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Compiler flags
if(CUDA_AVAILABLE)
    target_compile_definitions(nlo_solver PUBLIC USE_CUDA)
    set_target_properties(nlo_solver PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    # Link CUDA runtime and cuFFT libraries
    target_link_libraries(nlo_solver PUBLIC 
        CUDA::cudart
        CUDA::cufft
    )
endif()

# Build examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Build tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
install(TARGETS nlo_solver
    EXPORT NonLinearOpticsSolverTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
)

install(EXPORT NonLinearOpticsSolverTargets
    FILE NonLinearOpticsSolverTargets.cmake
    NAMESPACE NLO::
    DESTINATION lib/cmake/NonLinearOpticsSolver
)
