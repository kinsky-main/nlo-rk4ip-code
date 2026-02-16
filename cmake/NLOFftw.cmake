include_guard(GLOBAL)

function(_nlo_get_target_interface_includes target out_var)
  get_target_property(_nlo_include_dirs ${target} INTERFACE_INCLUDE_DIRECTORIES)
  if(NOT _nlo_include_dirs OR _nlo_include_dirs STREQUAL "_nlo_include_dirs-NOTFOUND")
    set(_nlo_include_dirs "")
  endif()
  set(${out_var} "${_nlo_include_dirs}" PARENT_SCOPE)
endfunction()

function(nlo_configure_fftw out_target out_include_dirs)
  if(NOT TARGET FFTW3::fftw3)
    include(FetchContent)

    if(NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
      set(CMAKE_POLICY_VERSION_MINIMUM "3.5")
    endif()

    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
    set(ENABLE_FLOAT OFF CACHE BOOL "single-precision" FORCE)
    set(ENABLE_LONG_DOUBLE OFF CACHE BOOL "long-double precision" FORCE)
    set(ENABLE_QUAD_PRECISION OFF CACHE BOOL "quadruple-precision" FORCE)
    set(ENABLE_OPENMP OFF CACHE BOOL "Use OpenMP for multithreading" FORCE)
    set(ENABLE_THREADS OFF CACHE BOOL "Use pthread for multithreading" FORCE)

    FetchContent_Declare(
      fftw
      URL "https://www.fftw.org/${NLO_FFTW_GIT_TAG}.tar.gz"
    )
    FetchContent_MakeAvailable(fftw)

    if(NOT TARGET fftw3)
      message(FATAL_ERROR "Expected FFTW target 'fftw3' was not created.")
    endif()

    # FFTW upstream only exports install-time include paths. Add a build-time path.
    target_include_directories(fftw3 INTERFACE
      "$<BUILD_INTERFACE:${fftw_SOURCE_DIR}/api>"
    )

    if(NOT TARGET FFTW3::fftw3)
      add_library(FFTW3::fftw3 ALIAS fftw3)
    endif()
  endif()

  if(NOT TARGET FFTW3::fftw3)
    message(FATAL_ERROR "FFTW target FFTW3::fftw3 was not resolved.")
  endif()

  _nlo_get_target_interface_includes(FFTW3::fftw3 _nlo_fftw_include_dirs)
  set(${out_target} "FFTW3::fftw3" PARENT_SCOPE)
  set(${out_include_dirs} "${_nlo_fftw_include_dirs}" PARENT_SCOPE)
endfunction()
