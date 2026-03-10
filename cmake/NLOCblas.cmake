include_guard(GLOBAL)

include(GNUInstallDirs)

set(NLO_CBLAS_RUNTIME_HINTS "" CACHE INTERNAL "Resolved CBLAS runtime files copied next to built targets")

function(_nlo_ensure_cblas_interface_target)
  if(TARGET NLO::cblas)
    return()
  endif()

  add_library(nlo_cblas INTERFACE)
  add_library(NLO::cblas ALIAS nlo_cblas)
endfunction()

function(_nlo_try_configure_system_cblas)
  if(TARGET NLO::cblas)
    return()
  endif()

  find_path(_nlo_cblas_include_dir
    NAMES cblas.h
    HINTS
      ENV OpenBLAS_HOME
      ENV OPENBLAS_ROOT
      ENV CONDA_PREFIX
    PATH_SUFFIXES
      include
      Include
  )

  find_library(_nlo_cblas_library
    NAMES openblas libopenblas openblas_static
    HINTS
      ENV OpenBLAS_HOME
      ENV OPENBLAS_ROOT
      ENV CONDA_PREFIX
    PATH_SUFFIXES
      lib
      lib64
      libs
      Library/lib
  )

  if(NOT _nlo_cblas_include_dir OR NOT _nlo_cblas_library)
    return()
  endif()

  _nlo_ensure_cblas_interface_target()
  target_include_directories(nlo_cblas INTERFACE "${_nlo_cblas_include_dir}")
  target_link_libraries(nlo_cblas INTERFACE "${_nlo_cblas_library}")
  set(NLO_CBLAS_RUNTIME_HINTS "" CACHE INTERNAL "Resolved CBLAS runtime files copied next to built targets" FORCE)
endfunction()

function(_nlo_configure_windows_prebuilt_openblas)
  if(TARGET NLO::cblas)
    return()
  endif()

  include(FetchContent)

  FetchContent_Declare(
    nlo_openblas_windows
    URL "${NLO_OPENBLAS_WINDOWS_URL}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )

  FetchContent_GetProperties(nlo_openblas_windows)
  if(NOT nlo_openblas_windows_POPULATED)
    if(POLICY CMP0169)
      cmake_policy(PUSH)
      cmake_policy(SET CMP0169 OLD)
      FetchContent_Populate(nlo_openblas_windows)
      cmake_policy(POP)
    else()
      FetchContent_Populate(nlo_openblas_windows)
    endif()
  endif()

  set(_nlo_openblas_root "${nlo_openblas_windows_SOURCE_DIR}")

  find_path(_nlo_cblas_include_dir
    NAMES cblas.h
    PATHS "${_nlo_openblas_root}"
    PATH_SUFFIXES
      include
      Include
    NO_DEFAULT_PATH
  )

  find_library(_nlo_cblas_implib
    NAMES libopenblas openblas
    PATHS "${_nlo_openblas_root}"
    PATH_SUFFIXES
      lib
      lib64
      libs
      Library/lib
      x64/lib
    NO_DEFAULT_PATH
  )

  find_file(_nlo_cblas_runtime
    NAMES libopenblas.dll openblas.dll
    PATHS "${_nlo_openblas_root}"
    PATH_SUFFIXES
      bin
      Bin
    NO_DEFAULT_PATH
  )

  if(NOT _nlo_cblas_include_dir OR NOT _nlo_cblas_implib)
    message(FATAL_ERROR
      "Failed to locate cblas.h and an import library in fetched OpenBLAS archive "
      "'${NLO_OPENBLAS_WINDOWS_URL}'.")
  endif()

  add_library(nlo_openblas_windows SHARED IMPORTED GLOBAL)
  set_target_properties(nlo_openblas_windows PROPERTIES
    IMPORTED_IMPLIB "${_nlo_cblas_implib}"
    INTERFACE_INCLUDE_DIRECTORIES "${_nlo_cblas_include_dir}"
  )
  if(_nlo_cblas_runtime)
    set_target_properties(nlo_openblas_windows PROPERTIES
      IMPORTED_LOCATION "${_nlo_cblas_runtime}"
    )
    set(NLO_CBLAS_RUNTIME_HINTS "${_nlo_cblas_runtime}"
      CACHE INTERNAL "Resolved CBLAS runtime files copied next to built targets" FORCE)
  else()
    set(NLO_CBLAS_RUNTIME_HINTS "" CACHE INTERNAL "Resolved CBLAS runtime files copied next to built targets" FORCE)
  endif()

  _nlo_ensure_cblas_interface_target()
  target_link_libraries(nlo_cblas INTERFACE nlo_openblas_windows)
endfunction()

function(_nlo_configure_fetched_openblas_source)
  if(TARGET NLO::cblas)
    return()
  endif()

  include(ExternalProject)
  include(FetchContent)

  FetchContent_Declare(
    nlo_openblas_source
    URL "${NLO_OPENBLAS_URL}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )

  FetchContent_GetProperties(nlo_openblas_source)
  if(NOT nlo_openblas_source_POPULATED)
    if(POLICY CMP0169)
      cmake_policy(PUSH)
      cmake_policy(SET CMP0169 OLD)
      FetchContent_Populate(nlo_openblas_source)
      cmake_policy(POP)
    else()
      FetchContent_Populate(nlo_openblas_source)
    endif()
  endif()

  set(_nlo_openblas_install_dir "${CMAKE_BINARY_DIR}/_deps/nlo_openblas")
  set(_nlo_openblas_library_path
    "${_nlo_openblas_install_dir}/${CMAKE_INSTALL_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}openblas${CMAKE_STATIC_LIBRARY_SUFFIX}")
  set(_nlo_openblas_include_dir "${_nlo_openblas_install_dir}/include")

  ExternalProject_Add(nlo_openblas_build
    SOURCE_DIR "${nlo_openblas_source_SOURCE_DIR}"
    CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${_nlo_openblas_install_dir}
      -DBUILD_SHARED_LIBS=OFF
      -DBUILD_TESTING=OFF
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      -DNOFORTRAN=ON
      -DBUILD_WITHOUT_LAPACK=ON
      -DUSE_OPENMP=OFF
      -DUSE_THREAD=OFF
      -DDYNAMIC_ARCH=OFF
      -DTARGET=GENERIC
    BUILD_BYPRODUCTS "${_nlo_openblas_library_path}"
    INSTALL_DIR "${_nlo_openblas_install_dir}"
  )

  add_library(nlo_openblas_static STATIC IMPORTED GLOBAL)
  set_target_properties(nlo_openblas_static PROPERTIES
    IMPORTED_LOCATION "${_nlo_openblas_library_path}"
    INTERFACE_INCLUDE_DIRECTORIES "${_nlo_openblas_include_dir}"
  )
  add_dependencies(nlo_openblas_static nlo_openblas_build)

  _nlo_ensure_cblas_interface_target()
  target_link_libraries(nlo_cblas INTERFACE nlo_openblas_static)
  target_include_directories(nlo_cblas INTERFACE "${_nlo_openblas_include_dir}")
  set(NLO_CBLAS_RUNTIME_HINTS "" CACHE INTERNAL "Resolved CBLAS runtime files copied next to built targets" FORCE)
endfunction()

function(nlo_configure_cblas out_target out_include_dirs)
  if(TARGET NLO::cblas)
    get_target_property(_nlo_cblas_include_dirs nlo_cblas INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT _nlo_cblas_include_dirs OR _nlo_cblas_include_dirs STREQUAL "_nlo_cblas_include_dirs-NOTFOUND")
      set(_nlo_cblas_include_dirs "")
    endif()
    set(${out_target} "NLO::cblas" PARENT_SCOPE)
    set(${out_include_dirs} "${_nlo_cblas_include_dirs}" PARENT_SCOPE)
    return()
  endif()

  if(NLO_CBLAS_PREFER_SYSTEM)
    _nlo_try_configure_system_cblas()
  endif()

  if(NOT TARGET NLO::cblas)
    if(WIN32)
      _nlo_configure_windows_prebuilt_openblas()
    else()
      _nlo_configure_fetched_openblas_source()
    endif()
  endif()

  if(NOT TARGET NLO::cblas)
    message(FATAL_ERROR "Failed to configure an OpenBLAS-backed CBLAS dependency.")
  endif()

  get_target_property(_nlo_cblas_include_dirs nlo_cblas INTERFACE_INCLUDE_DIRECTORIES)
  if(NOT _nlo_cblas_include_dirs OR _nlo_cblas_include_dirs STREQUAL "_nlo_cblas_include_dirs-NOTFOUND")
    set(_nlo_cblas_include_dirs "")
  endif()
  set(${out_target} "NLO::cblas" PARENT_SCOPE)
  set(${out_include_dirs} "${_nlo_cblas_include_dirs}" PARENT_SCOPE)
endfunction()
