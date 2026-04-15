include_guard(GLOBAL)

include(GNUInstallDirs)

set(CBLAS_RUNTIME_HINTS "" CACHE INTERNAL "Resolved CBLAS runtime files copied next to built targets")

function(ensure_cblas_interface_target)
  if(TARGET NLO::cblas)
    return()
  endif()

  add_library(cblas INTERFACE)
  add_library(NLO::cblas ALIAS cblas)
endfunction()

function(try_configure_system_cblas)
  if(TARGET NLO::cblas)
    return()
  endif()

  find_path(cblas_include_dir
    NAMES cblas.h
    HINTS
      ENV OpenBLAS_HOME
      ENV OPENBLAS_ROOT
      ENV CONDA_PREFIX
    PATH_SUFFIXES
      include
      Include
  )

  find_library(cblas_library
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

  if(NOT cblas_include_dir OR NOT cblas_library)
    return()
  endif()

  ensure_cblas_interface_target()
  target_include_directories(cblas INTERFACE "${cblas_include_dir}")
  target_link_libraries(cblas INTERFACE "${cblas_library}")
  set(CBLAS_RUNTIME_HINTS "" CACHE INTERNAL "Resolved CBLAS runtime files copied next to built targets" FORCE)
endfunction()

function(configure_windows_prebuilt_openblas)
  if(TARGET NLO::cblas)
    return()
  endif()

  include(FetchContent)

  FetchContent_Declare(
    openblas_windows
    URL "${OPENBLAS_WINDOWS_URL}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )

  FetchContent_GetProperties(openblas_windows)
  if(NOT openblas_windows_POPULATED)
    if(POLICY CMP0169)
      cmake_policy(PUSH)
      cmake_policy(SET CMP0169 OLD)
      FetchContent_Populate(openblas_windows)
      cmake_policy(POP)
    else()
      FetchContent_Populate(openblas_windows)
    endif()
  endif()

  set(openblas_root "${openblas_windows_SOURCE_DIR}")

  find_path(cblas_include_dir
    NAMES cblas.h
    PATHS "${openblas_root}"
    PATH_SUFFIXES
      include
      Include
    NO_DEFAULT_PATH
  )

  find_library(cblas_implib
    NAMES libopenblas openblas
    PATHS "${openblas_root}"
    PATH_SUFFIXES
      lib
      lib64
      libs
      Library/lib
      x64/lib
    NO_DEFAULT_PATH
  )

  find_file(cblas_runtime
    NAMES libopenblas.dll openblas.dll
    PATHS "${openblas_root}"
    PATH_SUFFIXES
      bin
      Bin
    NO_DEFAULT_PATH
  )

  if(NOT cblas_include_dir OR NOT cblas_implib)
    message(FATAL_ERROR
      "Failed to locate cblas.h and an import library in fetched OpenBLAS archive "
      "'${OPENBLAS_WINDOWS_URL}'.")
  endif()

  add_library(openblas_windows SHARED IMPORTED GLOBAL)
  set_target_properties(openblas_windows PROPERTIES
    IMPORTED_IMPLIB "${cblas_implib}"
    INTERFACE_INCLUDE_DIRECTORIES "${cblas_include_dir}"
  )
  if(cblas_runtime)
    set_target_properties(openblas_windows PROPERTIES
      IMPORTED_LOCATION "${cblas_runtime}"
    )
    set(CBLAS_RUNTIME_HINTS "${cblas_runtime}"
      CACHE INTERNAL "Resolved CBLAS runtime files copied next to built targets" FORCE)
  else()
    set(CBLAS_RUNTIME_HINTS "" CACHE INTERNAL "Resolved CBLAS runtime files copied next to built targets" FORCE)
  endif()

  ensure_cblas_interface_target()
  target_link_libraries(cblas INTERFACE openblas_windows)
endfunction()

function(configure_fetched_openblas_source)
  if(TARGET NLO::cblas)
    return()
  endif()

  include(ExternalProject)
  include(FetchContent)

  FetchContent_Declare(
    openblas_source
    URL "${OPENBLAS_URL}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )

  FetchContent_GetProperties(openblas_source)
  if(NOT openblas_source_POPULATED)
    if(POLICY CMP0169)
      cmake_policy(PUSH)
      cmake_policy(SET CMP0169 OLD)
      FetchContent_Populate(openblas_source)
      cmake_policy(POP)
    else()
      FetchContent_Populate(openblas_source)
    endif()
  endif()

  set(openblas_install_dir "${CMAKE_BINARY_DIR}/_deps/openblas")
  set(openblas_library_path
    "${openblas_install_dir}/${CMAKE_INSTALL_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}openblas${CMAKE_STATIC_LIBRARY_SUFFIX}")
  set(openblas_include_dir "${openblas_install_dir}/include")

  ExternalProject_Add(openblas_build
    SOURCE_DIR "${openblas_source_SOURCE_DIR}"
    CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${openblas_install_dir}
      -DBUILD_SHARED_LIBS=OFF
      -DBUILD_TESTING=OFF
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      -DNOFORTRAN=ON
      -DBUILD_WITHOUT_LAPACK=ON
      -DUSE_OPENMP=OFF
      -DUSE_THREAD=OFF
      -DDYNAMIC_ARCH=OFF
      -DTARGET=GENERIC
    BUILD_BYPRODUCTS "${openblas_library_path}"
    INSTALL_DIR "${openblas_install_dir}"
  )

  add_library(openblas_static STATIC IMPORTED GLOBAL)
  set_target_properties(openblas_static PROPERTIES
    IMPORTED_LOCATION "${openblas_library_path}"
    INTERFACE_INCLUDE_DIRECTORIES "${openblas_include_dir}"
  )
  add_dependencies(openblas_static openblas_build)

  ensure_cblas_interface_target()
  target_link_libraries(cblas INTERFACE openblas_static)
  target_include_directories(cblas INTERFACE "${openblas_include_dir}")
  set(CBLAS_RUNTIME_HINTS "" CACHE INTERNAL "Resolved CBLAS runtime files copied next to built targets" FORCE)
endfunction()

function(configure_cblas out_target out_include_dirs)
  if(TARGET NLO::cblas)
    get_target_property(cblas_include_dirs cblas INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT cblas_include_dirs OR cblas_include_dirs STREQUAL "cblas_include_dirs-NOTFOUND")
      set(cblas_include_dirs "")
    endif()
    set(${out_target} "NLO::cblas" PARENT_SCOPE)
    set(${out_include_dirs} "${cblas_include_dirs}" PARENT_SCOPE)
    return()
  endif()

  if(CBLAS_PREFER_SYSTEM)
    try_configure_system_cblas()
  endif()

  if(NOT TARGET NLO::cblas)
    if(WIN32)
      configure_windows_prebuilt_openblas()
    else()
      configure_fetched_openblas_source()
    endif()
  endif()

  if(NOT TARGET NLO::cblas)
    message(FATAL_ERROR "Failed to configure an OpenBLAS-backed CBLAS dependency.")
  endif()

  get_target_property(cblas_include_dirs cblas INTERFACE_INCLUDE_DIRECTORIES)
  if(NOT cblas_include_dirs OR cblas_include_dirs STREQUAL "cblas_include_dirs-NOTFOUND")
    set(cblas_include_dirs "")
  endif()
  set(${out_target} "NLO::cblas" PARENT_SCOPE)
  set(${out_include_dirs} "${cblas_include_dirs}" PARENT_SCOPE)
endfunction()
