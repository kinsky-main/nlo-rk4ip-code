function(nlo_configure_sqlite target_name)
  set(NLO_SQLITE_RUNTIME_HINTS "" CACHE INTERNAL
      "Candidate sqlite runtime library paths for packaging helpers" FORCE)

  function(_nlo_set_sqlite_runtime_hints)
    set(_nlo_sqlite_hints_local "")
    foreach(_nlo_candidate IN LISTS ARGN)
      if(_nlo_candidate AND EXISTS "${_nlo_candidate}")
        list(APPEND _nlo_sqlite_hints_local "${_nlo_candidate}")
      endif()
    endforeach()
    list(REMOVE_DUPLICATES _nlo_sqlite_hints_local)
    set(NLO_SQLITE_RUNTIME_HINTS "${_nlo_sqlite_hints_local}" CACHE INTERNAL
        "Candidate sqlite runtime library paths for packaging helpers" FORCE)
  endfunction()

  if(NOT NLO_SQLITE_USE_FETCHCONTENT)
    find_package(SQLite3 QUIET)
    if(SQLite3_FOUND)
      if(TARGET SQLite::SQLite3)
        get_target_property(_nlo_sqlite_imported_location
          SQLite::SQLite3 IMPORTED_LOCATION)
        get_target_property(_nlo_sqlite_imported_location_release
          SQLite::SQLite3 IMPORTED_LOCATION_RELEASE)
        get_target_property(_nlo_sqlite_imported_location_debug
          SQLite::SQLite3 IMPORTED_LOCATION_DEBUG)
        _nlo_set_sqlite_runtime_hints(
          "${_nlo_sqlite_imported_location}"
          "${_nlo_sqlite_imported_location_release}"
          "${_nlo_sqlite_imported_location_debug}"
        )
      endif()
      target_link_libraries(${target_name} PRIVATE SQLite::SQLite3)
      target_compile_definitions(${target_name} PRIVATE NLO_HAVE_SQLITE3=1)
      return()
    endif()

    set(_nlo_sqlite_hints)
    if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
      list(APPEND _nlo_sqlite_hints "$ENV{CONDA_PREFIX}")
    endif()
    if(DEFINED ENV{USERPROFILE} AND NOT "$ENV{USERPROFILE}" STREQUAL "")
      list(APPEND _nlo_sqlite_hints "$ENV{USERPROFILE}/miniconda3")
      list(APPEND _nlo_sqlite_hints "$ENV{USERPROFILE}/vcpkg/installed/x64-windows")
    endif()

    find_path(NLO_SQLITE3_INCLUDE_DIR
      NAMES sqlite3.h
      HINTS ${_nlo_sqlite_hints}
      PATH_SUFFIXES include Library/include
    )
    find_library(NLO_SQLITE3_LIBRARY
      NAMES sqlite3
      HINTS ${_nlo_sqlite_hints}
      PATH_SUFFIXES lib Library/lib
    )

    if(NLO_SQLITE3_INCLUDE_DIR AND NLO_SQLITE3_LIBRARY)
      get_filename_component(_nlo_sqlite_lib_dir "${NLO_SQLITE3_LIBRARY}" DIRECTORY)
      get_filename_component(_nlo_sqlite_lib_name "${NLO_SQLITE3_LIBRARY}" NAME_WE)
      _nlo_set_sqlite_runtime_hints(
        "${_nlo_sqlite_lib_dir}/sqlite3.dll"
        "${_nlo_sqlite_lib_dir}/${_nlo_sqlite_lib_name}.dll"
        "${_nlo_sqlite_lib_dir}/../bin/sqlite3.dll"
        "${_nlo_sqlite_lib_dir}/../sqlite3.dll"
      )
      target_include_directories(${target_name} PRIVATE ${NLO_SQLITE3_INCLUDE_DIR})
      target_link_libraries(${target_name} PRIVATE ${NLO_SQLITE3_LIBRARY})
      target_compile_definitions(${target_name} PRIVATE NLO_HAVE_SQLITE3=1)
      return()
    endif()
  endif()

  include(FetchContent)
  if(NOT DEFINED NLO_SQLITE_AMALGAMATION_URL OR NLO_SQLITE_AMALGAMATION_URL STREQUAL "")
    set(NLO_SQLITE_AMALGAMATION_URL
      "https://www.sqlite.org/2025/sqlite-amalgamation-3490200.zip")
  endif()

  if(NOT TARGET nlo_sqlite3)
    FetchContent_Declare(
      nlo_sqlite_amalgamation
      URL ${NLO_SQLITE_AMALGAMATION_URL}
    )
    FetchContent_MakeAvailable(nlo_sqlite_amalgamation)

    file(GLOB _nlo_sqlite_src_candidates
      "${nlo_sqlite_amalgamation_SOURCE_DIR}/sqlite3.c"
      "${nlo_sqlite_amalgamation_SOURCE_DIR}/sqlite-amalgamation-*/sqlite3.c"
    )
    file(GLOB _nlo_sqlite_inc_candidates
      "${nlo_sqlite_amalgamation_SOURCE_DIR}/sqlite3.h"
      "${nlo_sqlite_amalgamation_SOURCE_DIR}/sqlite-amalgamation-*/sqlite3.h"
    )

    list(LENGTH _nlo_sqlite_src_candidates _nlo_sqlite_src_count)
    list(LENGTH _nlo_sqlite_inc_candidates _nlo_sqlite_inc_count)
    if(_nlo_sqlite_src_count EQUAL 0 OR _nlo_sqlite_inc_count EQUAL 0)
      message(FATAL_ERROR
        "Failed to locate sqlite3.c/sqlite3.h in fetched SQLite amalgamation: "
        "${nlo_sqlite_amalgamation_SOURCE_DIR}")
    endif()

    list(GET _nlo_sqlite_src_candidates 0 _nlo_sqlite_src)
    list(GET _nlo_sqlite_inc_candidates 0 _nlo_sqlite_inc)
    get_filename_component(_nlo_sqlite_inc_dir "${_nlo_sqlite_inc}" DIRECTORY)

    add_library(nlo_sqlite3 STATIC "${_nlo_sqlite_src}")
    target_include_directories(nlo_sqlite3 PUBLIC "${_nlo_sqlite_inc_dir}")
  endif()

  set(NLO_SQLITE_RUNTIME_HINTS "" CACHE INTERNAL
      "Candidate sqlite runtime library paths for packaging helpers" FORCE)
  target_link_libraries(${target_name} PRIVATE nlo_sqlite3)
  target_compile_definitions(${target_name} PRIVATE NLO_HAVE_SQLITE3=1)
endfunction()
