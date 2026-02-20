function(nlo_configure_sqlite target_name)
  find_package(SQLite3 QUIET)
  if(SQLite3_FOUND)
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
    target_include_directories(${target_name} PRIVATE ${NLO_SQLITE3_INCLUDE_DIR})
    target_link_libraries(${target_name} PRIVATE ${NLO_SQLITE3_LIBRARY})
    target_compile_definitions(${target_name} PRIVATE NLO_HAVE_SQLITE3=1)
  endif()
endfunction()

