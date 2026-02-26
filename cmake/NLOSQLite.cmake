function(nlo_configure_sqlite target_name)
  set(NLO_SQLITE_RUNTIME_HINTS "" CACHE INTERNAL
      "Candidate sqlite runtime library paths for packaging helpers" FORCE)

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
