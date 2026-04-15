function(configure_sqlite target_name)
  set(SQLITE_RUNTIME_HINTS "" CACHE INTERNAL
      "Candidate sqlite runtime library paths for packaging helpers" FORCE)

  include(FetchContent)
  if(NOT DEFINED SQLITE_AMALGAMATION_URL OR SQLITE_AMALGAMATION_URL STREQUAL "")
    set(SQLITE_AMALGAMATION_URL
      "https://www.sqlite.org/2025/sqlite-amalgamation-3490200.zip")
  endif()

  if(NOT TARGET sqlite3)
    FetchContent_Declare(
      sqlite_amalgamation
      URL ${SQLITE_AMALGAMATION_URL}
    )
    FetchContent_MakeAvailable(sqlite_amalgamation)

    file(GLOB sqlite_src_candidates
      "${sqlite_amalgamation_SOURCE_DIR}/sqlite3.c"
      "${sqlite_amalgamation_SOURCE_DIR}/sqlite-amalgamation-*/sqlite3.c"
    )
    file(GLOB sqlite_inc_candidates
      "${sqlite_amalgamation_SOURCE_DIR}/sqlite3.h"
      "${sqlite_amalgamation_SOURCE_DIR}/sqlite-amalgamation-*/sqlite3.h"
    )

    list(LENGTH sqlite_src_candidates sqlite_src_count)
    list(LENGTH sqlite_inc_candidates sqlite_inc_count)
    if(sqlite_src_count EQUAL 0 OR sqlite_inc_count EQUAL 0)
      message(FATAL_ERROR
        "Failed to locate sqlite3.c/sqlite3.h in fetched SQLite amalgamation: "
        "${sqlite_amalgamation_SOURCE_DIR}")
    endif()

    list(GET sqlite_src_candidates 0 sqlite_src)
    list(GET sqlite_inc_candidates 0 sqlite_inc)
    get_filename_component(sqlite_inc_dir "${sqlite_inc}" DIRECTORY)

    add_library(sqlite3 STATIC "${sqlite_src}")
    target_include_directories(sqlite3 PUBLIC "${sqlite_inc_dir}")
  endif()

  set(SQLITE_RUNTIME_HINTS "" CACHE INTERNAL
      "Candidate sqlite runtime library paths for packaging helpers" FORCE)
  target_link_libraries(${target_name} PRIVATE sqlite3)
  target_compile_definitions(${target_name} PRIVATE HAVE_SQLITE3=1)
endfunction()
