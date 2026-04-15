if(NOT DEFINED RUNTIME_SOURCE OR NOT EXISTS "${RUNTIME_SOURCE}")
  message(FATAL_ERROR "RUNTIME_SOURCE is required and must exist.")
endif()
if(NOT DEFINED RUNTIME_DEST OR RUNTIME_DEST STREQUAL "")
  message(FATAL_ERROR "RUNTIME_DEST is required.")
endif()

file(MAKE_DIRECTORY "${RUNTIME_DEST}")

set(runtime_directories "")
if(DEFINED RUNTIME_HINTS AND NOT RUNTIME_HINTS STREQUAL "")
  foreach(hint IN LISTS RUNTIME_HINTS)
    if(EXISTS "${hint}")
      get_filename_component(hint_dir "${hint}" DIRECTORY)
      if(EXISTS "${hint_dir}")
        list(APPEND runtime_directories "${hint_dir}")
      endif()
    endif()
  endforeach()
  list(REMOVE_DUPLICATES runtime_directories)
endif()

file(GET_RUNTIME_DEPENDENCIES
  RESOLVED_DEPENDENCIES_VAR runtime_deps
  UNRESOLVED_DEPENDENCIES_VAR unresolved_runtime_deps
  DIRECTORIES ${runtime_directories}
  POST_EXCLUDE_REGEXES
    "^api-ms-win-.*"
    "^ext-ms-.*"
    "^[A-Za-z]:[/\\\\][Ww][Ii][Nn][Dd][Oo][Ww][Ss][/\\\\].*"
    "^[A-Za-z]:[/\\\\][Pp]rogram [Ff]iles[/\\\\][Mm][Aa][Tt][Ll][Aa][Bb][/\\\\].*"
    "^/System/Library/.*"
    "^/usr/lib/.*"
    "^/lib/.*"
  LIBRARIES "${RUNTIME_SOURCE}"
)

foreach(dep IN LISTS runtime_deps)
  string(TOLOWER "${dep}" dep_lower)
  if(dep_lower MATCHES "^[a-z]:[/\\\\]windows[/\\\\].*" OR
     dep_lower MATCHES "^[a-z]:[/\\\\]program files[/\\\\]matlab[/\\\\].*" OR
     dep_lower MATCHES "^/system/library/.*" OR
     dep_lower MATCHES "^/usr/lib/.*" OR
     dep_lower MATCHES "^/lib/.*")
    continue()
  endif()
  if(EXISTS "${dep}")
    file(COPY "${dep}" DESTINATION "${RUNTIME_DEST}")
  endif()
endforeach()

if(DEFINED RUNTIME_HINTS AND NOT RUNTIME_HINTS STREQUAL "")
  foreach(hint IN LISTS RUNTIME_HINTS)
    if(EXISTS "${hint}")
      file(COPY "${hint}" DESTINATION "${RUNTIME_DEST}")
    endif()
  endforeach()
endif()

set(unresolved_filtered "")
foreach(missing IN LISTS unresolved_runtime_deps)
  if(missing MATCHES "^api-ms-win-.*" OR missing MATCHES "^ext-ms-.*")
    continue()
  endif()
  list(APPEND unresolved_filtered "${missing}")
endforeach()

if(unresolved_filtered)
  message(WARNING "Unresolved runtime dependencies for ${RUNTIME_SOURCE}: ${unresolved_filtered}")
endif()
