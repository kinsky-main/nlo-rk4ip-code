if(NOT DEFINED NLO_RUNTIME_SOURCE OR NOT EXISTS "${NLO_RUNTIME_SOURCE}")
  message(FATAL_ERROR "NLO_RUNTIME_SOURCE is required and must exist.")
endif()
if(NOT DEFINED NLO_RUNTIME_DEST OR NLO_RUNTIME_DEST STREQUAL "")
  message(FATAL_ERROR "NLO_RUNTIME_DEST is required.")
endif()

file(MAKE_DIRECTORY "${NLO_RUNTIME_DEST}")

file(GET_RUNTIME_DEPENDENCIES
  RESOLVED_DEPENDENCIES_VAR _nlo_runtime_deps
  UNRESOLVED_DEPENDENCIES_VAR _nlo_unresolved_runtime_deps
  POST_EXCLUDE_REGEXES
    "^api-ms-win-.*"
    "^ext-ms-.*"
    "^[A-Za-z]:[/\\\\][Ww][Ii][Nn][Dd][Oo][Ww][Ss][/\\\\].*"
    "^[A-Za-z]:[/\\\\][Pp]rogram [Ff]iles[/\\\\][Mm][Aa][Tt][Ll][Aa][Bb][/\\\\].*"
    "^/System/Library/.*"
    "^/usr/lib/.*"
    "^/lib/.*"
  LIBRARIES "${NLO_RUNTIME_SOURCE}"
)

foreach(_nlo_dep IN LISTS _nlo_runtime_deps)
  string(TOLOWER "${_nlo_dep}" _nlo_dep_lower)
  if(_nlo_dep_lower MATCHES "^[a-z]:[/\\\\]windows[/\\\\].*" OR
     _nlo_dep_lower MATCHES "^[a-z]:[/\\\\]program files[/\\\\]matlab[/\\\\].*" OR
     _nlo_dep_lower MATCHES "^/system/library/.*" OR
     _nlo_dep_lower MATCHES "^/usr/lib/.*" OR
     _nlo_dep_lower MATCHES "^/lib/.*")
    continue()
  endif()
  if(EXISTS "${_nlo_dep}")
    file(COPY "${_nlo_dep}" DESTINATION "${NLO_RUNTIME_DEST}")
  endif()
endforeach()

if(DEFINED NLO_RUNTIME_HINTS AND NOT NLO_RUNTIME_HINTS STREQUAL "")
  foreach(_nlo_hint IN LISTS NLO_RUNTIME_HINTS)
    if(EXISTS "${_nlo_hint}")
      file(COPY "${_nlo_hint}" DESTINATION "${NLO_RUNTIME_DEST}")
    endif()
  endforeach()
endif()

set(_nlo_unresolved_filtered "")
foreach(_nlo_missing IN LISTS _nlo_unresolved_runtime_deps)
  if(_nlo_missing MATCHES "^api-ms-win-.*" OR _nlo_missing MATCHES "^ext-ms-.*")
    continue()
  endif()
  list(APPEND _nlo_unresolved_filtered "${_nlo_missing}")
endforeach()

if(_nlo_unresolved_filtered)
  message(WARNING "Unresolved runtime dependencies for ${NLO_RUNTIME_SOURCE}: ${_nlo_unresolved_filtered}")
endif()
