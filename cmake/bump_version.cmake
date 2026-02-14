if(NOT DEFINED VERSION_FILE OR VERSION_FILE STREQUAL "")
  message(FATAL_ERROR "VERSION_FILE is required.")
endif()

if(NOT EXISTS "${VERSION_FILE}")
  message(FATAL_ERROR "VERSION_FILE does not exist: ${VERSION_FILE}")
endif()

if(NOT DEFINED BUMP_KIND OR BUMP_KIND STREQUAL "")
  set(BUMP_KIND "PATCH")
endif()
string(TOUPPER "${BUMP_KIND}" BUMP_KIND)

if(NOT DEFINED PROJECT_ROOT OR PROJECT_ROOT STREQUAL "")
  set(PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
endif()

file(READ "${VERSION_FILE}" _nlo_version_file_content)
string(REGEX MATCH "project\\([^)]*\\)" _nlo_project_line "${_nlo_version_file_content}")
if(NOT _nlo_project_line)
  message(FATAL_ERROR "Could not locate project(...) declaration in ${VERSION_FILE}")
endif()

string(REGEX MATCH "VERSION[ \t]+([0-9]+)\\.([0-9]+)\\.([0-9]+)" _nlo_version_match "${_nlo_project_line}")
if(NOT _nlo_version_match)
  message(FATAL_ERROR "Could not locate project VERSION in ${VERSION_FILE}")
endif()

set(_nlo_major "${CMAKE_MATCH_1}")
set(_nlo_minor "${CMAKE_MATCH_2}")
set(_nlo_patch "${CMAKE_MATCH_3}")

if(BUMP_KIND STREQUAL "MINOR")
  math(EXPR _nlo_minor "${_nlo_minor} + 1")
  set(_nlo_patch 0)
elseif(BUMP_KIND STREQUAL "PATCH")
  math(EXPR _nlo_patch "${_nlo_patch} + 1")
else()
  message(FATAL_ERROR "Unsupported BUMP_KIND='${BUMP_KIND}'. Use PATCH or MINOR.")
endif()

set(_nlo_new_version "${_nlo_major}.${_nlo_minor}.${_nlo_patch}")
string(REGEX REPLACE "VERSION[ \t]+[0-9]+\\.[0-9]+\\.[0-9]+" "VERSION ${_nlo_new_version}" _nlo_project_line_updated "${_nlo_project_line}")
string(REPLACE "${_nlo_project_line}" "${_nlo_project_line_updated}" _nlo_updated_content "${_nlo_version_file_content}")

if(_nlo_updated_content STREQUAL _nlo_version_file_content)
  message(FATAL_ERROR "Version update failed for ${VERSION_FILE}")
endif()

file(WRITE "${VERSION_FILE}" "${_nlo_updated_content}")
message(STATUS "Updated project version to ${_nlo_new_version}")
