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

file(READ "${VERSION_FILE}" version_file_content)
string(REGEX MATCH "project\\([^)]*\\)" project_line "${version_file_content}")
if(NOT project_line)
  message(FATAL_ERROR "Could not locate project(...) declaration in ${VERSION_FILE}")
endif()

string(REGEX MATCH "VERSION[ \t]+([0-9]+)\\.([0-9]+)\\.([0-9]+)" version_match "${project_line}")
if(NOT version_match)
  message(FATAL_ERROR "Could not locate project VERSION in ${VERSION_FILE}")
endif()

set(major "${CMAKE_MATCH_1}")
set(minor "${CMAKE_MATCH_2}")
set(patch "${CMAKE_MATCH_3}")

if(BUMP_KIND STREQUAL "MINOR")
  math(EXPR minor "${minor} + 1")
  set(patch 0)
elseif(BUMP_KIND STREQUAL "PATCH")
  math(EXPR patch "${patch} + 1")
else()
  message(FATAL_ERROR "Unsupported BUMP_KIND='${BUMP_KIND}'. Use PATCH or MINOR.")
endif()

set(new_version "${major}.${minor}.${patch}")
string(REGEX REPLACE "VERSION[ \t]+[0-9]+\\.[0-9]+\\.[0-9]+" "VERSION ${new_version}" project_line_updated "${project_line}")
string(REPLACE "${project_line}" "${project_line_updated}" updated_content "${version_file_content}")

if(updated_content STREQUAL version_file_content)
  message(FATAL_ERROR "Version update failed for ${VERSION_FILE}")
endif()

file(WRITE "${VERSION_FILE}" "${updated_content}")
message(STATUS "Updated project version to ${new_version}")
