# check_pins.cmake - verify every archive pinned in NLODependencies.cmake can be
# downloaded and matches its recorded SHA-256, without configuring the project.
#
#   cmake -P cmake/check_pins.cmake
#
# Exit status is non-zero on the first unreachable URL or hash mismatch. Needs
# network access, so it is a manual/CI check rather than a CTest case.

cmake_minimum_required(VERSION 3.22)
include("${CMAKE_CURRENT_LIST_DIR}/NLODependencies.cmake")

# name|URL variable|SHA-256 variable  (pipe-separated: semicolons would flatten the list)
set(_pins
  "fftw|FFTW_URL|FFTW_SHA256"
  "sqlite|SQLITE_AMALGAMATION_URL|SQLITE_AMALGAMATION_SHA256"
  "VkFFT|VKFFT_URL|VKFFT_SHA256"
  "glslang|GLSLANG_URL|GLSLANG_SHA256"
  "Vulkan-Headers|VULKAN_HEADERS_URL|VULKAN_HEADERS_SHA256"
  "doxygen-awesome-css|DOXYGEN_AWESOME_CSS_URL|DOXYGEN_AWESOME_CSS_SHA256"
)

set(_tmp "${CMAKE_CURRENT_LIST_DIR}/../build-pin-check.tmp")
file(MAKE_DIRECTORY "${_tmp}")
set(_failed 0)

foreach(_entry IN LISTS _pins)
  string(REPLACE "|" ";" _entry "${_entry}")
  list(GET _entry 0 _name)
  list(GET _entry 1 _url_var)
  list(GET _entry 2 _sha_var)
  set(_url "${${_url_var}}")
  set(_sha "${${_sha_var}}")
  get_filename_component(_file "${_url}" NAME)

  # Hash is checked by hand rather than with EXPECTED_HASH so a mismatch is
  # reported for every pin instead of aborting the script at the first one.
  file(DOWNLOAD "${_url}" "${_tmp}/${_name}-${_file}" STATUS _status TIMEOUT 120)
  list(GET _status 0 _code)
  list(GET _status 1 _msg)
  if(NOT _code EQUAL 0)
    message(STATUS "FAIL  ${_name}: ${_url}\n        download: ${_msg}")
    set(_failed 1)
    continue()
  endif()
  file(SHA256 "${_tmp}/${_name}-${_file}" _actual)
  if(_actual STREQUAL _sha)
    message(STATUS "OK    ${_name}: ${_url}")
  else()
    message(STATUS "FAIL  ${_name}: ${_url}\n        expected ${_sha}\n        got      ${_actual}")
    set(_failed 1)
  endif()
endforeach()

file(REMOVE_RECURSE "${_tmp}")
if(_failed)
  message(FATAL_ERROR "One or more pinned archives could not be fetched or did not match their SHA-256.")
endif()
message(STATUS "All pinned archives downloaded and verified.")
