include_guard(GLOBAL)
include(CMakeParseArguments)

function(_nlo_resolve_git_dir out_var)
  set(_nlo_git_path "${CMAKE_SOURCE_DIR}/.git")
  set(_nlo_git_dir "")

  if(IS_DIRECTORY "${_nlo_git_path}")
    set(_nlo_git_dir "${_nlo_git_path}")
  elseif(EXISTS "${_nlo_git_path}")
    file(READ "${_nlo_git_path}" _nlo_git_pointer)
    string(REGEX MATCH "gitdir:[ \t]*([^\\r\\n]+)" _nlo_git_match "${_nlo_git_pointer}")
    if(_nlo_git_match)
      set(_nlo_git_dir "${CMAKE_MATCH_1}")
      if(NOT IS_ABSOLUTE "${_nlo_git_dir}")
        get_filename_component(_nlo_git_dir "${_nlo_git_dir}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
      endif()
    endif()
  endif()

  set("${out_var}" "${_nlo_git_dir}" PARENT_SCOPE)
endfunction()

function(nlo_install_minor_version_hook)
  set(options)
  set(oneValueArgs VERSION_FILE)
  set(multiValueArgs)
  cmake_parse_arguments(NLO "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT NLO_VERSION_FILE)
    message(FATAL_ERROR "nlo_install_minor_version_hook requires VERSION_FILE.")
  endif()

  _nlo_resolve_git_dir(_nlo_git_dir)
  if(NOT IS_DIRECTORY "${_nlo_git_dir}")
    return()
  endif()

  set(_nlo_hook_marker "nlolib-auto-minor-version-hook")
  set(_nlo_pre_commit_hook "${_nlo_git_dir}/hooks/pre-commit")
  get_filename_component(_nlo_hook_dir "${_nlo_pre_commit_hook}" DIRECTORY)
  set(_nlo_install_hook TRUE)

  if((EXISTS "${_nlo_pre_commit_hook}" AND NOT IS_WRITABLE "${_nlo_pre_commit_hook}") OR
     (NOT EXISTS "${_nlo_pre_commit_hook}" AND NOT IS_WRITABLE "${_nlo_hook_dir}"))
    message(STATUS "Pre-commit hook path is not writable; skipping automatic minor version hook installation.")
    return()
  endif()

  if(EXISTS "${_nlo_pre_commit_hook}")
    file(READ "${_nlo_pre_commit_hook}" _nlo_existing_hook)
    string(FIND "${_nlo_existing_hook}" "${_nlo_hook_marker}" _nlo_marker_index)
    if(_nlo_marker_index EQUAL -1)
      set(_nlo_install_hook FALSE)
      message(STATUS "Existing pre-commit hook detected; skipping automatic minor version hook installation.")
    endif()
  endif()

  if(NOT _nlo_install_hook)
    return()
  endif()

  set(NLO_HOOK_MARKER "${_nlo_hook_marker}")
  set(NLO_HOOK_VERSION_FILE "${NLO_VERSION_FILE}")
  set(NLO_HOOK_BUMP_SCRIPT "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/bump_version.cmake")
  get_filename_component(NLO_HOOK_PROJECT_ROOT "${NLO_VERSION_FILE}" DIRECTORY)
  configure_file(
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/pre-commit.in"
    "${_nlo_pre_commit_hook}"
    @ONLY
    NEWLINE_STYLE UNIX
  )
  file(CHMOD "${_nlo_pre_commit_hook}"
    PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
  )
endfunction()

function(nlo_add_build_test_patch_target)
  set(options)
  set(oneValueArgs VERSION_FILE TARGET_NAME)
  set(multiValueArgs)
  cmake_parse_arguments(NLO "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT NLO_VERSION_FILE)
    message(FATAL_ERROR "nlo_add_build_test_patch_target requires VERSION_FILE.")
  endif()

  if(NOT NLO_TARGET_NAME)
    set(NLO_TARGET_NAME "build_test_and_bump_version")
  endif()

  if(TARGET "${NLO_TARGET_NAME}")
    return()
  endif()

  add_custom_target("${NLO_TARGET_NAME}"
    COMMAND "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}" --config "$<CONFIG>" --target all
    COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure --build-config "$<CONFIG>"
    COMMAND
      "${CMAKE_COMMAND}"
      "-DVERSION_FILE=${NLO_VERSION_FILE}"
      "-DBUMP_KIND=PATCH"
      "-DPROJECT_ROOT=${CMAKE_SOURCE_DIR}"
      -P
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/bump_version.cmake"
    USES_TERMINAL
    COMMENT "Build all targets, run tests, and bump patch version when successful"
  )
endfunction()
