include_guard(GLOBAL)
include(CMakeParseArguments)

function(resolve_git_dir out_var)
  set(git_path "${CMAKE_SOURCE_DIR}/.git")
  set(git_dir "")

  if(IS_DIRECTORY "${git_path}")
    set(git_dir "${git_path}")
  elseif(EXISTS "${git_path}")
    file(READ "${git_path}" git_pointer)
    string(REGEX MATCH "gitdir:[ \t]*([^\\r\\n]+)" git_match "${git_pointer}")
    if(git_match)
      set(git_dir "${CMAKE_MATCH_1}")
      if(NOT IS_ABSOLUTE "${git_dir}")
        get_filename_component(git_dir "${git_dir}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
      endif()
    endif()
  endif()

  # Worktrees point at "<common-git-dir>/worktrees/<name>".
  # Hooks live in the common git dir referenced by "commondir".
  if(IS_DIRECTORY "${git_dir}" AND EXISTS "${git_dir}/commondir")
    file(READ "${git_dir}/commondir" common_dir_raw)
    string(STRIP "${common_dir_raw}" common_dir)
    if(common_dir)
      if(NOT IS_ABSOLUTE "${common_dir}")
        get_filename_component(common_dir "${common_dir}" ABSOLUTE BASE_DIR "${git_dir}")
      endif()
      if(IS_DIRECTORY "${common_dir}")
        set(git_dir "${common_dir}")
      endif()
    endif()
  endif()

  set("${out_var}" "${git_dir}" PARENT_SCOPE)
endfunction()

function(directory_is_writable path out_var)
  set(can_write FALSE)

  if(IS_DIRECTORY "${path}")
    string(RANDOM LENGTH 8 ALPHABET 0123456789abcdef probe_suffix)
    set(probe_file "${path}/.nlo-write-probe-${probe_suffix}")
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E touch "${probe_file}"
      RESULT_VARIABLE touch_result
      OUTPUT_QUIET
      ERROR_QUIET
    )
    if(touch_result EQUAL 0)
      set(can_write TRUE)
      if(EXISTS "${probe_file}")
        file(REMOVE "${probe_file}")
      endif()
    endif()
  endif()

  set("${out_var}" "${can_write}" PARENT_SCOPE)
endfunction()

function(file_is_writable path out_var)
  set(can_write FALSE)

  if(EXISTS "${path}")
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E touch "${path}"
      RESULT_VARIABLE touch_result
      OUTPUT_QUIET
      ERROR_QUIET
    )
    if(touch_result EQUAL 0)
      set(can_write TRUE)
    endif()
  endif()

  set("${out_var}" "${can_write}" PARENT_SCOPE)
endfunction()

function(install_minor_version_hook)
  set(options)
  set(oneValueArgs VERSION_FILE)
  set(multiValueArgs)
  cmake_parse_arguments(NLO "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT NLO_VERSION_FILE)
    message(FATAL_ERROR "install_minor_version_hook requires VERSION_FILE.")
  endif()

  resolve_git_dir(git_dir)
  if(NOT IS_DIRECTORY "${git_dir}")
    return()
  endif()

  set(hook_marker "nlolib-auto-minor-version-hook")
  set(pre_commit_hook "${git_dir}/hooks/pre-commit")
  get_filename_component(hook_dir "${pre_commit_hook}" DIRECTORY)
  set(install_hook TRUE)

  if(NOT IS_DIRECTORY "${hook_dir}")
    get_filename_component(hook_parent "${hook_dir}" DIRECTORY)
    directory_is_writable("${hook_parent}" hook_parent_writable)
    if(NOT IS_DIRECTORY "${hook_parent}" OR NOT hook_parent_writable)
      message(STATUS "Pre-commit hook path is not writable; skipping automatic minor version hook installation.")
      return()
    endif()
    file(MAKE_DIRECTORY "${hook_dir}")
  endif()

  if(EXISTS "${pre_commit_hook}")
    file_is_writable("${pre_commit_hook}" hook_path_writable)
  else()
    directory_is_writable("${hook_dir}" hook_path_writable)
  endif()
  if(NOT hook_path_writable)
    message(STATUS "Pre-commit hook path is not writable; skipping automatic minor version hook installation.")
    return()
  endif()

  if(EXISTS "${pre_commit_hook}")
    file(READ "${pre_commit_hook}" existing_hook)
    string(FIND "${existing_hook}" "${hook_marker}" marker_index)
    if(marker_index EQUAL -1)
      set(install_hook FALSE)
      message(STATUS "Existing pre-commit hook detected; skipping automatic minor version hook installation.")
    endif()
  endif()

  if(NOT install_hook)
    return()
  endif()

  set(HOOK_MARKER "${hook_marker}")
  set(HOOK_VERSION_FILE "${NLO_VERSION_FILE}")
  set(HOOK_BUMP_SCRIPT "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/bump_version.cmake")
  get_filename_component(HOOK_PROJECT_ROOT "${NLO_VERSION_FILE}" DIRECTORY)
  configure_file(
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/pre-commit.in"
    "${pre_commit_hook}"
    @ONLY
    NEWLINE_STYLE UNIX
  )
  file(CHMOD "${pre_commit_hook}"
    PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
  )
endfunction()

function(add_patch_bump_on_build)
  set(options)
  set(oneValueArgs VERSION_FILE TARGET_NAME)
  set(multiValueArgs DEPENDS_ON_TARGETS)
  cmake_parse_arguments(NLO "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT NLO_VERSION_FILE)
    message(FATAL_ERROR "add_patch_bump_on_build requires VERSION_FILE.")
  endif()

  if(NOT NLO_TARGET_NAME)
    set(NLO_TARGET_NAME "patch_bump_on_build")
  endif()

  if(TARGET "${NLO_TARGET_NAME}")
    return()
  endif()

  add_custom_target("${NLO_TARGET_NAME}" ALL
    COMMAND
      "${CMAKE_COMMAND}"
      "-DVERSION_FILE=${NLO_VERSION_FILE}"
      "-DBUMP_KIND=PATCH"
      "-DPROJECT_ROOT=${CMAKE_SOURCE_DIR}"
      -P
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/bump_version.cmake"
    COMMENT "Bump patch version after successful build"
    VERBATIM
  )

  if(NLO_DEPENDS_ON_TARGETS)
    add_dependencies("${NLO_TARGET_NAME}" ${NLO_DEPENDS_ON_TARGETS})
  endif()
endfunction()

function(add_build_test_patch_target)
  set(options)
  set(oneValueArgs VERSION_FILE TARGET_NAME)
  set(multiValueArgs)
  cmake_parse_arguments(NLO "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT NLO_VERSION_FILE)
    message(FATAL_ERROR "add_build_test_patch_target requires VERSION_FILE.")
  endif()

  if(NOT NLO_TARGET_NAME)
    set(NLO_TARGET_NAME "build_test_and_bump_version")
  endif()

  if(TARGET "${NLO_TARGET_NAME}")
    return()
  endif()

  add_custom_target("${NLO_TARGET_NAME}"
    COMMAND "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}" --config "$<CONFIG>"
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
