# CMakeLists.txt for the MATLAB binding.
#
# Consumed by the core repository's CMakeLists.txt, either as the matlab/
# submodule or through -DNLOLIB_MATLAB_DIR=<this checkout>. Not a project of
# its own: CMAKE_SOURCE_DIR is the core repository. The staged layout is what
# an installed toolbox looks like, and differs from this repository's: the
# examples are staged under examples/matlab and the guide under docs/.

set(MATLAB_STAGE_DIR "${CMAKE_BINARY_DIR}/matlab_toolbox" CACHE PATH
    "Directory where MATLAB toolbox files are staged for packaging")
if(WIN32)
  set(MATLAB_PLATFORM_SUBDIR "win64")
elseif(APPLE)
  set(MATLAB_PLATFORM_SUBDIR "maci64")
else()
  set(MATLAB_PLATFORM_SUBDIR "glnxa64")
endif()

add_custom_target(matlab_stage
    COMMENT "Staging MATLAB toolbox files"
    DEPENDS nlolib
)
add_custom_command(TARGET matlab_stage POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${MATLAB_STAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${MATLAB_STAGE_DIR}/+nlolib"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${MATLAB_STAGE_DIR}/lib"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${MATLAB_STAGE_DIR}/lib/${MATLAB_PLATFORM_SUBDIR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${MATLAB_STAGE_DIR}/examples/matlab"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${MATLAB_STAGE_DIR}/docs"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_LIST_DIR}/+nlolib"
        "${MATLAB_STAGE_DIR}/+nlolib"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_LIST_DIR}/nlolib_setup.m"
        "${MATLAB_STAGE_DIR}/nlolib_setup.m"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_LIST_DIR}/resources"
        "${MATLAB_STAGE_DIR}/resources"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_LIST_DIR}/examples"
        "${MATLAB_STAGE_DIR}/examples/matlab"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_LIST_DIR}/docs/user_guide.md"
        "${MATLAB_STAGE_DIR}/docs/matlab_user_guide.md"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/docs/runtime_operators.md"
        "${MATLAB_STAGE_DIR}/docs/runtime_operators.md"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/src/nlolib_matlab.h"
        "${MATLAB_STAGE_DIR}/lib/nlolib_matlab.h"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:nlolib>"
        "${MATLAB_STAGE_DIR}/lib/${MATLAB_PLATFORM_SUBDIR}/$<TARGET_FILE_NAME:nlolib>"
    COMMAND ${CMAKE_COMMAND}
        -DRUNTIME_SOURCE="$<TARGET_FILE:nlolib>"
        -DRUNTIME_DEST="${MATLAB_STAGE_DIR}/lib/${MATLAB_PLATFORM_SUBDIR}"
        -DRUNTIME_HINTS="${SQLITE_RUNTIME_HINTS}"
        -P "${CMAKE_SOURCE_DIR}/cmake/copy_runtime_deps.cmake"
    COMMENT "Copying MATLAB wrapper, setup script, examples, docs, header, and shared library to ${MATLAB_STAGE_DIR}"
)

if(BUILD_TESTING AND NLOLIB_BUILD_MATLAB_TESTS)
  add_subdirectory(tests)
endif()
