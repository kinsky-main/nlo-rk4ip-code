# CMakelists.txt for MATLAB bindings

set(NLO_MATLAB_STAGE_DIR "${CMAKE_BINARY_DIR}/matlab_toolbox" CACHE PATH
    "Directory where MATLAB toolbox files are staged for packaging")
if(WIN32)
  set(_NLO_MATLAB_PLATFORM_SUBDIR "win64")
elseif(APPLE)
  set(_NLO_MATLAB_PLATFORM_SUBDIR "maci64")
else()
  set(_NLO_MATLAB_PLATFORM_SUBDIR "glnxa64")
endif()

add_custom_target(matlab_stage
    COMMENT "Staging MATLAB toolbox files"
    DEPENDS nlolib
)
add_custom_command(TARGET matlab_stage POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${NLO_MATLAB_STAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${NLO_MATLAB_STAGE_DIR}/+nlolib"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${NLO_MATLAB_STAGE_DIR}/lib"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${NLO_MATLAB_STAGE_DIR}/lib/win64"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${NLO_MATLAB_STAGE_DIR}/lib/glnxa64"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${NLO_MATLAB_STAGE_DIR}/lib/maci64"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${NLO_MATLAB_STAGE_DIR}/examples/matlab"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/matlab/+nlolib"
        "${NLO_MATLAB_STAGE_DIR}/+nlolib"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/matlab/nlolib_setup.m"
        "${NLO_MATLAB_STAGE_DIR}/nlolib_setup.m"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/examples/matlab"
        "${NLO_MATLAB_STAGE_DIR}/examples/matlab"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/src/nlolib_matlab.h"
        "${NLO_MATLAB_STAGE_DIR}/lib/nlolib_matlab.h"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:nlolib>"
        "${NLO_MATLAB_STAGE_DIR}/lib/$<TARGET_FILE_NAME:nlolib>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:nlolib>"
        "${NLO_MATLAB_STAGE_DIR}/lib/${_NLO_MATLAB_PLATFORM_SUBDIR}/$<TARGET_FILE_NAME:nlolib>"
    COMMAND ${CMAKE_COMMAND}
        -DNLO_RUNTIME_SOURCE="$<TARGET_FILE:nlolib>"
        -DNLO_RUNTIME_DEST="${NLO_MATLAB_STAGE_DIR}/lib"
        -DNLO_RUNTIME_HINTS="${NLO_SQLITE_RUNTIME_HINTS}"
        -P "${CMAKE_SOURCE_DIR}/cmake/copy_runtime_deps.cmake"
    COMMAND ${CMAKE_COMMAND}
        -DNLO_RUNTIME_SOURCE="$<TARGET_FILE:nlolib>"
        -DNLO_RUNTIME_DEST="${NLO_MATLAB_STAGE_DIR}/lib/${_NLO_MATLAB_PLATFORM_SUBDIR}"
        -DNLO_RUNTIME_HINTS="${NLO_SQLITE_RUNTIME_HINTS}"
        -P "${CMAKE_SOURCE_DIR}/cmake/copy_runtime_deps.cmake"
    COMMENT "Copying MATLAB wrapper, setup script, examples, header, and shared library to ${NLO_MATLAB_STAGE_DIR}"
)
