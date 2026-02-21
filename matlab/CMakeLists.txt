# CMakelists.txt for MATLAB bindings

set(NLO_MATLAB_STAGE_DIR "${CMAKE_BINARY_DIR}/matlab_toolbox" CACHE PATH
    "Directory where MATLAB toolbox files are staged for packaging")

add_custom_target(matlab_stage
    COMMENT "Staging MATLAB toolbox files"
    DEPENDS nlolib
)
add_custom_command(TARGET matlab_stage POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${NLO_MATLAB_STAGE_DIR}/+nlolib"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${NLO_MATLAB_STAGE_DIR}/lib"
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
    COMMENT "Copying MATLAB wrapper, setup script, examples, header, and shared library to ${NLO_MATLAB_STAGE_DIR}"
)

if(WIN32)
  add_custom_command(TARGET matlab_stage POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
          "$<TARGET_FILE_DIR:nlolib>"
          "${NLO_MATLAB_STAGE_DIR}/lib"
      COMMENT "Copying nlolib output directory (including dependent DLLs) to ${NLO_MATLAB_STAGE_DIR}/lib"
  )
endif()
