set(JULIA_STAGE_DIR "${CMAKE_BINARY_DIR}/julia_package" CACHE PATH
    "Directory where the Julia package files are staged")

add_custom_target(julia_stage
    COMMENT "Staging Julia package files"
    DEPENDS nlolib
)

add_custom_command(TARGET julia_stage POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${JULIA_STAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${JULIA_STAGE_DIR}/src"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${JULIA_STAGE_DIR}/test"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${JULIA_STAGE_DIR}/lib"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${JULIA_STAGE_DIR}/examples"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/julia/Project.toml"
        "${JULIA_STAGE_DIR}/Project.toml"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/julia/src"
        "${JULIA_STAGE_DIR}/src"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/julia/test"
        "${JULIA_STAGE_DIR}/test"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/examples/julia"
        "${JULIA_STAGE_DIR}/examples/julia"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/src/nlolib.h"
        "${JULIA_STAGE_DIR}/lib/nlolib.h"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:nlolib>"
        "${JULIA_STAGE_DIR}/lib/$<TARGET_FILE_NAME:nlolib>"
    COMMAND ${CMAKE_COMMAND}
        -DNLO_RUNTIME_SOURCE="$<TARGET_FILE:nlolib>"
        -DNLO_RUNTIME_DEST="${JULIA_STAGE_DIR}/lib"
        -DNLO_RUNTIME_HINTS="${SQLITE_RUNTIME_HINTS}"
        -P "${CMAKE_SOURCE_DIR}/cmake/copy_runtime_deps.cmake"
    COMMENT "Copying Julia package sources, tests, examples, public header, and shared library to ${JULIA_STAGE_DIR}"
)
