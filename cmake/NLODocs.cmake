function(configure_docs_target)
  if(NOT NLOLIB_BUILD_DOCS)
    return()
  endif()

  find_package(Doxygen QUIET)
  if(NOT DOXYGEN_FOUND)
    message(STATUS "Doxygen not found, docs target will not be available.")
    return()
  endif()

  include(FetchContent)
  FetchContent_Declare(
    doxygen-awesome-css
    URL "${DOXYGEN_AWESOME_CSS_URL}"
    URL_HASH "SHA256=${DOXYGEN_AWESOME_CSS_SHA256}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(doxygen-awesome-css)
  FetchContent_GetProperties(doxygen-awesome-css SOURCE_DIR AWESOME_CSS_DIR)

  set(DOXYGEN_IN "${CMAKE_SOURCE_DIR}/cmake/Doxyfile.in")
  set(DOXYGEN_OUT "${CMAKE_BINARY_DIR}/Doxyfile")
  set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/docs")
  configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
  add_custom_target(docs
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM
  )
endfunction()
