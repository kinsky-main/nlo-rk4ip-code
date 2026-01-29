if(NOT TARGET nlolib)
  message(FATAL_ERROR "Target 'nlolib' not found. Ensure src/ is added before python/.")
endif()

# Place the shared library next to the Python package for easy loading.
set(_PYTHON_PKG_DIR "${CMAKE_CURRENT_LIST_DIR}")
foreach(cfg Debug Release RelWithDebInfo MinSizeRel)
  set_target_properties(nlolib PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_${cfg} "${_PYTHON_PKG_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_${cfg} "${_PYTHON_PKG_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY_${cfg} "${_PYTHON_PKG_DIR}"
  )
endforeach()

set_target_properties(nlolib PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${_PYTHON_PKG_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${_PYTHON_PKG_DIR}"
  ARCHIVE_OUTPUT_DIRECTORY "${_PYTHON_PKG_DIR}"
)
