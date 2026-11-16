# Consumed by the core repository's CMakeLists.txt, either as the python/
# submodule or through -DNLOLIB_PYTHON_DIR=<this checkout>. Not a project of its
# own: CMAKE_SOURCE_DIR is the core repository.
if(NOT TARGET nlolib)
  message(FATAL_ERROR "Target 'nlolib' not found. Configure the core repository "
                      "with -DNLOLIB_PYTHON_DIR pointing here; see README.md.")
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

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
