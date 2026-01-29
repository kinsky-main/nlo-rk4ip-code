# - Try to find FFTW3 (double precision) library and headers.
# Once done, defines:
#   FFTW3_FOUND
#   FFTW3_INCLUDE_DIRS
#   FFTW3_LIBRARIES
# And provides target:
#   FFTW3::fftw3

set(_FFTW3_HINTS)
if(DEFINED FFTW3_ROOT)
  list(APPEND _FFTW3_HINTS "${FFTW3_ROOT}")
endif()
if(DEFINED FFTW_ROOT)
  list(APPEND _FFTW3_HINTS "${FFTW_ROOT}")
endif()
if(DEFINED FFTW3_DIR)
  list(APPEND _FFTW3_HINTS "${FFTW3_DIR}")
endif()

find_path(FFTW3_INCLUDE_DIR
  NAMES fftw3.h
  HINTS ${_FFTW3_HINTS}
  PATH_SUFFIXES include
)

find_library(FFTW3_LIBRARY
  NAMES fftw3 fftw3-3 fftw3d libfftw3
  HINTS ${_FFTW3_HINTS}
  PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW3
  REQUIRED_VARS FFTW3_LIBRARY FFTW3_INCLUDE_DIR
)

if(FFTW3_FOUND)
  set(FFTW3_LIBRARIES ${FFTW3_LIBRARY})
  set(FFTW3_INCLUDE_DIRS ${FFTW3_INCLUDE_DIR})

  if(NOT TARGET FFTW3::fftw3)
    add_library(FFTW3::fftw3 UNKNOWN IMPORTED)
    set_target_properties(FFTW3::fftw3 PROPERTIES
      IMPORTED_LOCATION "${FFTW3_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIR}"
    )
  endif()
endif()
