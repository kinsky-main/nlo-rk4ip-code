# SPDX-License-Identifier: MIT
# Findsimd.cmake - locate SIMDe (SIMD Everywhere) headers
#
# Provides:
#   simd_FOUND
#   simd_INCLUDE_DIR
#   simd::simd (INTERFACE target)
#
# Usage:
#   find_package(simd)

find_path(simd_INCLUDE_DIR
  NAMES simde/simde-arch.h
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(simd REQUIRED_VARS simd_INCLUDE_DIR)

if(simd_FOUND AND NOT TARGET simd::simd)
  add_library(simd::simd INTERFACE IMPORTED)
  set_target_properties(simd::simd PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${simd_INCLUDE_DIR}"
  )
endif()
