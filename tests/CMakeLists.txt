# Tests CMakeLists.txt
#
# The binding suites (test_python_*, test_julia_*, test_matlab_bindings) are
# registered by each binding's own CMakeLists.txt; what is here is the native
# suite plus the source-only checks that belong to the core repository.

find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_subdirectory(fft)
add_subdirectory(core)
add_subdirectory(numerics)
add_subdirectory(abi)

# sbom.cdx.json is generated from cmake/NLODependencies.cmake and stamps the
# project version; fail if either moved without regenerating it
# (python tools/release/check_versions.py --write).
add_test(NAME test_sbom_pins_match
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/write_sbom.py" --check
)

# The three bindings share one public surface (tests/api_surface.json), with no
# compile-time link between the implementations. Scans the sources only, so it
# needs neither Julia nor MATLAB nor a built library, and fails on any
# difference. Only the bindings that are checked out are compared; ci.yml runs
# the script without --bindings, which requires all three.
set(_parity_bindings "")
set(_parity_args "")
foreach(_binding python julia matlab)
  string(TOUPPER "${_binding}" _BINDING)
  if(EXISTS "${NLOLIB_${_BINDING}_DIR}/CMakeLists.txt")
    list(APPEND _parity_bindings "${_binding}")
    list(APPEND _parity_args "--${_binding}-dir" "${NLOLIB_${_BINDING}_DIR}")
  endif()
endforeach()
if(_parity_bindings)
  list(JOIN _parity_bindings "," _parity_list)
  add_test(NAME test_api_parity
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_api_parity.py"
            --bindings "${_parity_list}" ${_parity_args} --surface enforce
  )
endif()

# Static check of the MATLAB binding's C type-name strings. Needs no MATLAB
# installation, so it runs whenever the binding is checked out.
if(EXISTS "${NLOLIB_MATLAB_DIR}/CMakeLists.txt")
  add_test(NAME test_matlab_binding_types
    COMMAND "${CMAKE_COMMAND}"
            "-DNLOLIB_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
            "-DNLOLIB_MATLAB_DIR=${NLOLIB_MATLAB_DIR}"
            -P "${CMAKE_SOURCE_DIR}/cmake/check_matlab_binding_types.cmake"
  )
endif()
