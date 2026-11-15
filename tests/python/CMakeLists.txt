find_package(Python3 COMPONENTS Interpreter REQUIRED)

# sbom.cdx.json is generated from cmake/NLODependencies.cmake; fail if a pin was
# bumped without regenerating it (python tools/write_sbom.py).
add_test(NAME test_sbom_pins_match
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/write_sbom.py" --check
)

# Every binding re-declares the C API enums and limits in its own source, with
# no compile-time link between the copies. Scans the sources only, so it needs
# neither Julia nor MATLAB nor a built library.
add_test(NAME test_api_parity
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_api_parity.py"
)

# pyproject.toml and julia/Project.toml each declare a literal version, copied
# from the CMake project version. They must agree at every commit so a release
# tag can be cut from any commit: the tag names the version the artifacts carry.
# Sync them with python tools/release/check_versions.py --write.
add_test(NAME test_version_copies_match
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/release/check_versions.py"
)

set(_PY_PKG_DIR "${CMAKE_SOURCE_DIR}/python")
if(WIN32)
  set(_NLOLIB_DLL "${_PY_PKG_DIR}/$<CONFIG>/nlolib.dll")
elseif(APPLE)
  set(_NLOLIB_DLL "${_PY_PKG_DIR}/libnlolib.dylib")
else()
  set(_NLOLIB_DLL "${_PY_PKG_DIR}/libnlolib.so")
endif()

add_test(NAME test_python_api_smoke
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/python/test_python_api_smoke.py"
)

add_test(NAME test_python_operator_regression
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/python/test_python_operator_regression.py"
)

add_test(NAME test_python_storage_chunking
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/python/test_python_storage_chunking.py"
)

add_test(NAME test_python_runtime_expr_translation
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/python/test_python_runtime_expr_translation.py"
)

add_test(NAME test_python_grin_plot_validation
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/python/test_python_grin_plot_validation.py"
)

add_test(NAME test_python_example_metrics
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/python/test_python_example_metrics.py"
)

add_test(NAME test_python_example_spectral
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/python/test_python_example_spectral.py"
)

add_test(NAME test_python_example_oop_structure
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/python/test_python_example_oop_structure.py"
)

add_test(NAME test_python_reference_benchmarks
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/python/test_python_reference_benchmarks.py"
)

add_test(NAME test_python_package_surface
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/python/test_python_package_surface.py"
)

add_test(NAME test_python_perf_profile
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/python/test_python_perf_profile.py"
)

add_test(NAME test_python_second_order_soliton_plot_cleanup
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/python/test_python_second_order_soliton_plot_cleanup.py"
)

add_test(NAME test_python_tensor_backend_scaling_cleanup
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/python/test_python_tensor_backend_scaling_cleanup.py"
)

add_test(NAME test_python_run_store
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/python/test_python_run_store.py"
)

set(_NLOLIB_TEST_PATH "$ENV{PATH}")
string(REPLACE ";" "\\;" _NLOLIB_TEST_PATH "${_NLOLIB_TEST_PATH}")

set_tests_properties(test_python_api_smoke PROPERTIES
  ENVIRONMENT "PYTHONPATH=${_PY_PKG_DIR};NLOLIB_LIBRARY=$<TARGET_FILE:nlolib>;PATH=${_NLOLIB_TEST_PATH}"
)

set_tests_properties(test_python_operator_regression PROPERTIES
  ENVIRONMENT "PYTHONPATH=${_PY_PKG_DIR};NLOLIB_LIBRARY=$<TARGET_FILE:nlolib>;PATH=${_NLOLIB_TEST_PATH}"
)

set_tests_properties(test_python_storage_chunking PROPERTIES
  ENVIRONMENT "PYTHONPATH=${_PY_PKG_DIR};NLOLIB_LIBRARY=$<TARGET_FILE:nlolib>;PATH=${_NLOLIB_TEST_PATH}"
)

set_tests_properties(test_python_runtime_expr_translation PROPERTIES
  ENVIRONMENT "PYTHONPATH=${_PY_PKG_DIR};NLOLIB_LIBRARY=$<TARGET_FILE:nlolib>;PATH=${_NLOLIB_TEST_PATH}"
)

set_tests_properties(test_python_grin_plot_validation PROPERTIES
  ENVIRONMENT "PYTHONPATH=${_PY_PKG_DIR};NLOLIB_LIBRARY=$<TARGET_FILE:nlolib>;PATH=${_NLOLIB_TEST_PATH}"
)

set_tests_properties(test_python_example_metrics PROPERTIES
  ENVIRONMENT "PYTHONPATH=${_PY_PKG_DIR};NLOLIB_LIBRARY=$<TARGET_FILE:nlolib>;PATH=${_NLOLIB_TEST_PATH}"
)

set_tests_properties(test_python_example_spectral PROPERTIES
  ENVIRONMENT "PYTHONPATH=${_PY_PKG_DIR};NLOLIB_LIBRARY=$<TARGET_FILE:nlolib>;PATH=${_NLOLIB_TEST_PATH}"
)

set_tests_properties(test_python_example_oop_structure PROPERTIES
  ENVIRONMENT "PYTHONPATH=${_PY_PKG_DIR};NLOLIB_LIBRARY=$<TARGET_FILE:nlolib>;PATH=${_NLOLIB_TEST_PATH}"
)

set_tests_properties(test_python_reference_benchmarks PROPERTIES
  ENVIRONMENT "PYTHONPATH=${_PY_PKG_DIR};NLOLIB_LIBRARY=$<TARGET_FILE:nlolib>;PATH=${_NLOLIB_TEST_PATH}"
)

set_tests_properties(test_python_package_surface PROPERTIES
  ENVIRONMENT "PYTHONPATH=${_PY_PKG_DIR};NLOLIB_LIBRARY=$<TARGET_FILE:nlolib>;PATH=${_NLOLIB_TEST_PATH}"
)

set_tests_properties(test_python_perf_profile PROPERTIES
  ENVIRONMENT "PYTHONPATH=${_PY_PKG_DIR};NLOLIB_LIBRARY=$<TARGET_FILE:nlolib>;PATH=${_NLOLIB_TEST_PATH}"
)

set_tests_properties(test_python_second_order_soliton_plot_cleanup PROPERTIES
  ENVIRONMENT "PYTHONPATH=${_PY_PKG_DIR};NLOLIB_LIBRARY=$<TARGET_FILE:nlolib>;PATH=${_NLOLIB_TEST_PATH}"
)

set_tests_properties(test_python_tensor_backend_scaling_cleanup PROPERTIES
  ENVIRONMENT "PYTHONPATH=${_PY_PKG_DIR};NLOLIB_LIBRARY=$<TARGET_FILE:nlolib>;PATH=${_NLOLIB_TEST_PATH}"
)

set_tests_properties(test_python_run_store PROPERTIES
  ENVIRONMENT "PYTHONPATH=${_PY_PKG_DIR};NLOLIB_LIBRARY=$<TARGET_FILE:nlolib>;PATH=${_NLOLIB_TEST_PATH}"
)
