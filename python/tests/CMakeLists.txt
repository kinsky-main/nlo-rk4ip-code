find_package(Python3 COMPONENTS Interpreter REQUIRED)

# The directory that holds the nlolib package: this repository's root.
get_filename_component(_PY_PKG_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

set(_NLOLIB_TEST_PATH "$ENV{PATH}")
string(REPLACE ";" "\;" _NLOLIB_TEST_PATH "${_NLOLIB_TEST_PATH}")

# test_python_wheel_jit_relocated builds the package, installs it into a fresh
# virtualenv outside the source tree, and asserts from there that a Vulkan run
# took the operator JIT path; skips without Vulkan. Nothing else covers the
# installed layout, where the archived branch silently fell back to the
# interpreter.
#
# test_python_fused_rk_step compares the fused Vulkan RK algebra against
# NLO_DISABLE_VK_FUSED_STEP=1; skips without Vulkan.
foreach(_test
    test_python_api_smoke
    test_python_operator_regression
    test_python_storage_chunking
    test_python_runtime_expr_translation
    test_python_grin_plot_validation
    test_python_example_metrics
    test_python_example_spectral
    test_python_example_oop_structure
    test_python_reference_benchmarks
    test_python_package_surface
    test_python_perf_profile
    test_python_second_order_soliton_plot_cleanup
    test_python_tensor_backend_scaling_cleanup
    test_python_wheel_jit_relocated
    test_python_fused_rk_step)
  add_test(NAME ${_test}
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/${_test}.py"
  )
  set_tests_properties(${_test} PROPERTIES
    ENVIRONMENT "PYTHONPATH=${_PY_PKG_DIR};NLOLIB_LIBRARY=$<TARGET_FILE:nlolib>;PATH=${_NLOLIB_TEST_PATH}"
  )
endforeach()

set_tests_properties(test_python_wheel_jit_relocated PROPERTIES TIMEOUT 900)
