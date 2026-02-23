find_package(Python3 COMPONENTS Interpreter REQUIRED)

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
