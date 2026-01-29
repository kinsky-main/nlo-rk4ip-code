find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(_PY_PKG_DIR "${CMAKE_SOURCE_DIR}/python")
if(WIN32)
  set(_NLOLIB_DLL "${_PY_PKG_DIR}/$<CONFIG>/nlolib.dll")
elseif(APPLE)
  set(_NLOLIB_DLL "${_PY_PKG_DIR}/libnlolib.dylib")
else()
  set(_NLOLIB_DLL "${_PY_PKG_DIR}/libnlolib.so")
endif()

add_test(NAME test_python_bindings
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/python/test_bindings.py"
)

set_tests_properties(test_python_bindings PROPERTIES
  ENVIRONMENT "PYTHONPATH=${_PY_PKG_DIR};NLOLIB_LIBRARY=${_NLOLIB_DLL}"
)
