add_executable(test_core_state_alloc
  ${CMAKE_CURRENT_LIST_DIR}/test_state_alloc.c
  ${CMAKE_SOURCE_DIR}/src/core/state.c
  ${CMAKE_SOURCE_DIR}/src/core/init.c
)

target_include_directories(test_core_state_alloc PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_core_state_alloc PRIVATE simd::simd)
if(WIN32)
  target_compile_definitions(test_core_state_alloc PRIVATE NLOLIB_EXPORTS)
endif()
if(UNIX)
  target_link_libraries(test_core_state_alloc PRIVATE m)
endif()

add_test(NAME test_core_state_alloc COMMAND test_core_state_alloc)
