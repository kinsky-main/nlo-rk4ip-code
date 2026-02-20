add_executable(runtime_temporal_demo_c
  ${CMAKE_CURRENT_LIST_DIR}/c/runtime_temporal_demo.c
)

target_link_libraries(runtime_temporal_demo_c PRIVATE nlolib)

if(WIN32)
  add_custom_command(TARGET runtime_temporal_demo_c POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:nlolib>
            $<TARGET_FILE_DIR:runtime_temporal_demo_c>
  )
endif()
