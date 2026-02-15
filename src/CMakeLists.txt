# Main library CMakeLists.txt

set(NLOLIB_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/nlolib.c
  ${CMAKE_CURRENT_LIST_DIR}/core/state.c
  ${CMAKE_CURRENT_LIST_DIR}/core/init.c
  ${CMAKE_CURRENT_LIST_DIR}/physics/operators.c
  ${CMAKE_CURRENT_LIST_DIR}/numerics/math_ops.c
  ${CMAKE_CURRENT_LIST_DIR}/numerics/vector_ops.c
  ${CMAKE_CURRENT_LIST_DIR}/numerics/rk4_kernel.c
  ${CMAKE_CURRENT_LIST_DIR}/utility/rk4_debug.c
  ${CMAKE_CURRENT_LIST_DIR}/backend/vector_backend.c
  ${CMAKE_CURRENT_LIST_DIR}/fft/fft.c
)

if(NLO_VECTOR_BACKEND_VULKAN)
  list(APPEND NLOLIB_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/numerics/vk_vector_ops.c
  )
endif()

add_library(nlolib SHARED ${NLOLIB_SOURCES})
set_target_properties(nlolib PROPERTIES OUTPUT_NAME nlolib)

target_include_directories(nlolib PUBLIC ${CMAKE_CURRENT_LIST_DIR})

if(WIN32)
  target_compile_definitions(nlolib PRIVATE NLOLIB_EXPORTS)
endif()

if(NLO_ENABLE_RK4_DEBUG_DIAGNOSTICS)
  target_compile_definitions(nlolib PRIVATE
    $<$<CONFIG:Debug>:NLO_ENABLE_RK4_DEBUG_DIAGNOSTICS=1>
  )
endif()

add_subdirectory(fft)
target_link_libraries(nlolib PUBLIC nlo_fft_backend simd::simd)

if(UNIX)
  target_link_libraries(nlolib PUBLIC m)
endif()

if(NLO_VECTOR_BACKEND_VULKAN)
  include(NLOVulkanBackend)
  nlo_configure_vulkan_backend(
    nlolib
    "${CMAKE_CURRENT_LIST_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}"
  )
endif()

if(WIN32 AND NLO_ENABLE_FFTW AND FFTW3_LIBRARY)
  get_filename_component(_FFTW3_DIR "${FFTW3_LIBRARY}" DIRECTORY)
  get_filename_component(_FFTW3_NAME "${FFTW3_LIBRARY}" NAME_WE)
  set(_FFTW3_DLL "${_FFTW3_DIR}/${_FFTW3_NAME}.dll")

  if(EXISTS "${_FFTW3_DLL}")
    add_custom_command(TARGET nlolib POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${_FFTW3_DLL}"
      "$<TARGET_FILE_DIR:nlolib>"
    )
  endif()
endif()

