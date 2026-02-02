# Main library CMakeLists.txt

set(NLOLIB_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/nlolib.c
  ${CMAKE_CURRENT_LIST_DIR}/core/state.c
  ${CMAKE_CURRENT_LIST_DIR}/core/init.c
  ${CMAKE_CURRENT_LIST_DIR}/physics/operators.c
  ${CMAKE_CURRENT_LIST_DIR}/numerics/math_ops.c
  ${CMAKE_CURRENT_LIST_DIR}/numerics/vector_ops.c
  ${CMAKE_CURRENT_LIST_DIR}/fft/fft.c
)

add_library(nlolib SHARED ${NLOLIB_SOURCES})
set_target_properties(nlolib PROPERTIES OUTPUT_NAME nlolib)

target_include_directories(nlolib PUBLIC ${CMAKE_CURRENT_LIST_DIR})

if(WIN32)
  target_compile_definitions(nlolib PRIVATE NLOLIB_EXPORTS)
endif()

add_subdirectory(fft)
target_link_libraries(nlolib PUBLIC nlo_fft_backend simd::simd)

if(WIN32 AND NLO_FFT_BACKEND_FFTW AND FFTW3_LIBRARY)
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
