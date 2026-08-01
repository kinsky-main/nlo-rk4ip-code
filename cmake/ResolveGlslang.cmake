# SPDX-License-Identifier: MIT
# ResolveGlslang.cmake - resolve glslang for VkFFT's Vulkan backend
#
# Provides:
#   resolve_glslang_for_vkfft(<out_target> <out_found>)
#
# The returned target exposes the include path that contains
# glslang_c_interface.h because upstream VkFFT includes that header directly.

include_guard(GLOBAL)

function(nlolib_find_glslang_c_include out_include_dir)
  find_path(_glslang_c_include_dir
    NAMES glslang_c_interface.h
    HINTS
      ENV VULKAN_SDK
    PATH_SUFFIXES
      Include/glslang/Include
      include/glslang/Include
      glslang/Include
  )

  set(${out_include_dir} "${_glslang_c_include_dir}" PARENT_SCOPE)
endfunction()

function(nlolib_define_pkg_config_glslang target_name out_found)
  find_package(PkgConfig QUIET)
  if(NOT PkgConfig_FOUND)
    set(${out_found} OFF PARENT_SCOPE)
    return()
  endif()

  pkg_check_modules(NLOLIB_GLSLANG_PC QUIET glslang spirv)
  if(NOT NLOLIB_GLSLANG_PC_FOUND)
    set(${out_found} OFF PARENT_SCOPE)
    return()
  endif()

  nlolib_find_glslang_c_include(_glslang_c_include_dir)
  if(NOT _glslang_c_include_dir)
    set(${out_found} OFF PARENT_SCOPE)
    return()
  endif()

  add_library(${target_name} INTERFACE)
  target_include_directories(${target_name} INTERFACE
    "${_glslang_c_include_dir}"
    ${NLOLIB_GLSLANG_PC_INCLUDE_DIRS}
  )
  target_compile_options(${target_name} INTERFACE
    ${NLOLIB_GLSLANG_PC_CFLAGS_OTHER}
  )
  target_link_directories(${target_name} INTERFACE
    ${NLOLIB_GLSLANG_PC_LIBRARY_DIRS}
  )
  target_link_libraries(${target_name} INTERFACE
    ${NLOLIB_GLSLANG_PC_LIBRARIES}
  )

  set(${out_found} ON PARENT_SCOPE)
endfunction()

function(nlolib_define_manual_glslang target_name out_found)
  nlolib_find_glslang_c_include(_glslang_c_include_dir)
  if(NOT _glslang_c_include_dir)
    set(${out_found} OFF PARENT_SCOPE)
    return()
  endif()

  set(_glslang_lib_names
    SPIRV
    SPIRV-Tools-opt
    SPIRV-Tools
    SPIRV-Tools-link
    glslang
    MachineIndependent
    OSDependent
    HLSL
    OGLCompiler
    GenericCodeGen
    SPVRemapper
  )
  set(_glslang_libraries "")
  foreach(_lib_name IN LISTS _glslang_lib_names)
    string(MAKE_C_IDENTIFIER "${_lib_name}" _lib_var_name)
    find_library(_glslang_${_lib_var_name}_library
      NAMES "${_lib_name}"
      HINTS
        ENV VULKAN_SDK
      PATH_SUFFIXES
        Lib
        lib
        lib64
    )
    if(NOT _glslang_${_lib_var_name}_library)
      set(${out_found} OFF PARENT_SCOPE)
      return()
    endif()
    list(APPEND _glslang_libraries "${_glslang_${_lib_var_name}_library}")
  endforeach()

  find_package(Threads QUIET)

  add_library(${target_name} INTERFACE)
  target_include_directories(${target_name} INTERFACE "${_glslang_c_include_dir}")
  target_link_libraries(${target_name} INTERFACE ${_glslang_libraries})
  if(Threads_FOUND)
    target_link_libraries(${target_name} INTERFACE Threads::Threads)
  else()
    target_link_libraries(${target_name} INTERFACE pthread)
  endif()

  set(${out_found} ON PARENT_SCOPE)
endfunction()

function(nlolib_define_fetched_glslang target_name out_found)
  include(FetchContent)

  set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "Build glslang binaries" FORCE)
  set(ENABLE_GLSLANG_JS OFF CACHE BOOL "Build glslang JavaScript output" FORCE)
  set(ENABLE_HLSL OFF CACHE BOOL "Build HLSL support in glslang" FORCE)
  set(ENABLE_OPT OFF CACHE BOOL "Build SPIR-V optimizer support in glslang" FORCE)
  set(GLSLANG_TESTS OFF CACHE BOOL "Build glslang tests" FORCE)

  FetchContent_Declare(
    glslang_main
    GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
    GIT_TAG "${GLSLANG_GIT_TAG}"
  )
  FetchContent_MakeAvailable(glslang_main)

  if(NOT TARGET glslang OR NOT TARGET SPIRV)
    set(${out_found} OFF PARENT_SCOPE)
    return()
  endif()

  add_library(${target_name} INTERFACE)
  target_include_directories(${target_name} INTERFACE
    "${glslang_main_SOURCE_DIR}/glslang/Include"
  )
  target_link_libraries(${target_name} INTERFACE SPIRV glslang)

  set(${out_found} ON PARENT_SCOPE)
endfunction()

function(resolve_glslang_for_vkfft out_target out_found)
  set(_target_name nlolib_vkfft_glslang)

  if(TARGET ${_target_name})
    set(${out_target} ${_target_name} PARENT_SCOPE)
    set(${out_found} ON PARENT_SCOPE)
    return()
  endif()

  string(TOUPPER "${NLOLIB_GLSLANG_PROVIDER}" _provider)
  if(NOT _provider MATCHES "^(AUTO|SYSTEM|FETCH)$")
    message(FATAL_ERROR
      "NLOLIB_GLSLANG_PROVIDER must be AUTO, SYSTEM, or FETCH "
      "(got '${NLOLIB_GLSLANG_PROVIDER}').")
  endif()

  set(_found OFF)

  if(NOT _provider STREQUAL "FETCH")
    find_package(Vulkan QUIET COMPONENTS glslang)
    if(TARGET Vulkan::glslang)
      nlolib_find_glslang_c_include(_glslang_c_include_dir)
      add_library(${_target_name} INTERFACE)
      target_link_libraries(${_target_name} INTERFACE Vulkan::glslang)
      if(_glslang_c_include_dir)
        target_include_directories(${_target_name} INTERFACE "${_glslang_c_include_dir}")
      endif()
      set(_found ON)
    endif()
  endif()

  if(NOT _found AND NOT _provider STREQUAL "FETCH")
    nlolib_define_pkg_config_glslang(${_target_name} _found)
  endif()

  if(NOT _found AND NOT _provider STREQUAL "FETCH")
    nlolib_define_manual_glslang(${_target_name} _found)
  endif()

  if(NOT _found AND NOT _provider STREQUAL "SYSTEM")
    nlolib_define_fetched_glslang(${_target_name} _found)
  endif()

  if(NOT _found)
    if(_provider STREQUAL "SYSTEM")
      message(FATAL_ERROR
        "VkFFT requires glslang development files, but system glslang was not found. "
        "Install glslang development libraries (Ubuntu: glslang-dev spirv-tools) "
        "or configure with -DNLOLIB_GLSLANG_PROVIDER=AUTO/FETCH.")
    else()
      message(FATAL_ERROR
        "VkFFT requires glslang development files and the fetch fallback failed. "
        "Install glslang development libraries (Ubuntu: glslang-dev spirv-tools) "
        "or set VULKAN_SDK to a Vulkan SDK with glslang.")
    endif()
  endif()

  set(${out_target} ${_target_name} PARENT_SCOPE)
  set(${out_found} ON PARENT_SCOPE)
endfunction()
