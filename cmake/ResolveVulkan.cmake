# SPDX-License-Identifier: MIT
# ResolveVulkan.cmake - resolve Vulkan headers and loader across platforms
#
# Provides:
#   nlo_resolve_vulkan(<out_headers_available> <out_loader_available>)
#
# Behavior:
#   1) Tries CMake's FindVulkan module first.
#   2) If headers are not found and NLO_VULKAN_FETCH_HEADERS=ON, fetches Vulkan-Headers.
#   3) Tries to locate Vulkan loader library if not provided by FindVulkan.

include_guard(GLOBAL)

function(_nlo_define_vulkan_headers_target include_dir)
  if(TARGET Vulkan::Headers)
    return()
  endif()

  if(NOT include_dir)
    return()
  endif()

  add_library(nlo_vulkan_headers INTERFACE)
  target_include_directories(nlo_vulkan_headers INTERFACE "${include_dir}")
  add_library(Vulkan::Headers ALIAS nlo_vulkan_headers)
endfunction()

function(_nlo_define_vulkan_loader_target loader_lib include_dir)
  if(TARGET Vulkan::Vulkan)
    return()
  endif()

  if(NOT loader_lib)
    return()
  endif()

  add_library(Vulkan::Vulkan UNKNOWN IMPORTED)
  set_target_properties(Vulkan::Vulkan PROPERTIES
    IMPORTED_LOCATION "${loader_lib}"
  )

  if(include_dir)
    set_target_properties(Vulkan::Vulkan PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${include_dir}"
    )
  endif()
endfunction()

function(nlo_resolve_vulkan out_headers_available out_loader_available)
  set(_headers_available OFF)
  set(_loader_available OFF)
  set(_header_include_dir "")

  find_package(Vulkan QUIET)

  if(Vulkan_FOUND)
    if(Vulkan_INCLUDE_DIRS)
      list(GET Vulkan_INCLUDE_DIRS 0 _header_include_dir)
    elseif(DEFINED Vulkan_INCLUDE_DIR)
      set(_header_include_dir "${Vulkan_INCLUDE_DIR}")
    endif()

    if(_header_include_dir)
      set(_headers_available ON)
      _nlo_define_vulkan_headers_target("${_header_include_dir}")
    endif()

    if(TARGET Vulkan::Vulkan)
      set(_loader_available ON)
    endif()
  endif()

  if(NOT _headers_available)
    find_path(_nlo_vulkan_header_dir
      NAMES vulkan/vulkan.h
      HINTS
        ENV VULKAN_SDK
      PATH_SUFFIXES
        include
    )

    if(_nlo_vulkan_header_dir)
      set(_header_include_dir "${_nlo_vulkan_header_dir}")
      set(_headers_available ON)
      _nlo_define_vulkan_headers_target("${_header_include_dir}")
    elseif(NLO_VULKAN_FETCH_HEADERS)
      include(FetchContent)
      FetchContent_Declare(
        vulkan_headers
        URL "${NLO_VULKAN_HEADERS_URL}"
      )
      FetchContent_MakeAvailable(vulkan_headers)

      if(EXISTS "${vulkan_headers_SOURCE_DIR}/include/vulkan/vulkan.h")
        set(_header_include_dir "${vulkan_headers_SOURCE_DIR}/include")
        set(_headers_available ON)
        _nlo_define_vulkan_headers_target("${_header_include_dir}")
      endif()
    endif()
  endif()

  if(NOT _loader_available)
    find_library(_nlo_vulkan_loader_lib
      NAMES
        vulkan-1
        vulkan
      HINTS
        ENV VULKAN_SDK
      PATH_SUFFIXES
        Lib
        lib
    )

    if(_nlo_vulkan_loader_lib)
      _nlo_define_vulkan_loader_target("${_nlo_vulkan_loader_lib}" "${_header_include_dir}")
      set(_loader_available ON)
    endif()
  endif()

  set(${out_headers_available} ${_headers_available} PARENT_SCOPE)
  set(${out_loader_available} ${_loader_available} PARENT_SCOPE)
endfunction()

