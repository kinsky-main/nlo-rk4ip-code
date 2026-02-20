function(nlo_configure_build_configurations)
  set(_nlo_supported_configs Debug Release RelWithDebInfo MinSizeRel)

  set(CMAKE_CONFIGURATION_TYPES
    "${_nlo_supported_configs}"
    CACHE STRING "Supported build configurations"
    FORCE
  )

  if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "" CACHE STRING "Choose the build type.")
  endif()
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${_nlo_supported_configs})
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE
      Debug
      CACHE STRING "Choose the build type."
      FORCE
    )
  endif()
endfunction()
