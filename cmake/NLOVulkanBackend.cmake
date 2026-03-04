function(nlo_configure_vulkan_backend target target_source_dir target_binary_dir)
  include(ResolveVulkan)
  nlo_resolve_vulkan(_nlo_vk_headers_available _nlo_vk_loader_available)
  if(NOT _nlo_vk_headers_available)
    message(FATAL_ERROR
      "Vulkan headers were not found and could not be fetched. "
      "Provide VULKAN_SDK/include or ensure network access for fetching Vulkan-Headers.")
  endif()
  if(NOT _nlo_vk_loader_available)
    message(FATAL_ERROR
      "Vulkan loader library was not found. "
      "Install Vulkan loader/SDK (e.g. libvulkan-dev on Linux or LunarG Vulkan SDK on Windows).")
  endif()

  find_program(NLO_GLSLANG_VALIDATOR
    NAMES glslangValidator glslangValidator.exe
    HINTS
      ENV VULKAN_SDK
    PATH_SUFFIXES
      Bin
      bin
  )
  if(NOT NLO_GLSLANG_VALIDATOR)
    message(FATAL_ERROR
      "glslangValidator was not found. Install Vulkan SDK and ensure "
      "its Bin directory is on PATH (or VULKAN_SDK is set).")
  endif()

  set(NLO_VK_KERNEL_SOURCE_DIR "${target_source_dir}/backend/vulkan/kernels")
  set(NLO_VK_KERNEL_BINARY_DIR "${target_binary_dir}/backend/vulkan/kernels")
  file(MAKE_DIRECTORY "${NLO_VK_KERNEL_BINARY_DIR}")

  set(_nlo_vk_kernel_names
    nlo_real_fill
    nlo_real_mul_inplace
    nlo_complex_fill
    nlo_complex_scalar_mul_inplace
    nlo_complex_add_inplace
    nlo_complex_mul_inplace
    nlo_complex_magnitude_squared
    nlo_complex_exp_inplace
    nlo_complex_real_pow_inplace
    nlo_complex_relative_error_reduce
    nlo_real_max_reduce
    nlo_complex_weighted_rms_reduce
    nlo_pair_sum_reduce
    nlo_complex_axis_unshifted_from_delta
    nlo_complex_axis_centered_from_delta
    nlo_complex_mesh_from_axis_tfast_t
    nlo_complex_mesh_from_axis_tfast_y
    nlo_complex_mesh_from_axis_tfast_x
  )

  set(_nlo_vk_spv_outputs "")
  foreach(_nlo_vk_kernel IN LISTS _nlo_vk_kernel_names)
    set(_nlo_vk_kernel_src "${NLO_VK_KERNEL_SOURCE_DIR}/${_nlo_vk_kernel}.comp")
    set(_nlo_vk_kernel_spv "${NLO_VK_KERNEL_BINARY_DIR}/${_nlo_vk_kernel}.spv")
    add_custom_command(
      OUTPUT "${_nlo_vk_kernel_spv}"
      COMMAND "${NLO_GLSLANG_VALIDATOR}"
        -V
        --target-env vulkan1.2
        -I"${NLO_VK_KERNEL_SOURCE_DIR}"
        -o "${_nlo_vk_kernel_spv}"
        "${_nlo_vk_kernel_src}"
      DEPENDS
        "${_nlo_vk_kernel_src}"
        "${NLO_VK_KERNEL_SOURCE_DIR}/nlo_complex_device.glslinc"
      COMMENT "Compiling Vulkan compute shader ${_nlo_vk_kernel}.comp"
      VERBATIM
    )
    list(APPEND _nlo_vk_spv_outputs "${_nlo_vk_kernel_spv}")
  endforeach()

  add_custom_target(nlo_vk_shaders DEPENDS ${_nlo_vk_spv_outputs})
  add_dependencies(${target} nlo_vk_shaders)

  set(NLO_VK_SHADER_REAL_FILL_PATH "${NLO_VK_KERNEL_BINARY_DIR}/nlo_real_fill.spv")
  set(NLO_VK_SHADER_REAL_MUL_INPLACE_PATH "${NLO_VK_KERNEL_BINARY_DIR}/nlo_real_mul_inplace.spv")
  set(NLO_VK_SHADER_COMPLEX_FILL_PATH "${NLO_VK_KERNEL_BINARY_DIR}/nlo_complex_fill.spv")
  set(NLO_VK_SHADER_COMPLEX_SCALAR_MUL_INPLACE_PATH "${NLO_VK_KERNEL_BINARY_DIR}/nlo_complex_scalar_mul_inplace.spv")
  set(NLO_VK_SHADER_COMPLEX_ADD_INPLACE_PATH "${NLO_VK_KERNEL_BINARY_DIR}/nlo_complex_add_inplace.spv")
  set(NLO_VK_SHADER_COMPLEX_MUL_INPLACE_PATH "${NLO_VK_KERNEL_BINARY_DIR}/nlo_complex_mul_inplace.spv")
  set(NLO_VK_SHADER_COMPLEX_MAGNITUDE_SQUARED_PATH "${NLO_VK_KERNEL_BINARY_DIR}/nlo_complex_magnitude_squared.spv")
  set(NLO_VK_SHADER_COMPLEX_EXP_INPLACE_PATH "${NLO_VK_KERNEL_BINARY_DIR}/nlo_complex_exp_inplace.spv")
  set(NLO_VK_SHADER_COMPLEX_REAL_POW_INPLACE_PATH "${NLO_VK_KERNEL_BINARY_DIR}/nlo_complex_real_pow_inplace.spv")
  set(NLO_VK_SHADER_COMPLEX_RELATIVE_ERROR_REDUCE_PATH "${NLO_VK_KERNEL_BINARY_DIR}/nlo_complex_relative_error_reduce.spv")
  set(NLO_VK_SHADER_REAL_MAX_REDUCE_PATH "${NLO_VK_KERNEL_BINARY_DIR}/nlo_real_max_reduce.spv")
  set(NLO_VK_SHADER_COMPLEX_WEIGHTED_RMS_REDUCE_PATH "${NLO_VK_KERNEL_BINARY_DIR}/nlo_complex_weighted_rms_reduce.spv")
  set(NLO_VK_SHADER_PAIR_SUM_REDUCE_PATH "${NLO_VK_KERNEL_BINARY_DIR}/nlo_pair_sum_reduce.spv")
  set(NLO_VK_SHADER_COMPLEX_AXIS_UNSHIFTED_FROM_DELTA_PATH "${NLO_VK_KERNEL_BINARY_DIR}/nlo_complex_axis_unshifted_from_delta.spv")
  set(NLO_VK_SHADER_COMPLEX_AXIS_CENTERED_FROM_DELTA_PATH "${NLO_VK_KERNEL_BINARY_DIR}/nlo_complex_axis_centered_from_delta.spv")
  set(NLO_VK_SHADER_COMPLEX_MESH_FROM_AXIS_TFAST_T_PATH "${NLO_VK_KERNEL_BINARY_DIR}/nlo_complex_mesh_from_axis_tfast_t.spv")
  set(NLO_VK_SHADER_COMPLEX_MESH_FROM_AXIS_TFAST_Y_PATH "${NLO_VK_KERNEL_BINARY_DIR}/nlo_complex_mesh_from_axis_tfast_y.spv")
  set(NLO_VK_SHADER_COMPLEX_MESH_FROM_AXIS_TFAST_X_PATH "${NLO_VK_KERNEL_BINARY_DIR}/nlo_complex_mesh_from_axis_tfast_x.spv")

  configure_file(
    "${target_source_dir}/backend/vulkan/nlo_vk_shader_paths.h.in"
    "${target_binary_dir}/generated/nlo_vk_shader_paths.h"
    @ONLY
  )

  target_include_directories(${target} PRIVATE "${target_binary_dir}/generated")
  target_link_libraries(${target} PUBLIC Vulkan::Headers Vulkan::Vulkan)
endfunction()
