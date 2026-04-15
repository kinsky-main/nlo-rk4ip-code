function(configure_vulkan_backend target target_source_dir target_binary_dir)
  include(ResolveVulkan)
  resolve_vulkan(vk_headers_available vk_loader_available)
  if(NOT vk_headers_available)
    message(FATAL_ERROR
      "Vulkan headers were not found and could not be fetched. "
      "Provide VULKAN_SDK/include or ensure network access for fetching Vulkan-Headers.")
  endif()
  if(NOT vk_loader_available)
    message(FATAL_ERROR
      "Vulkan loader library was not found. "
      "Install Vulkan loader/SDK (e.g. libvulkan-dev on Linux or LunarG Vulkan SDK on Windows).")
  endif()

  find_program(GLSLANG_VALIDATOR
    NAMES glslangValidator glslangValidator.exe
    HINTS
      ENV VULKAN_SDK
    PATH_SUFFIXES
      Bin
      bin
  )
  if(NOT GLSLANG_VALIDATOR)
    message(FATAL_ERROR
      "glslangValidator was not found. Install Vulkan SDK and ensure "
      "its Bin directory is on PATH (or VULKAN_SDK is set).")
  endif()

  set(VK_KERNEL_SOURCE_DIR "${target_source_dir}/backend/vulkan/kernels")
  set(VK_KERNEL_BINARY_DIR "${target_binary_dir}/backend/vulkan/kernels")
  file(MAKE_DIRECTORY "${VK_KERNEL_BINARY_DIR}")

  set(vk_kernel_names
    real_fill
    real_mul_inplace
    complex_fill
    complex_scalar_mul_inplace
    complex_add_inplace
    complex_mul_inplace
    complex_magnitude_squared
    complex_exp_inplace
    complex_real_pow_inplace
    complex_relative_error_reduce
    real_max_reduce
    complex_weighted_rms_reduce
    pair_sum_reduce
    complex_axis_unshifted_from_delta
    complex_axis_centered_from_delta
    complex_mesh_from_axis_tfast_t
    complex_mesh_from_axis_tfast_y
    complex_mesh_from_axis_tfast_x
  )

  set(vk_spv_outputs "")
  foreach(vk_kernel IN LISTS vk_kernel_names)
    set(vk_kernel_src "${VK_KERNEL_SOURCE_DIR}/${vk_kernel}.comp")
    set(vk_kernel_spv "${VK_KERNEL_BINARY_DIR}/${vk_kernel}.spv")
    add_custom_command(
      OUTPUT "${vk_kernel_spv}"
      COMMAND "${GLSLANG_VALIDATOR}"
        -V
        --target-env vulkan1.2
        -I"${VK_KERNEL_SOURCE_DIR}"
        -o "${vk_kernel_spv}"
        "${vk_kernel_src}"
      DEPENDS
        "${vk_kernel_src}"
        "${VK_KERNEL_SOURCE_DIR}/complex_device.glslinc"
        "${VK_KERNEL_SOURCE_DIR}/double_math.glslinc"
      COMMENT "Compiling Vulkan compute shader ${vk_kernel}.comp"
      VERBATIM
    )
    list(APPEND vk_spv_outputs "${vk_kernel_spv}")
  endforeach()

  add_custom_target(vk_shaders DEPENDS ${vk_spv_outputs})
  add_dependencies(${target} vk_shaders)

  set(VK_SHADER_REAL_FILL_PATH "${VK_KERNEL_BINARY_DIR}/real_fill.spv")
  set(VK_SHADER_REAL_MUL_INPLACE_PATH "${VK_KERNEL_BINARY_DIR}/real_mul_inplace.spv")
  set(VK_SHADER_COMPLEX_FILL_PATH "${VK_KERNEL_BINARY_DIR}/complex_fill.spv")
  set(VK_SHADER_COMPLEX_SCALAR_MUL_INPLACE_PATH "${VK_KERNEL_BINARY_DIR}/complex_scalar_mul_inplace.spv")
  set(VK_SHADER_COMPLEX_ADD_INPLACE_PATH "${VK_KERNEL_BINARY_DIR}/complex_add_inplace.spv")
  set(VK_SHADER_COMPLEX_MUL_INPLACE_PATH "${VK_KERNEL_BINARY_DIR}/complex_mul_inplace.spv")
  set(VK_SHADER_COMPLEX_MAGNITUDE_SQUARED_PATH "${VK_KERNEL_BINARY_DIR}/complex_magnitude_squared.spv")
  set(VK_SHADER_COMPLEX_EXP_INPLACE_PATH "${VK_KERNEL_BINARY_DIR}/complex_exp_inplace.spv")
  set(VK_SHADER_COMPLEX_REAL_POW_INPLACE_PATH "${VK_KERNEL_BINARY_DIR}/complex_real_pow_inplace.spv")
  set(VK_SHADER_COMPLEX_RELATIVE_ERROR_REDUCE_PATH "${VK_KERNEL_BINARY_DIR}/complex_relative_error_reduce.spv")
  set(VK_SHADER_REAL_MAX_REDUCE_PATH "${VK_KERNEL_BINARY_DIR}/real_max_reduce.spv")
  set(VK_SHADER_COMPLEX_WEIGHTED_RMS_REDUCE_PATH "${VK_KERNEL_BINARY_DIR}/complex_weighted_rms_reduce.spv")
  set(VK_SHADER_PAIR_SUM_REDUCE_PATH "${VK_KERNEL_BINARY_DIR}/pair_sum_reduce.spv")
  set(VK_SHADER_COMPLEX_AXIS_UNSHIFTED_FROM_DELTA_PATH "${VK_KERNEL_BINARY_DIR}/complex_axis_unshifted_from_delta.spv")
  set(VK_SHADER_COMPLEX_AXIS_CENTERED_FROM_DELTA_PATH "${VK_KERNEL_BINARY_DIR}/complex_axis_centered_from_delta.spv")
  set(VK_SHADER_COMPLEX_MESH_FROM_AXIS_TFAST_T_PATH "${VK_KERNEL_BINARY_DIR}/complex_mesh_from_axis_tfast_t.spv")
  set(VK_SHADER_COMPLEX_MESH_FROM_AXIS_TFAST_Y_PATH "${VK_KERNEL_BINARY_DIR}/complex_mesh_from_axis_tfast_y.spv")
  set(VK_SHADER_COMPLEX_MESH_FROM_AXIS_TFAST_X_PATH "${VK_KERNEL_BINARY_DIR}/complex_mesh_from_axis_tfast_x.spv")

  configure_file(
    "${target_source_dir}/backend/vulkan/vk_shader_paths.h.in"
    "${target_binary_dir}/generated/vk_shader_paths.h"
    @ONLY
  )

  target_include_directories(${target} PRIVATE "${target_binary_dir}/generated")
  target_link_libraries(${target} PUBLIC Vulkan::Headers Vulkan::Vulkan)
endfunction()
