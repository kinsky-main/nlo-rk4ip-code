# Delay-load the Vulkan loader on Windows so vulkan-1.dll is only pulled in on
# the first Vulkan call, not when the library itself is loaded.  Without this a
# machine with no Vulkan runtime -- common for CPU-only use, and for the MATLAB
# toolbox in particular -- cannot load nlolib at all.  vk_auto_context.c probes
# for the loader before touching any entry point, so the delayed import is never
# triggered when it is absent.
function(nlolib_delay_load_vulkan target)
  if(NOT WIN32 OR NOT MSVC)
    return()
  endif()
  target_link_options(${target} PRIVATE "/DELAYLOAD:vulkan-1.dll")
  target_link_libraries(${target} PRIVATE delayimp)
endfunction()

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
    complex_affine_comb_real
  )

  set(vk_precision_flags "")
  if(NLOLIB_VULKAN_FP32)
    set(vk_precision_flags "-DNLO_VK_FP32=1")
  endif()

  set(vk_spv_outputs "")
  foreach(vk_kernel IN LISTS vk_kernel_names)
    set(vk_kernel_src "${VK_KERNEL_SOURCE_DIR}/nlo_${vk_kernel}.comp")
    set(vk_kernel_spv "${VK_KERNEL_BINARY_DIR}/${vk_kernel}.spv")
    add_custom_command(
      OUTPUT "${vk_kernel_spv}"
      COMMAND "${GLSLANG_VALIDATOR}"
        -V
        --target-env vulkan1.2
        ${vk_precision_flags}
        -I"${VK_KERNEL_SOURCE_DIR}"
        -o "${vk_kernel_spv}"
        "${vk_kernel_src}"
      DEPENDS
        "${vk_kernel_src}"
        "${VK_KERNEL_SOURCE_DIR}/nlo_precision.glslinc"
        "${VK_KERNEL_SOURCE_DIR}/complex_device.glslinc"
        "${VK_KERNEL_SOURCE_DIR}/nlo_complex_device.glslinc"
        "${VK_KERNEL_SOURCE_DIR}/double_math.glslinc"
        "${VK_KERNEL_SOURCE_DIR}/nlo_double_math.glslinc"
      COMMENT "Compiling Vulkan compute shader ${vk_kernel}.comp"
      VERBATIM
    )
    list(APPEND vk_spv_outputs "${vk_kernel_spv}")
  endforeach()

  add_custom_target(vk_shaders DEPENDS ${vk_spv_outputs})

  # Compile the SPIR-V modules into the library rather than loading them from
  # disk at runtime, so a packaged toolbox does not depend on the build tree.
  set(VK_SHADER_BLOB_DIR "${target_binary_dir}/generated")
  set(VK_SHADER_BLOB_SOURCE "${VK_SHADER_BLOB_DIR}/vk_shader_blobs.c")
  set(VK_SHADER_BLOB_HEADER "${VK_SHADER_BLOB_DIR}/vk_shader_blobs.h")
  file(MAKE_DIRECTORY "${VK_SHADER_BLOB_DIR}")

  add_custom_command(
    OUTPUT "${VK_SHADER_BLOB_SOURCE}" "${VK_SHADER_BLOB_HEADER}"
    COMMAND ${CMAKE_COMMAND}
      "-DSPV_NAMES=${vk_kernel_names}"
      "-DSPV_FILES=${vk_spv_outputs}"
      "-DOUTPUT_SOURCE=${VK_SHADER_BLOB_SOURCE}"
      "-DOUTPUT_HEADER=${VK_SHADER_BLOB_HEADER}"
      -P "${CMAKE_SOURCE_DIR}/cmake/embed_spirv.cmake"
    DEPENDS ${vk_spv_outputs} "${CMAKE_SOURCE_DIR}/cmake/embed_spirv.cmake"
    COMMENT "Embedding SPIR-V modules into vk_shader_blobs.c"
    VERBATIM
  )
  add_custom_target(vk_shader_blobs
    DEPENDS "${VK_SHADER_BLOB_SOURCE}" "${VK_SHADER_BLOB_HEADER}")
  add_dependencies(vk_shader_blobs vk_shaders)
  add_dependencies(${target} vk_shader_blobs)

  target_sources(${target} PRIVATE "${VK_SHADER_BLOB_SOURCE}")
  set_source_files_properties("${VK_SHADER_BLOB_SOURCE}" PROPERTIES GENERATED TRUE)

  target_include_directories(${target} PRIVATE "${VK_SHADER_BLOB_DIR}")
  target_link_libraries(${target} PUBLIC Vulkan::Headers Vulkan::Vulkan)
  nlolib_delay_load_vulkan(${target})

  # The operator JIT (numerics/vk_operator_jit.c) compiles GLSL at runtime and
  # #includes these helpers; embed them so installed builds can resolve them.
  set(VK_JIT_GLSL_FILES
    "${VK_KERNEL_SOURCE_DIR}/nlo_precision.glslinc"
    "${VK_KERNEL_SOURCE_DIR}/complex_device.glslinc"
    "${VK_KERNEL_SOURCE_DIR}/nlo_complex_device.glslinc"
    "${VK_KERNEL_SOURCE_DIR}/double_math.glslinc"
    "${VK_KERNEL_SOURCE_DIR}/nlo_double_math.glslinc"
    "${VK_KERNEL_SOURCE_DIR}/complex_math.glslinc"
  )
  set(VK_JIT_GLSL_SOURCE "${VK_SHADER_BLOB_DIR}/vk_jit_glsl_sources.c")
  set(VK_JIT_GLSL_HEADER "${VK_SHADER_BLOB_DIR}/vk_jit_glsl_sources.h")
  add_custom_command(
    OUTPUT "${VK_JIT_GLSL_SOURCE}" "${VK_JIT_GLSL_HEADER}"
    COMMAND ${CMAKE_COMMAND}
      "-DGLSL_FILES=${VK_JIT_GLSL_FILES}"
      "-DOUTPUT_SOURCE=${VK_JIT_GLSL_SOURCE}"
      "-DOUTPUT_HEADER=${VK_JIT_GLSL_HEADER}"
      -P "${CMAKE_SOURCE_DIR}/cmake/embed_glsl_sources.cmake"
    DEPENDS ${VK_JIT_GLSL_FILES} "${CMAKE_SOURCE_DIR}/cmake/embed_glsl_sources.cmake"
    COMMENT "Embedding GLSL includes for the Vulkan operator JIT"
    VERBATIM
  )
  add_custom_target(vk_jit_glsl_sources
    DEPENDS "${VK_JIT_GLSL_SOURCE}" "${VK_JIT_GLSL_HEADER}")
  add_dependencies(${target} vk_jit_glsl_sources)
  target_sources(${target} PRIVATE "${VK_JIT_GLSL_SOURCE}")
  set_source_files_properties("${VK_JIT_GLSL_SOURCE}" PROPERTIES GENERATED TRUE)
  nlolib_link_vulkan_jit_dependencies(${target})

  # Published so tests and benchmarks that compile the Vulkan backend sources
  # directly can pick up the same embedded shaders.
  set(NLOLIB_VK_SHADER_BLOB_DIR "${VK_SHADER_BLOB_DIR}"
      CACHE INTERNAL "Directory holding the generated SPIR-V blob sources" FORCE)
  set(NLOLIB_VK_SHADER_BLOB_SOURCE "${VK_SHADER_BLOB_SOURCE}"
      CACHE INTERNAL "Generated SPIR-V blob translation unit" FORCE)
  set(NLOLIB_VK_JIT_GLSL_SOURCE "${VK_JIT_GLSL_SOURCE}"
      CACHE INTERNAL "Generated GLSL include table for the operator JIT" FORCE)
endfunction()

# glslang (runtime GLSL->SPIR-V, shared with VkFFT) and a thread library for
# the JIT's process-wide SPIR-V cache lock.
function(nlolib_link_vulkan_jit_dependencies target)
  include(ResolveGlslang)
  resolve_glslang_for_vkfft(jit_glslang_target jit_glslang_found)
  find_package(Threads REQUIRED)
  target_link_libraries(${target} PRIVATE ${jit_glslang_target} Threads::Threads)
endfunction()

# Attach the embedded SPIR-V modules to a target that compiles the Vulkan
# backend sources itself (tests, benchmarks) rather than linking nlolib.
function(nlolib_use_embedded_spirv target)
  if(NOT TARGET vk_shader_blobs)
    return()
  endif()
  add_dependencies(${target} vk_shader_blobs)
  set_source_files_properties("${NLOLIB_VK_SHADER_BLOB_SOURCE}" PROPERTIES GENERATED TRUE)
  target_sources(${target} PRIVATE "${NLOLIB_VK_SHADER_BLOB_SOURCE}")
  # The operator JIT lives with the Vulkan backend sources these targets compile.
  add_dependencies(${target} vk_jit_glsl_sources)
  set_source_files_properties("${NLOLIB_VK_JIT_GLSL_SOURCE}" PROPERTIES GENERATED TRUE)
  target_sources(${target} PRIVATE
    "${NLOLIB_VK_JIT_GLSL_SOURCE}"
    "${CMAKE_SOURCE_DIR}/src/numerics/vk_operator_jit.c"
  )
  nlolib_link_vulkan_jit_dependencies(${target})
  target_include_directories(${target} PRIVATE "${NLOLIB_VK_SHADER_BLOB_DIR}")
endfunction()
