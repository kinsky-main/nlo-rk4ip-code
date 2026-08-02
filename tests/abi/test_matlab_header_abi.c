/**
 * @file test_matlab_header_abi.c
 * @brief Compile-time proof that src/nlolib_matlab.h mirrors the real ABI.
 *
 * nlolib_matlab.h is a hand-maintained flattened copy of the public API,
 * written so MATLAB's loadlibrary() parser can read it (no vulkan/vulkan.h,
 * no function-pointer struct members). Because MATLAB marshals every call
 * using that header's layout, any drift between it and the canonical headers
 * silently corrupts arguments at run time instead of failing to build.
 *
 * This translation unit includes the mirror header with every one of its
 * type, enum-constant and function names macro-renamed to a `mirror_` prefix,
 * then includes the canonical headers, then asserts sizeof/offsetof parity
 * for every mirrored struct.
 *
 * A field added to core/state.h but not to nlolib_matlab.h (or vice versa)
 * becomes a build failure here naming the exact struct and field.
 */

#include <stddef.h>

/* --- 1. Rename every type the mirror header declares. ------------------- */
#define nlo_complex                  mirror_nlo_complex
#define propagation_params           mirror_propagation_params
#define time_grid                    mirror_time_grid
#define frequency_grid               mirror_frequency_grid
#define spatial_grid                 mirror_spatial_grid
#define tensor_layout                mirror_tensor_layout
#define tensor3d_desc                mirror_tensor3d_desc
#define nonlinear_model              mirror_nonlinear_model
#define runtime_operator_params      mirror_runtime_operator_params
#define simulation_config            mirror_simulation_config
#define physics_config               mirror_physics_config
#define sim_config                   mirror_sim_config
#define vector_backend_type          mirror_vector_backend_type
#define fft_backend_type             mirror_fft_backend_type
#define VkPhysicalDevice             mirror_VkPhysicalDevice
#define VkDevice                     mirror_VkDevice
#define VkQueue                      mirror_VkQueue
#define VkCommandPool                mirror_VkCommandPool
#define vk_backend_config            mirror_vk_backend_config
#define execution_options            mirror_execution_options
#define runtime_limits               mirror_runtime_limits
#define nlolib_status                mirror_nlolib_status
#define nlolib_log_level             mirror_nlolib_log_level
#define nlolib_progress_stream_mode  mirror_nlolib_progress_stream_mode
#define nlo_perf_profile_snapshot    mirror_nlo_perf_profile_snapshot
#define progress_event_type          mirror_progress_event_type
#define progress_info                mirror_progress_info
#define storage_db_cap_policy        mirror_storage_db_cap_policy
#define storage_options              mirror_storage_options
#define storage_result               mirror_storage_result
#define step_event                   mirror_step_event
#define propagate_output_mode        mirror_propagate_output_mode
#define propagate_options            mirror_propagate_options
#define propagate_output             mirror_propagate_output

/* --- 2. Rename the enum constants (they would otherwise redefine). ------ */
#define NLOLIB_STATUS_OK                   mirror_NLOLIB_STATUS_OK
#define NLOLIB_STATUS_INVALID_ARGUMENT     mirror_NLOLIB_STATUS_INVALID_ARGUMENT
#define NLOLIB_STATUS_ALLOCATION_FAILED    mirror_NLOLIB_STATUS_ALLOCATION_FAILED
#define NLOLIB_STATUS_NOT_IMPLEMENTED      mirror_NLOLIB_STATUS_NOT_IMPLEMENTED
#define NLOLIB_STATUS_ABORTED              mirror_NLOLIB_STATUS_ABORTED
#define NLOLIB_LOG_LEVEL_ERROR             mirror_NLOLIB_LOG_LEVEL_ERROR
#define NLOLIB_LOG_LEVEL_WARN              mirror_NLOLIB_LOG_LEVEL_WARN
#define NLOLIB_LOG_LEVEL_INFO              mirror_NLOLIB_LOG_LEVEL_INFO
#define NLOLIB_LOG_LEVEL_DEBUG             mirror_NLOLIB_LOG_LEVEL_DEBUG
#define NLOLIB_PROGRESS_STREAM_STDERR      mirror_NLOLIB_PROGRESS_STREAM_STDERR
#define NLOLIB_PROGRESS_STREAM_STDOUT      mirror_NLOLIB_PROGRESS_STREAM_STDOUT
#define NLOLIB_PROGRESS_STREAM_BOTH        mirror_NLOLIB_PROGRESS_STREAM_BOTH
#define TENSOR_LAYOUT_XYT_T_FAST           mirror_TENSOR_LAYOUT_XYT_T_FAST
#define NONLINEAR_MODEL_EXPR               mirror_NONLINEAR_MODEL_EXPR
#define NONLINEAR_MODEL_KERR_RAMAN         mirror_NONLINEAR_MODEL_KERR_RAMAN
#define VECTOR_BACKEND_CPU                 mirror_VECTOR_BACKEND_CPU
#define VECTOR_BACKEND_VULKAN              mirror_VECTOR_BACKEND_VULKAN
#define VECTOR_BACKEND_AUTO                mirror_VECTOR_BACKEND_AUTO
#define FFT_BACKEND_AUTO                   mirror_FFT_BACKEND_AUTO
#define FFT_BACKEND_FFTW                   mirror_FFT_BACKEND_FFTW
#define FFT_BACKEND_VKFFT                  mirror_FFT_BACKEND_VKFFT
#define PROGRESS_EVENT_ACCEPTED            mirror_PROGRESS_EVENT_ACCEPTED
#define PROGRESS_EVENT_REJECTED            mirror_PROGRESS_EVENT_REJECTED
#define PROGRESS_EVENT_FINISH              mirror_PROGRESS_EVENT_FINISH
#define STORAGE_DB_CAP_POLICY_STOP_WRITES  mirror_STORAGE_DB_CAP_POLICY_STOP_WRITES
#define STORAGE_DB_CAP_POLICY_FAIL         mirror_STORAGE_DB_CAP_POLICY_FAIL
#define PROPAGATE_OUTPUT_DENSE             mirror_PROPAGATE_OUTPUT_DENSE
#define PROPAGATE_OUTPUT_FINAL_ONLY        mirror_PROPAGATE_OUTPUT_FINAL_ONLY

/* --- 3. Rename the function declarations. ------------------------------ */
#define nlolib_propagate                mirror_nlolib_propagate
#define nlolib_query_runtime_limits     mirror_nlolib_query_runtime_limits
#define nlolib_perf_profile_set_enabled mirror_nlolib_perf_profile_set_enabled
#define nlolib_perf_profile_is_enabled  mirror_nlolib_perf_profile_is_enabled
#define nlolib_perf_profile_reset       mirror_nlolib_perf_profile_reset
#define nlolib_perf_profile_read        mirror_nlolib_perf_profile_read
#define nlolib_storage_is_available     mirror_nlolib_storage_is_available
#define nlolib_set_log_file             mirror_nlolib_set_log_file
#define nlolib_set_log_buffer           mirror_nlolib_set_log_buffer
#define nlolib_clear_log_buffer         mirror_nlolib_clear_log_buffer
#define nlolib_read_log_buffer          mirror_nlolib_read_log_buffer
#define nlolib_set_log_level            mirror_nlolib_set_log_level
#define nlolib_set_progress_options     mirror_nlolib_set_progress_options
#define nlolib_set_progress_stream      mirror_nlolib_set_progress_stream

#include "nlolib_matlab.h"

/* --- 4. Undo the renames so the canonical headers declare real names. --- */
#undef nlo_complex
#undef propagation_params
#undef time_grid
#undef frequency_grid
#undef spatial_grid
#undef tensor_layout
#undef tensor3d_desc
#undef nonlinear_model
#undef runtime_operator_params
#undef simulation_config
#undef physics_config
#undef sim_config
#undef vector_backend_type
#undef fft_backend_type
#undef VkPhysicalDevice
#undef VkDevice
#undef VkQueue
#undef VkCommandPool
#undef vk_backend_config
#undef execution_options
#undef runtime_limits
#undef nlolib_status
#undef nlolib_log_level
#undef nlolib_progress_stream_mode
#undef nlo_perf_profile_snapshot
#undef progress_event_type
#undef progress_info
#undef storage_db_cap_policy
#undef storage_options
#undef storage_result
#undef step_event
#undef propagate_output_mode
#undef propagate_options
#undef propagate_output

#undef NLOLIB_STATUS_OK
#undef NLOLIB_STATUS_INVALID_ARGUMENT
#undef NLOLIB_STATUS_ALLOCATION_FAILED
#undef NLOLIB_STATUS_NOT_IMPLEMENTED
#undef NLOLIB_STATUS_ABORTED
#undef NLOLIB_LOG_LEVEL_ERROR
#undef NLOLIB_LOG_LEVEL_WARN
#undef NLOLIB_LOG_LEVEL_INFO
#undef NLOLIB_LOG_LEVEL_DEBUG
#undef NLOLIB_PROGRESS_STREAM_STDERR
#undef NLOLIB_PROGRESS_STREAM_STDOUT
#undef NLOLIB_PROGRESS_STREAM_BOTH
#undef TENSOR_LAYOUT_XYT_T_FAST
#undef NONLINEAR_MODEL_EXPR
#undef NONLINEAR_MODEL_KERR_RAMAN
#undef VECTOR_BACKEND_CPU
#undef VECTOR_BACKEND_VULKAN
#undef VECTOR_BACKEND_AUTO
#undef FFT_BACKEND_AUTO
#undef FFT_BACKEND_FFTW
#undef FFT_BACKEND_VKFFT
#undef PROGRESS_EVENT_ACCEPTED
#undef PROGRESS_EVENT_REJECTED
#undef PROGRESS_EVENT_FINISH
#undef STORAGE_DB_CAP_POLICY_STOP_WRITES
#undef STORAGE_DB_CAP_POLICY_FAIL
#undef PROPAGATE_OUTPUT_DENSE
#undef PROPAGATE_OUTPUT_FINAL_ONLY

#undef nlolib_propagate
#undef nlolib_query_runtime_limits
#undef nlolib_perf_profile_set_enabled
#undef nlolib_perf_profile_is_enabled
#undef nlolib_perf_profile_reset
#undef nlolib_perf_profile_read
#undef nlolib_storage_is_available
#undef nlolib_set_log_file
#undef nlolib_set_log_buffer
#undef nlolib_clear_log_buffer
#undef nlolib_read_log_buffer
#undef nlolib_set_log_level
#undef nlolib_set_progress_options
#undef nlolib_set_progress_stream

/*
 * The mirror header defines these unconditionally while the canonical headers
 * guard them with #ifndef. Drop the mirror's values so the canonical ones
 * apply below -- otherwise a changed bound (e.g. constants[16] -> [32]) would
 * be masked and the array sizes would agree spuriously.
 */
#undef NT_MAX
#undef RUNTIME_OPERATOR_CONSTANTS_MAX
#undef STORAGE_RUN_ID_MAX

#include "nlolib.h"

/* --- 5. Parity assertions. --------------------------------------------- */
/* STATIC_ASSERT comes from backend/nlo_complex.h (C11 _Static_assert with a
 * C99 negative-array fallback). */

#define ASSERT_SIZE(T) \
    STATIC_ASSERT(sizeof(mirror_##T) == sizeof(T), \
                  "nlolib_matlab.h size mismatch: " #T)

/* Field name differs on the mirror side when it collides with a renamed type. */
#define ASSERT_OFFSET_AS(T, MIRROR_FIELD, FIELD) \
    STATIC_ASSERT(offsetof(mirror_##T, MIRROR_FIELD) == offsetof(T, FIELD), \
                  "nlolib_matlab.h offset mismatch: " #T "." #FIELD)

#define ASSERT_OFFSET(T, FIELD) ASSERT_OFFSET_AS(T, FIELD, FIELD)

ASSERT_SIZE(nlo_complex);
ASSERT_OFFSET(nlo_complex, re);
ASSERT_OFFSET(nlo_complex, im);

ASSERT_SIZE(propagation_params);
ASSERT_OFFSET(propagation_params, starting_step_size);
ASSERT_OFFSET(propagation_params, max_step_size);
ASSERT_OFFSET(propagation_params, min_step_size);
ASSERT_OFFSET(propagation_params, error_tolerance);
ASSERT_OFFSET(propagation_params, propagation_distance);

ASSERT_SIZE(time_grid);
ASSERT_OFFSET(time_grid, nt);
ASSERT_OFFSET(time_grid, pulse_period);
ASSERT_OFFSET(time_grid, delta_time);
ASSERT_OFFSET(time_grid, wt_axis);

ASSERT_SIZE(frequency_grid);
ASSERT_OFFSET_AS(frequency_grid, mirror_frequency_grid, frequency_grid);

ASSERT_SIZE(spatial_grid);
ASSERT_OFFSET(spatial_grid, nx);
ASSERT_OFFSET(spatial_grid, ny);
ASSERT_OFFSET(spatial_grid, delta_x);
ASSERT_OFFSET(spatial_grid, delta_y);
ASSERT_OFFSET(spatial_grid, spatial_frequency_grid);
ASSERT_OFFSET(spatial_grid, kx_axis);
ASSERT_OFFSET(spatial_grid, ky_axis);
ASSERT_OFFSET(spatial_grid, potential_grid);

ASSERT_SIZE(tensor3d_desc);
ASSERT_OFFSET(tensor3d_desc, nt);
ASSERT_OFFSET(tensor3d_desc, nx);
ASSERT_OFFSET(tensor3d_desc, ny);
ASSERT_OFFSET(tensor3d_desc, layout);

ASSERT_SIZE(runtime_operator_params);
ASSERT_OFFSET(runtime_operator_params, linear_factor_expr);
ASSERT_OFFSET(runtime_operator_params, linear_expr);
ASSERT_OFFSET(runtime_operator_params, potential_expr);
ASSERT_OFFSET(runtime_operator_params, dispersion_factor_expr);
ASSERT_OFFSET(runtime_operator_params, dispersion_expr);
ASSERT_OFFSET(runtime_operator_params, nonlinear_expr);
ASSERT_OFFSET_AS(runtime_operator_params, mirror_nonlinear_model, nonlinear_model);
ASSERT_OFFSET(runtime_operator_params, nonlinear_gamma);
ASSERT_OFFSET(runtime_operator_params, raman_fraction);
ASSERT_OFFSET(runtime_operator_params, raman_tau1);
ASSERT_OFFSET(runtime_operator_params, raman_tau2);
ASSERT_OFFSET(runtime_operator_params, shock_omega0);
ASSERT_OFFSET(runtime_operator_params, raman_response_time);
ASSERT_OFFSET(runtime_operator_params, raman_response_len);
ASSERT_OFFSET(runtime_operator_params, num_constants);
ASSERT_OFFSET(runtime_operator_params, constants);

ASSERT_SIZE(simulation_config);
ASSERT_OFFSET(simulation_config, propagation);
ASSERT_OFFSET(simulation_config, tensor);
ASSERT_OFFSET(simulation_config, time);
ASSERT_OFFSET(simulation_config, frequency);
ASSERT_OFFSET(simulation_config, spatial);

ASSERT_SIZE(sim_config);
ASSERT_OFFSET(sim_config, propagation);
ASSERT_OFFSET(sim_config, tensor);
ASSERT_OFFSET(sim_config, time);
ASSERT_OFFSET(sim_config, frequency);
ASSERT_OFFSET(sim_config, spatial);
ASSERT_OFFSET(sim_config, runtime);

ASSERT_SIZE(physics_config);

ASSERT_SIZE(vk_backend_config);
ASSERT_OFFSET(vk_backend_config, physical_device);
ASSERT_OFFSET(vk_backend_config, device);
ASSERT_OFFSET(vk_backend_config, queue);
ASSERT_OFFSET(vk_backend_config, queue_family_index);
ASSERT_OFFSET(vk_backend_config, command_pool);
ASSERT_OFFSET(vk_backend_config, descriptor_set_budget_bytes);
ASSERT_OFFSET(vk_backend_config, descriptor_set_count_override);

ASSERT_SIZE(execution_options);
ASSERT_OFFSET(execution_options, backend_type);
ASSERT_OFFSET(execution_options, fft_backend);
ASSERT_OFFSET(execution_options, device_heap_fraction);
ASSERT_OFFSET(execution_options, record_ring_target);
ASSERT_OFFSET(execution_options, forced_device_budget_bytes);
ASSERT_OFFSET(execution_options, vulkan);

ASSERT_SIZE(runtime_limits);
ASSERT_OFFSET(runtime_limits, max_num_time_samples_runtime);
ASSERT_OFFSET(runtime_limits, max_num_recorded_samples_in_memory);
ASSERT_OFFSET(runtime_limits, max_num_recorded_samples_with_storage);
ASSERT_OFFSET(runtime_limits, estimated_required_working_set_bytes);
ASSERT_OFFSET(runtime_limits, estimated_device_budget_bytes);
ASSERT_OFFSET(runtime_limits, storage_available);

ASSERT_SIZE(nlo_perf_profile_snapshot);
ASSERT_OFFSET(nlo_perf_profile_snapshot, dispersion_ms);
ASSERT_OFFSET(nlo_perf_profile_snapshot, nonlinear_ms);
ASSERT_OFFSET(nlo_perf_profile_snapshot, dispersion_calls);
ASSERT_OFFSET(nlo_perf_profile_snapshot, nonlinear_calls);
ASSERT_OFFSET(nlo_perf_profile_snapshot, gpu_dispatch_count);
ASSERT_OFFSET(nlo_perf_profile_snapshot, gpu_copy_count);
ASSERT_OFFSET(nlo_perf_profile_snapshot, gpu_device_copy_count);
ASSERT_OFFSET(nlo_perf_profile_snapshot, gpu_device_copy_bytes);
ASSERT_OFFSET(nlo_perf_profile_snapshot, gpu_host_transfer_copy_count);
ASSERT_OFFSET(nlo_perf_profile_snapshot, gpu_host_transfer_copy_bytes);
ASSERT_OFFSET(nlo_perf_profile_snapshot, gpu_memory_pass_count);
ASSERT_OFFSET(nlo_perf_profile_snapshot, gpu_memory_pass_bytes);
ASSERT_OFFSET(nlo_perf_profile_snapshot, gpu_upload_count);
ASSERT_OFFSET(nlo_perf_profile_snapshot, gpu_download_count);
ASSERT_OFFSET(nlo_perf_profile_snapshot, gpu_upload_bytes);
ASSERT_OFFSET(nlo_perf_profile_snapshot, gpu_download_bytes);

ASSERT_SIZE(progress_info);
ASSERT_OFFSET(progress_info, event_type);
ASSERT_OFFSET(progress_info, step_index);
ASSERT_OFFSET(progress_info, reject_attempt);
ASSERT_OFFSET(progress_info, z);
ASSERT_OFFSET(progress_info, z_end);
ASSERT_OFFSET(progress_info, percent);
ASSERT_OFFSET(progress_info, step_size);
ASSERT_OFFSET(progress_info, next_step_size);
ASSERT_OFFSET(progress_info, error);
ASSERT_OFFSET(progress_info, elapsed_seconds);
ASSERT_OFFSET(progress_info, eta_seconds);

ASSERT_SIZE(storage_options);
ASSERT_OFFSET(storage_options, sqlite_path);
ASSERT_OFFSET(storage_options, run_id);
ASSERT_OFFSET(storage_options, sqlite_max_bytes);
ASSERT_OFFSET(storage_options, chunk_records);
ASSERT_OFFSET(storage_options, cap_policy);
ASSERT_OFFSET(storage_options, log_final_output_field_to_db);

ASSERT_SIZE(storage_result);
ASSERT_OFFSET(storage_result, run_id);
ASSERT_OFFSET(storage_result, records_captured);
ASSERT_OFFSET(storage_result, records_spilled);
ASSERT_OFFSET(storage_result, chunks_written);
ASSERT_OFFSET(storage_result, db_size_bytes);
ASSERT_OFFSET(storage_result, truncated);

ASSERT_SIZE(step_event);
ASSERT_OFFSET(step_event, step_index);
ASSERT_OFFSET(step_event, z_current);
ASSERT_OFFSET(step_event, step_size);
ASSERT_OFFSET(step_event, next_step_size);
ASSERT_OFFSET(step_event, error);

ASSERT_SIZE(propagate_options);
ASSERT_OFFSET(propagate_options, num_recorded_samples);
ASSERT_OFFSET(propagate_options, output_mode);
ASSERT_OFFSET(propagate_options, return_records);
ASSERT_OFFSET(propagate_options, exec_options);
ASSERT_OFFSET_AS(propagate_options, mirror_storage_options, storage_options);
ASSERT_OFFSET(propagate_options, explicit_record_z);
ASSERT_OFFSET(propagate_options, explicit_record_z_count);
/* void* stands in for the progress_callback function pointer; see the note in
 * nlolib_matlab.h. This assertion is what makes that substitution safe. */
ASSERT_OFFSET(propagate_options, progress_callback);
STATIC_ASSERT(sizeof(((mirror_propagate_options *)0)->progress_callback) ==
                  sizeof(((propagate_options *)0)->progress_callback),
              "nlolib_matlab.h: void* is not layout-compatible with progress_callback");
ASSERT_OFFSET(propagate_options, progress_user_data);

ASSERT_SIZE(propagate_output);
ASSERT_OFFSET(propagate_output, output_records);
ASSERT_OFFSET(propagate_output, output_record_capacity);
ASSERT_OFFSET(propagate_output, records_written);
ASSERT_OFFSET_AS(propagate_output, mirror_storage_result, storage_result);
ASSERT_OFFSET(propagate_output, output_step_events);
ASSERT_OFFSET(propagate_output, output_step_event_capacity);
ASSERT_OFFSET(propagate_output, step_events_written);
ASSERT_OFFSET(propagate_output, step_events_dropped);

/* The MATLAB mirror deliberately types output_records as double* (interleaved
 * re/im) rather than nlo_complex*, so the wrapper can hand loadlibrary a
 * doublePtr. Only the pointee differs; the pointer itself must still match. */
STATIC_ASSERT(sizeof(((mirror_propagate_output *)0)->output_records) ==
                  sizeof(((propagate_output *)0)->output_records),
              "nlolib_matlab.h: output_records pointer size mismatch");

/* Bounds baked into struct layout. */
STATIC_ASSERT(sizeof(((mirror_storage_result *)0)->run_id) ==
                  sizeof(((storage_result *)0)->run_id),
              "nlolib_matlab.h: STORAGE_RUN_ID_MAX drift");
STATIC_ASSERT(sizeof(((mirror_runtime_operator_params *)0)->constants) ==
                  sizeof(((runtime_operator_params *)0)->constants),
              "nlolib_matlab.h: RUNTIME_OPERATOR_CONSTANTS_MAX drift");

int main(void)
{
    /* All checking happens at compile time. */
    return 0;
}
