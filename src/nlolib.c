/**
 * @file nlolib.c
 * @dir src
 * @brief Public API entry points for NLOLib.
 * @author Wenzel Kinsky
 * @date 2026-01-29
 */

#include "nlolib.h"
#include "backend/nlo_complex.h"
#include "core/sim_dimensions_internal.h"
#include "io/log_format.h"
#include "io/log_sink.h"
#include "io/snapshot_store.h"
#include "numerics/rk4_kernel.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static const char* nlo_backend_type_to_string(nlo_vector_backend_type backend_type)
{
    if (backend_type == NLO_VECTOR_BACKEND_CPU) {
        return "CPU";
    }
    if (backend_type == NLO_VECTOR_BACKEND_VULKAN) {
        return "VULKAN";
    }
    if (backend_type == NLO_VECTOR_BACKEND_AUTO) {
        return "AUTO";
    }

    return "UNKNOWN";
}

static nlolib_status nlo_propagate_fail(const char* stage, nlolib_status status)
{
    nlo_log_emit(NLO_LOG_LEVEL_ERROR,
                 "[nlolib] propagate failed:\n"
                 "  - stage: %s\n"
                 "  - status: %d",
                 (stage != NULL) ? stage : "unknown",
                 (int)status);
    return status;
}

static nlolib_status nlo_map_log_status(int status)
{
    if (status == 0) {
        return NLOLIB_STATUS_OK;
    }
    if (status == 1) {
        return NLOLIB_STATUS_INVALID_ARGUMENT;
    }
    if (status == 2) {
        return NLOLIB_STATUS_ALLOCATION_FAILED;
    }

    return NLOLIB_STATUS_INVALID_ARGUMENT;
}

static size_t nlo_compute_input_bytes(size_t count, size_t stride)
{
    if (stride == 0u || count > (SIZE_MAX / stride)) {
        return 0u;
    }

    return count * stride;
}

static size_t nlo_compute_record_bytes(size_t num_recorded_samples, size_t num_time_samples)
{
    const size_t per_record_bytes = nlo_compute_input_bytes(num_time_samples, sizeof(nlo_complex));
    if (per_record_bytes == 0u || num_recorded_samples > (SIZE_MAX / per_record_bytes)) {
        return 0u;
    }

    return per_record_bytes * num_recorded_samples;
}

static int nlo_storage_enabled(const nlo_storage_options* storage_options)
{
    return (storage_options != NULL &&
            storage_options->sqlite_path != NULL &&
            storage_options->sqlite_path[0] != '\0');
}

static int nlo_storage_log_final_output_enabled(const nlo_storage_options* storage_options)
{
    return (storage_options != NULL && storage_options->log_final_output_field_to_db != 0);
}

static sim_config nlo_merge_simulation_and_physics(
    const nlo_simulation_config* simulation_config,
    const nlo_physics_config* physics_config
)
{
    sim_config merged;
    memset(&merged, 0, sizeof(merged));
    if (simulation_config != NULL) {
        merged.propagation = simulation_config->propagation;
        merged.time = simulation_config->time;
        merged.frequency = simulation_config->frequency;
        merged.spatial = simulation_config->spatial;
    }
    if (physics_config != NULL) {
        merged.runtime = *physics_config;
    }
    return merged;
}

NLOLIB_API nlolib_status nlolib_query_runtime_limits(
    const nlo_simulation_config* simulation_config,
    const nlo_physics_config* physics_config,
    const nlo_execution_options* exec_options,
    nlo_runtime_limits* out_limits
)
{
    if (out_limits == NULL) {
        return nlo_propagate_fail("validate.runtime_limits.null_out", NLOLIB_STATUS_INVALID_ARGUMENT);
    }

    const sim_config merged = nlo_merge_simulation_and_physics(simulation_config, physics_config);
    if (nlo_query_runtime_limits_internal(&merged, exec_options, out_limits) != 0) {
        return nlo_propagate_fail("query_runtime_limits_internal", NLOLIB_STATUS_ALLOCATION_FAILED);
    }

    return NLOLIB_STATUS_OK;
}

static void nlo_log_nlse_propagate_call(
    const sim_config* config,
    size_t num_time_samples,
    const nlo_complex* input_field,
    size_t num_recorded_samples,
    nlo_complex* output_records,
    const nlo_execution_options* exec_options
)
{
    const nlo_execution_options local_exec_options =
        (exec_options != NULL)
            ? *exec_options
            : nlo_execution_options_default(NLO_VECTOR_BACKEND_AUTO);

    const size_t field_bytes = nlo_compute_input_bytes(num_time_samples, sizeof(nlo_complex));
    const size_t records_bytes = nlo_compute_record_bytes(num_recorded_samples, num_time_samples);
    size_t nt = 0u;
    size_t nx = 0u;
    size_t ny = 0u;
    int explicit_nd = 0;
    const int has_spatial_shape =
        (nlo_resolve_sim_dimensions_internal(config, num_time_samples, &nt, &nx, &ny, &explicit_nd) == 0);
    size_t frequency_grid_samples = num_time_samples;
    if (has_spatial_shape && nt > 0u) {
        frequency_grid_samples = nt;
    }
    const size_t frequency_grid_bytes = nlo_compute_input_bytes(frequency_grid_samples, sizeof(nlo_complex));

    char num_time_samples_text[48];
    char num_recorded_samples_text[48];
    char field_bytes_text[48];
    char records_bytes_text[48];
    char field_size_text[32];
    char records_size_text[32];
    char frequency_size_text[32];
    char frequency_bytes_text[48];
    char runtime_constants_text[48];
    char nt_text[48];
    char nx_text[48];
    char ny_text[48];

    (void)nlo_log_format_u64_grouped(num_time_samples_text,
                                     sizeof(num_time_samples_text),
                                     (uint64_t)num_time_samples);
    (void)nlo_log_format_u64_grouped(num_recorded_samples_text,
                                     sizeof(num_recorded_samples_text),
                                     (uint64_t)num_recorded_samples);
    (void)nlo_log_format_u64_grouped(field_bytes_text, sizeof(field_bytes_text), (uint64_t)field_bytes);
    (void)nlo_log_format_u64_grouped(records_bytes_text, sizeof(records_bytes_text), (uint64_t)records_bytes);
    (void)nlo_log_format_u64_grouped(runtime_constants_text,
                                     sizeof(runtime_constants_text),
                                     (uint64_t)((config != NULL) ? config->runtime.num_constants : 0u));
    (void)nlo_log_format_u64_grouped(frequency_bytes_text,
                                     sizeof(frequency_bytes_text),
                                     (uint64_t)frequency_grid_bytes);
    (void)nlo_log_format_u64_grouped(nt_text, sizeof(nt_text), (uint64_t)nt);
    (void)nlo_log_format_u64_grouped(nx_text, sizeof(nx_text), (uint64_t)nx);
    (void)nlo_log_format_u64_grouped(ny_text, sizeof(ny_text), (uint64_t)ny);
    (void)nlo_log_format_bytes_human(field_size_text, sizeof(field_size_text), field_bytes);
    (void)nlo_log_format_bytes_human(records_size_text, sizeof(records_size_text), records_bytes);
    (void)nlo_log_format_bytes_human(frequency_size_text, sizeof(frequency_size_text), frequency_grid_bytes);

    char constants_lines[768];
    size_t constants_len = 0u;
    constants_lines[0] = '\0';
    if (config != NULL && config->runtime.num_constants > 0u) {
        const size_t max_constants =
            (config->runtime.num_constants < NLO_RUNTIME_OPERATOR_CONSTANTS_MAX)
                ? config->runtime.num_constants
                : NLO_RUNTIME_OPERATOR_CONSTANTS_MAX;
        for (size_t idx = 0u; idx < max_constants; ++idx) {
            const int written = snprintf(constants_lines + constants_len,
                                         sizeof(constants_lines) - constants_len,
                                         "    - c%zu: %.9e\n",
                                         idx,
                                         config->runtime.constants[idx]);
            if (written < 0) {
                break;
            }
            const size_t add = (size_t)written;
            if (add >= (sizeof(constants_lines) - constants_len)) {
                constants_len = sizeof(constants_lines) - 1u;
                break;
            }
            constants_len += add;
        }
    } else {
        (void)snprintf(constants_lines, sizeof(constants_lines), "    - (none)\n");
    }

    char message[4096];
    const int written = snprintf(
        message,
        sizeof(message),
        "[nlolib] propagate request:\n"
        "  - backend_requested: %s\n"
        "  - num_time_samples: %s\n"
        "  - num_recorded_samples: %s\n"
        "  - field_size: %s (%s B)\n"
        "  - records_size: %s (%s B)\n"
        "  - pointers:\n"
        "    - config: %p\n"
        "    - input_field: %p\n"
        "    - output_records: %p\n"
        "    - exec_options: %p\n"
        "  - runtime_expressions:\n"
        "    - dispersion_factor_expr: %s\n"
        "    - dispersion_expr: %s\n"
        "    - transverse_factor_expr: %s\n"
        "    - transverse_expr: %s\n"
        "    - nonlinear_expr: %s\n"
        "  - runtime_constants (%s):\n"
        "%s"
        "  - grids:\n"
        "    - frequency_grid: %p (%s)\n"
        "    - frequency_grid_bytes: %s B\n"
        "    - spatial_dimensions: nt=%s nx=%s ny=%s\n"
        "    - explicit_nd: %d\n"
        "    - dimensions_valid: %d\n"
        "    - delta_x: %.9e\n"
        "    - delta_y: %.9e\n"
        "    - spatial_frequency_grid: %p\n"
        "    - potential_grid: %p\n",
        nlo_backend_type_to_string(local_exec_options.backend_type),
        num_time_samples_text,
        num_recorded_samples_text,
        field_size_text,
        field_bytes_text,
        records_size_text,
        records_bytes_text,
        (const void*)config,
        (const void*)input_field,
        (const void*)output_records,
        (const void*)exec_options,
        (config != NULL && config->runtime.dispersion_factor_expr != NULL)
            ? config->runtime.dispersion_factor_expr
            : "(null)",
        (config != NULL && config->runtime.dispersion_expr != NULL) ? config->runtime.dispersion_expr : "(null)",
        (config != NULL && config->runtime.transverse_factor_expr != NULL)
            ? config->runtime.transverse_factor_expr
            : "(null)",
        (config != NULL && config->runtime.transverse_expr != NULL) ? config->runtime.transverse_expr : "(null)",
        (config != NULL && config->runtime.nonlinear_expr != NULL) ? config->runtime.nonlinear_expr : "(null)",
        runtime_constants_text,
        constants_lines,
        (config != NULL) ? (const void*)config->frequency.frequency_grid : NULL,
        frequency_size_text,
        frequency_bytes_text,
        nt_text,
        nx_text,
        ny_text,
        explicit_nd,
        has_spatial_shape,
        (config != NULL) ? config->spatial.delta_x : 0.0,
        (config != NULL) ? config->spatial.delta_y : 0.0,
        (config != NULL) ? (const void*)config->spatial.spatial_frequency_grid : NULL,
        (config != NULL) ? (const void*)config->spatial.potential_grid : NULL);

    if (written > 0) {
        const size_t length = (size_t)((written < (int)sizeof(message)) ? written : (int)(sizeof(message) - 1u));
        nlo_log_emit_raw(NLO_LOG_LEVEL_INFO, message, length);
    }
}

NLOLIB_API nlo_propagate_options nlolib_propagate_options_default(void)
{
    nlo_propagate_options options;
    options.num_recorded_samples = 2u;
    options.output_mode = NLO_PROPAGATE_OUTPUT_DENSE;
    options.return_records = 1;
    options.exec_options = NULL;
    options.storage_options = NULL;
    return options;
}

NLOLIB_API nlo_propagate_output nlolib_propagate_output_default(void)
{
    nlo_propagate_output output;
    output.output_records = NULL;
    output.output_record_capacity = 0u;
    output.records_written = NULL;
    output.storage_result = NULL;
    return output;
}

NLOLIB_API nlolib_status nlolib_propagate(
    const nlo_simulation_config* simulation_config,
    const nlo_physics_config* physics_config,
    size_t num_time_samples,
    const nlo_complex* input_field,
    const nlo_propagate_options* options,
    nlo_propagate_output* output
)
{
    const sim_config merged = nlo_merge_simulation_and_physics(simulation_config, physics_config);
    const sim_config* config = &merged;
    const nlo_propagate_options local_options =
        (options != NULL) ? *options : nlolib_propagate_options_default();
    nlo_propagate_output local_output = (output != NULL) ? *output : nlolib_propagate_output_default();

    size_t num_recorded_samples = local_options.num_recorded_samples;
    if (local_options.output_mode == NLO_PROPAGATE_OUTPUT_FINAL_ONLY) {
        num_recorded_samples = 1u;
    }
    if (num_recorded_samples == 0u) {
        return nlo_propagate_fail("validate.num_recorded_samples", NLOLIB_STATUS_INVALID_ARGUMENT);
    }

    if (local_options.return_records == 0) {
        local_output.output_records = NULL;
        local_output.output_record_capacity = 0u;
    }

    if (local_output.records_written != NULL) {
        *local_output.records_written = 0u;
    }
    if (local_output.storage_result != NULL) {
        *local_output.storage_result = (nlo_storage_result){0};
    }

    nlo_log_nlse_propagate_call(config,
                                num_time_samples,
                                input_field,
                                num_recorded_samples,
                                local_output.output_records,
                                local_options.exec_options);

    if (config == NULL || input_field == NULL) {
        return nlo_propagate_fail("validate.null_pointer", NLOLIB_STATUS_INVALID_ARGUMENT);
    }
    if (local_options.return_records == 0 && !nlo_storage_enabled(local_options.storage_options)) {
        return nlo_propagate_fail("validate.output_or_storage", NLOLIB_STATUS_INVALID_ARGUMENT);
    }
    if (local_options.return_records != 0 && local_output.output_records == NULL) {
        return nlo_propagate_fail("validate.output_or_storage", NLOLIB_STATUS_INVALID_ARGUMENT);
    }

    if (num_time_samples == 0u) {
        return nlo_propagate_fail("validate.num_time_samples", NLOLIB_STATUS_INVALID_ARGUMENT);
    }
    if (num_recorded_samples == 0u) {
        return nlo_propagate_fail("validate.num_recorded_samples", NLOLIB_STATUS_INVALID_ARGUMENT);
    }
    if (num_recorded_samples >
        nlo_runtime_limits_default().max_num_recorded_samples_with_storage) {
        return nlo_propagate_fail("validate.num_recorded_samples_precision", NLOLIB_STATUS_INVALID_ARGUMENT);
    }
    if (local_options.return_records != 0) {
        if (local_output.output_record_capacity < num_recorded_samples) {
            return nlo_propagate_fail("validate.output_record_capacity", NLOLIB_STATUS_INVALID_ARGUMENT);
        }
        if (nlo_compute_record_bytes(num_recorded_samples, num_time_samples) == 0u) {
            return nlo_propagate_fail("validate.record_bytes", NLOLIB_STATUS_INVALID_ARGUMENT);
        }
    }
    if (nlo_storage_enabled(local_options.storage_options) && !nlo_snapshot_store_is_available()) {
        return nlo_propagate_fail("validate.storage_unavailable", NLOLIB_STATUS_NOT_IMPLEMENTED);
    }
    {
        size_t nt = 0u;
        size_t nx = 0u;
        size_t ny = 0u;
        int explicit_nd = 0;
        if (nlo_resolve_sim_dimensions_internal(config,
                                                num_time_samples,
                                                &nt,
                                                &nx,
                                                &ny,
                                                &explicit_nd) != 0) {
            return nlo_propagate_fail("validate.spatial_dimensions", NLOLIB_STATUS_INVALID_ARGUMENT);
        }
    }

    nlo_execution_options local_exec_options =
        (local_options.exec_options != NULL)
            ? *local_options.exec_options
            : nlo_execution_options_default(NLO_VECTOR_BACKEND_AUTO);
    simulation_state* state = NULL;
    const int init_status =
        nlo_storage_enabled(local_options.storage_options)
            ? nlo_init_simulation_state_with_storage(config,
                                                     num_time_samples,
                                                     num_recorded_samples,
                                                     &local_exec_options,
                                                     local_options.storage_options,
                                                     NULL,
                                                     &state)
            : nlo_init_simulation_state(config,
                                        num_time_samples,
                                        num_recorded_samples,
                                        &local_exec_options,
                                        NULL,
                                        &state);
    if (init_status != 0 || state == NULL) {
        return nlo_propagate_fail("init_simulation_state", NLOLIB_STATUS_ALLOCATION_FAILED);
    }

    nlo_log_emit(NLO_LOG_LEVEL_INFO,
                 "[nlolib] backend resolved:\n"
                 "  - requested: %s\n"
                 "  - actual: %s",
                 nlo_backend_type_to_string(local_exec_options.backend_type),
                 nlo_backend_type_to_string(nlo_vector_backend_get_type(state->backend)));

    if (simulation_state_upload_initial_field(state, input_field) != NLO_VEC_STATUS_OK) {
        free_simulation_state(state);
        return nlo_propagate_fail("upload_initial_field", NLOLIB_STATUS_ALLOCATION_FAILED);
    }

    solve_rk4(state);

    if (state->snapshot_status != NLO_VEC_STATUS_OK) {
        free_simulation_state(state);
        return nlo_propagate_fail("snapshot_capture", NLOLIB_STATUS_ALLOCATION_FAILED);
    }

    int final_downloaded = 0;
    const nlo_complex* final_output_field_cached = NULL;
    if (num_recorded_samples == 1u && state->snapshot_store != NULL) {
        nlo_complex* final_record = local_output.output_records;
        if (final_record == NULL) {
            final_record = simulation_state_get_field_record(state, 0u);
            if (final_record == NULL) {
                final_record = state->snapshot_scratch_record;
            }
        }
        if (final_record == NULL) {
            free_simulation_state(state);
            return nlo_propagate_fail("storage.final_record_buffer", NLOLIB_STATUS_ALLOCATION_FAILED);
        }

        if (simulation_state_download_current_field(state, final_record) != NLO_VEC_STATUS_OK) {
            free_simulation_state(state);
            return nlo_propagate_fail("storage.download_final_field", NLOLIB_STATUS_ALLOCATION_FAILED);
        }
        final_output_field_cached = final_record;
        final_downloaded = (final_record == local_output.output_records) ? 1 : 0;

        const nlo_snapshot_store_status write_status =
            nlo_snapshot_store_write_record(state->snapshot_store,
                                            0u,
                                            final_record,
                                            state->num_time_samples);
        if (write_status == NLO_SNAPSHOT_STORE_STATUS_ERROR) {
            free_simulation_state(state);
            return nlo_propagate_fail("storage.write_final_field", NLOLIB_STATUS_ALLOCATION_FAILED);
        }
        if (nlo_snapshot_store_flush(state->snapshot_store) == NLO_SNAPSHOT_STORE_STATUS_ERROR) {
            free_simulation_state(state);
            return nlo_propagate_fail("storage.flush_final_field", NLOLIB_STATUS_ALLOCATION_FAILED);
        }
        nlo_snapshot_store_get_result(state->snapshot_store, &state->snapshot_result);
    }

    if (local_options.return_records != 0 &&
        local_output.output_records != NULL &&
        num_recorded_samples == 1u) {
        if (!final_downloaded &&
            simulation_state_download_current_field(state, local_output.output_records) != NLO_VEC_STATUS_OK) {
            free_simulation_state(state);
            return nlo_propagate_fail("download_current_field", NLOLIB_STATUS_ALLOCATION_FAILED);
        }
        final_output_field_cached = local_output.output_records;
    } else if (local_options.return_records != 0 && local_output.output_records != NULL) {
        if (state->num_host_records != num_recorded_samples || state->field_buffer == NULL) {
            free_simulation_state(state);
            return nlo_propagate_fail("validate.host_record_buffer", NLOLIB_STATUS_ALLOCATION_FAILED);
        }

        const size_t records_bytes = nlo_compute_record_bytes(num_recorded_samples, num_time_samples);
        if (records_bytes == 0u) {
            free_simulation_state(state);
            return nlo_propagate_fail("validate.output_record_bytes", NLOLIB_STATUS_ALLOCATION_FAILED);
        }
        memcpy(local_output.output_records, state->field_buffer, records_bytes);
        final_output_field_cached =
            local_output.output_records + ((num_recorded_samples - 1u) * state->num_time_samples);
    }

    if (state->snapshot_store != NULL &&
        nlo_storage_log_final_output_enabled(local_options.storage_options)) {
        const nlo_complex* final_output_field = final_output_field_cached;

        nlo_complex* final_output_scratch = NULL;
        if (final_output_field == NULL) {
            final_output_scratch = simulation_state_get_field_record(state, num_recorded_samples - 1u);
            if (final_output_scratch == NULL) {
                final_output_scratch = state->snapshot_scratch_record;
            }
            if (final_output_scratch == NULL) {
                free_simulation_state(state);
                return nlo_propagate_fail("storage.final_output_buffer", NLOLIB_STATUS_ALLOCATION_FAILED);
            }
            if (simulation_state_download_current_field(state, final_output_scratch) != NLO_VEC_STATUS_OK) {
                free_simulation_state(state);
                return nlo_propagate_fail("storage.download_output_field", NLOLIB_STATUS_ALLOCATION_FAILED);
            }
            final_output_field = final_output_scratch;
        }

        if (nlo_snapshot_store_write_final_output_field(state->snapshot_store,
                                                        final_output_field,
                                                        state->num_time_samples) ==
            NLO_SNAPSHOT_STORE_STATUS_ERROR) {
            free_simulation_state(state);
            return nlo_propagate_fail("storage.write_output_field", NLOLIB_STATUS_ALLOCATION_FAILED);
        }
    }

    if (local_output.storage_result != NULL) {
        if (state->snapshot_store != NULL) {
            nlo_snapshot_store_get_result(state->snapshot_store, local_output.storage_result);
        } else {
            *local_output.storage_result = (nlo_storage_result){0};
        }
    }

    free_simulation_state(state);
    if (local_output.records_written != NULL) {
        *local_output.records_written = (local_options.return_records != 0) ? num_recorded_samples : 0u;
    }
    return NLOLIB_STATUS_OK;
}

NLOLIB_API int nlolib_storage_is_available(void)
{
    return nlo_snapshot_store_is_available();
}

NLOLIB_API nlolib_status nlolib_set_log_file(const char* path_utf8, int append)
{
    return nlo_map_log_status(nlo_log_set_file(path_utf8, append));
}

NLOLIB_API nlolib_status nlolib_set_log_buffer(size_t capacity_bytes)
{
    return nlo_map_log_status(nlo_log_set_buffer(capacity_bytes));
}

NLOLIB_API nlolib_status nlolib_clear_log_buffer(void)
{
    return nlo_map_log_status(nlo_log_clear_buffer());
}

NLOLIB_API nlolib_status nlolib_read_log_buffer(
    char* dst,
    size_t dst_bytes,
    size_t* out_written,
    int consume
)
{
    return nlo_map_log_status(nlo_log_read_buffer(dst, dst_bytes, out_written, consume));
}

NLOLIB_API nlolib_status nlolib_set_log_level(int level)
{
    return nlo_map_log_status(nlo_log_set_level(level));
}

NLOLIB_API nlolib_status nlolib_set_progress_options(
    int enabled,
    int milestone_percent,
    int emit_on_step_adjust
)
{
    return nlo_map_log_status(
        nlo_log_set_progress_options(enabled, milestone_percent, emit_on_step_adjust));
}
