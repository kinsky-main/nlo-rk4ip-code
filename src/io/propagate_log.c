/**
 * @file propagate_log.c
 * @brief Structured logging helpers for nlolib propagation entry points.
 */

#include "io/propagate_log.h"
#include "core/sim_dimensions_internal.h"
#include "io/log_format.h"
#include "io/log_sink.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef NLO_LOG_DEFAULT_DISPERSION_FACTOR_EXPR
#define NLO_LOG_DEFAULT_DISPERSION_FACTOR_EXPR "i*c0*w*w-c1"
#endif

#ifndef NLO_LOG_DEFAULT_LINEAR_FACTOR_EXPR
#define NLO_LOG_DEFAULT_LINEAR_FACTOR_EXPR "i*c0*wt*wt-c1"
#endif

#ifndef NLO_LOG_DEFAULT_DISPERSION_EXPR
#define NLO_LOG_DEFAULT_DISPERSION_EXPR "exp(h*D)"
#endif

#ifndef NLO_LOG_DEFAULT_LINEAR_EXPR
#define NLO_LOG_DEFAULT_LINEAR_EXPR "exp(h*D)"
#endif

#ifndef NLO_LOG_DEFAULT_POTENTIAL_EXPR
#define NLO_LOG_DEFAULT_POTENTIAL_EXPR "0"
#endif

#ifndef NLO_LOG_DEFAULT_NONLINEAR_EXPR
#define NLO_LOG_DEFAULT_NONLINEAR_EXPR "i*A*(c2*I + V)"
#endif

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

static const char* nlo_resolve_runtime_expr_alias(const char* primary, const char* alias)
{
    if (primary != NULL && primary[0] != '\0') {
        return primary;
    }
    if (alias != NULL && alias[0] != '\0') {
        return alias;
    }
    return NULL;
}

void nlo_log_propagate_request(
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
    const int tensor_mode_active = (config != NULL && config->tensor.nt > 0u) ? 1 : 0;

    char num_time_samples_text[48];
    char num_recorded_samples_text[48];
    char field_bytes_text[48];
    char records_bytes_text[48];
    char field_size_text[48];
    char records_size_text[48];
    char frequency_size_text[48];
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
    (void)nlo_log_format_u64_grouped(nt_text, sizeof(nt_text), (uint64_t)nt);
    (void)nlo_log_format_u64_grouped(nx_text, sizeof(nx_text), (uint64_t)nx);
    (void)nlo_log_format_u64_grouped(ny_text, sizeof(ny_text), (uint64_t)ny);
    (void)nlo_log_format_bytes_summary(field_size_text, sizeof(field_size_text), field_bytes);
    (void)nlo_log_format_bytes_summary(records_size_text, sizeof(records_size_text), records_bytes);
    (void)nlo_log_format_bytes_summary(frequency_size_text,
                                       sizeof(frequency_size_text),
                                       frequency_grid_bytes);

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
    const char* linear_factor_op = NULL;
    const char* linear_op = NULL;
    const char* potential_op = NULL;
    const char* nonlinear_op = NULL;
    if (config != NULL) {
        linear_factor_op = tensor_mode_active
                               ? nlo_resolve_runtime_expr_alias(config->runtime.linear_factor_expr,
                                                                config->runtime.dispersion_factor_expr)
                               : nlo_resolve_runtime_expr_alias(config->runtime.dispersion_factor_expr,
                                                                config->runtime.linear_factor_expr);
        if (linear_factor_op == NULL || linear_factor_op[0] == '\0') {
            linear_factor_op = tensor_mode_active
                                   ? NLO_LOG_DEFAULT_LINEAR_FACTOR_EXPR
                                   : NLO_LOG_DEFAULT_DISPERSION_FACTOR_EXPR;
        }

        linear_op = tensor_mode_active
                        ? nlo_resolve_runtime_expr_alias(config->runtime.linear_expr,
                                                         config->runtime.dispersion_expr)
                        : nlo_resolve_runtime_expr_alias(config->runtime.dispersion_expr,
                                                         config->runtime.linear_expr);
        if (linear_op == NULL || linear_op[0] == '\0') {
            linear_op = tensor_mode_active
                            ? NLO_LOG_DEFAULT_LINEAR_EXPR
                            : NLO_LOG_DEFAULT_DISPERSION_EXPR;
        }

        potential_op = config->runtime.potential_expr;
        if (potential_op == NULL || potential_op[0] == '\0') {
            potential_op = NLO_LOG_DEFAULT_POTENTIAL_EXPR;
        }

        nonlinear_op = config->runtime.nonlinear_expr;
        if (nonlinear_op == NULL || nonlinear_op[0] == '\0') {
            nonlinear_op = NLO_LOG_DEFAULT_NONLINEAR_EXPR;
        }
    }

    const int written = snprintf(
        message,
        sizeof(message),
        "[nlolib] propagate request:\n"
        "  - backend_requested: %s\n"
        "  - num_time_samples: %s\n"
        "  - num_recorded_samples: %s\n"
        "  - field_size: %s\n"
        "  - records_size: %s\n"
        "  - pointers:\n"
        "    - config: %p\n"
        "    - input_field: %p\n"
        "    - output_records: %p\n"
        "    - exec_options: %p\n"
        "  - runtime_operators:\n"
        "    - linear_factor_op: %s\n"
        "    - linear_op: %s\n"
        "    - potential_op: %s\n"
        "    - nonlinear_op: %s\n"
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
        records_size_text,
        (const void*)config,
        (const void*)input_field,
        (const void*)output_records,
        (const void*)exec_options,
        (linear_factor_op != NULL) ? linear_factor_op : "(null)",
        (linear_op != NULL) ? linear_op : "(null)",
        (potential_op != NULL) ? potential_op : "(null)",
        (nonlinear_op != NULL) ? nonlinear_op : "(null)",
        runtime_constants_text,
        constants_lines,
        (config != NULL) ? (const void*)config->frequency.frequency_grid : NULL,
        frequency_size_text,
        field_bytes_text,
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

void nlo_log_propagate_allocation_summary(
    nlo_vector_backend_type requested_backend,
    nlo_vector_backend_type actual_backend,
    const nlo_allocation_info* allocation_info,
    const nlo_vec_backend_memory_info* mem_info
)
{
    const nlo_allocation_info empty_allocation = {0};
    const nlo_vec_backend_memory_info empty_mem_info = {0};
    const nlo_allocation_info* safe_allocation =
        (allocation_info != NULL) ? allocation_info : &empty_allocation;
    const nlo_vec_backend_memory_info* safe_mem_info =
        (mem_info != NULL) ? mem_info : &empty_mem_info;

    size_t record_ring_bytes = 0u;
    if (safe_allocation->per_record_bytes != 0u &&
        safe_allocation->device_ring_capacity <= (SIZE_MAX / safe_allocation->per_record_bytes)) {
        record_ring_bytes = safe_allocation->per_record_bytes * safe_allocation->device_ring_capacity;
    }

    char per_record_text[48];
    char working_vector_text[48];
    char host_snapshot_text[48];
    char ring_capacity_text[48];
    char ring_bytes_text[48];
    char device_budget_text[48];
    char total_device_text[48];
    char available_device_text[48];

    (void)nlo_log_format_u64_grouped(ring_capacity_text,
                                     sizeof(ring_capacity_text),
                                     (uint64_t)safe_allocation->device_ring_capacity);
    (void)nlo_log_format_bytes_summary(per_record_text,
                                       sizeof(per_record_text),
                                       safe_allocation->per_record_bytes);
    (void)nlo_log_format_bytes_summary(working_vector_text,
                                       sizeof(working_vector_text),
                                       safe_allocation->working_vector_bytes);
    (void)nlo_log_format_bytes_summary(host_snapshot_text,
                                       sizeof(host_snapshot_text),
                                       safe_allocation->host_snapshot_bytes);
    (void)nlo_log_format_bytes_summary(ring_bytes_text, sizeof(ring_bytes_text), record_ring_bytes);
    (void)nlo_log_format_bytes_summary(device_budget_text,
                                       sizeof(device_budget_text),
                                       safe_allocation->device_budget_bytes);
    (void)nlo_log_format_bytes_summary(total_device_text,
                                       sizeof(total_device_text),
                                       safe_mem_info->device_local_total_bytes);
    (void)nlo_log_format_bytes_summary(available_device_text,
                                       sizeof(available_device_text),
                                       safe_mem_info->device_local_available_bytes);

    nlo_log_emit(NLO_LOG_LEVEL_INFO,
                 "[nlolib] backend resolved:\n"
                 "  - requested: %s\n"
                 "  - actual: %s",
                 nlo_backend_type_to_string(requested_backend),
                 nlo_backend_type_to_string(actual_backend));

    nlo_log_emit(
        NLO_LOG_LEVEL_INFO,
        "[nlolib] allocation summary:\n"
        "  - per_record_bytes: %s\n"
        "  - working_vector_bytes_estimate: %s\n"
        "  - host_snapshot_bytes: %s\n"
        "  - record_ring_capacity: %s\n"
        "  - record_ring_bytes: %s\n"
        "  - device_budget_bytes_effective: %s\n"
        "  - device_local_total_bytes: %s\n"
        "  - device_local_available_bytes: %s",
        per_record_text,
        working_vector_text,
        host_snapshot_text,
        ring_capacity_text,
        ring_bytes_text,
        device_budget_text,
        total_device_text,
        available_device_text);
}
