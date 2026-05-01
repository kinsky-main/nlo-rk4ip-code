/**
 * @file tensor_scaling_plan.c
 * @brief Helper utilities for tensor benchmark shape planning and region classification.
 */

#include "tensor_scaling_plan.h"

#include <limits.h>
#include <stdint.h>

static int bench_mul_size_t(size_t lhs, size_t rhs, size_t* out_value)
{
    if (out_value == NULL) {
        return -1;
    }

    if (lhs == 0u || rhs == 0u) {
        *out_value = 0u;
        return 0;
    }

    if (lhs > (SIZE_MAX / rhs)) {
        return -1;
    }

    *out_value = lhs * rhs;
    return 0;
}

int bench_tensor_shape_from_scale(
    size_t scale,
    bench_tensor_shape* out_shape
)
{
    size_t cube = 0u;
    size_t total = 0u;

    if (scale == 0u || out_shape == NULL) {
        return -1;
    }

    if (bench_mul_size_t(scale, scale, &cube) != 0 ||
        bench_mul_size_t(cube, scale, &cube) != 0 ||
        bench_mul_size_t(cube, 2u, &total) != 0) {
        return -1;
    }

    out_shape->scale = scale;
    out_shape->nt = 2u * scale;
    out_shape->nx = scale;
    out_shape->ny = scale;
    out_shape->total_samples = total;
    return 0;
}

bench_tensor_region bench_tensor_classify_fit_region(
    const bench_tensor_region_inputs* inputs
)
{
    if (inputs == NULL || inputs->cpu_init_ok == 0) {
        return BENCH_TENSOR_REGION_TOO_LARGE;
    }

    if (inputs->host_budget_bytes > 0u &&
        inputs->working_set_bytes > inputs->host_budget_bytes) {
        return BENCH_TENSOR_REGION_TOO_LARGE;
    }

    if (inputs->gpu_budget_bytes > 0u &&
        inputs->working_set_bytes <= inputs->gpu_budget_bytes &&
        inputs->gpu_init_ok != 0) {
        return BENCH_TENSOR_REGION_GPU_FIT;
    }

    if (inputs->host_budget_bytes == 0u ||
        inputs->working_set_bytes <= inputs->host_budget_bytes) {
        return BENCH_TENSOR_REGION_HOST_FIT_ONLY;
    }

    return BENCH_TENSOR_REGION_TOO_LARGE;
}

size_t bench_tensor_records_for_output_bytes(
    size_t per_record_bytes,
    size_t target_output_bytes
)
{
    size_t quotient = 0u;

    if (per_record_bytes == 0u) {
        return 0u;
    }

    quotient = target_output_bytes / per_record_bytes;
    if ((target_output_bytes % per_record_bytes) != 0u) {
        quotient += 1u;
    }
    if (quotient < 2u) {
        quotient = 2u;
    }

    return quotient;
}

const char* bench_tensor_region_label(bench_tensor_region region)
{
    switch (region) {
    case BENCH_TENSOR_REGION_GPU_FIT:
        return "gpu_fit";
    case BENCH_TENSOR_REGION_HOST_FIT_ONLY:
        return "host_fit_only";
    case BENCH_TENSOR_REGION_TOO_LARGE:
        return "too_large";
    case BENCH_TENSOR_REGION_OUTPUT_SPILL:
        return "output_spill";
    case BENCH_TENSOR_REGION_NONE:
    default:
        return "none";
    }
}
