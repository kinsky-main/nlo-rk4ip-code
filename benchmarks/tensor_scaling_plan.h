/**
 * @file tensor_scaling_plan.h
 * @brief Helper utilities for tensor benchmark shape planning and region classification.
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    BENCH_TENSOR_REGION_NONE = 0,
    BENCH_TENSOR_REGION_GPU_FIT = 1,
    BENCH_TENSOR_REGION_HOST_FIT_ONLY = 2,
    BENCH_TENSOR_REGION_TOO_LARGE = 3,
    BENCH_TENSOR_REGION_OUTPUT_SPILL = 4
} bench_tensor_region;

typedef struct {
    size_t scale;
    size_t nt;
    size_t nx;
    size_t ny;
    size_t total_samples;
} bench_tensor_shape;

typedef struct {
    size_t working_set_bytes;
    size_t host_budget_bytes;
    size_t gpu_budget_bytes;
    int cpu_init_ok;
    int gpu_init_ok;
} bench_tensor_region_inputs;

int bench_tensor_shape_from_scale(
    size_t scale,
    bench_tensor_shape* out_shape
);

bench_tensor_region bench_tensor_classify_fit_region(
    const bench_tensor_region_inputs* inputs
);

size_t bench_tensor_records_for_output_bytes(
    size_t per_record_bytes,
    size_t target_output_bytes
);

const char* bench_tensor_region_label(bench_tensor_region region);

#ifdef __cplusplus
}
#endif
