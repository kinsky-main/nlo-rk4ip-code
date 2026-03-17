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
    NLO_BENCH_TENSOR_REGION_NONE = 0,
    NLO_BENCH_TENSOR_REGION_GPU_FIT = 1,
    NLO_BENCH_TENSOR_REGION_HOST_FIT_ONLY = 2,
    NLO_BENCH_TENSOR_REGION_TOO_LARGE = 3,
    NLO_BENCH_TENSOR_REGION_OUTPUT_SPILL = 4
} nlo_bench_tensor_region;

typedef struct {
    size_t scale;
    size_t nt;
    size_t nx;
    size_t ny;
    size_t total_samples;
} nlo_bench_tensor_shape;

typedef struct {
    size_t working_set_bytes;
    size_t host_budget_bytes;
    size_t gpu_budget_bytes;
    int cpu_init_ok;
    int gpu_init_ok;
} nlo_bench_tensor_region_inputs;

int nlo_bench_tensor_shape_from_scale(
    size_t scale,
    nlo_bench_tensor_shape* out_shape
);

nlo_bench_tensor_region nlo_bench_tensor_classify_fit_region(
    const nlo_bench_tensor_region_inputs* inputs
);

size_t nlo_bench_tensor_records_for_output_bytes(
    size_t per_record_bytes,
    size_t target_output_bytes
);

const char* nlo_bench_tensor_region_label(nlo_bench_tensor_region region);

#ifdef __cplusplus
}
#endif
