/**
 * @file bench_tensor_scaling.h
 * @brief Tensor 3D scaling benchmark runner.
 */
#pragma once

#include "vulkan_bench_context.h"
#include <stddef.h>

typedef struct {
    int backend_request;
    size_t warmup_runs;
    size_t measured_runs;
    int dry_run;
    size_t planner_host_bytes;
    size_t planner_gpu_bytes;
    const size_t* tensor_scales;
    size_t tensor_scale_count;
    const char* csv_path;
    const char* storage_dir;
} bench_tensor_options;

int bench_run_tensor_scaling(
    const bench_tensor_options* options,
    const bench_vk_context* vk_context,
    int gpu_available,
    const char* gpu_skip_reason,
    int* out_error_count
);
