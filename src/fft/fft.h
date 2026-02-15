/**
 * @file fft.h
 * @brief Backend-aware FFT API operating on vector backend buffers.
 */
#pragma once

#include "backend/vector_backend.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nlo_fft_plan nlo_fft_plan;

nlo_vec_status nlo_fft_plan_create(
    nlo_vector_backend* backend,
    size_t signal_size,
    nlo_fft_plan** out_plan
);

void nlo_fft_plan_destroy(nlo_fft_plan* plan);

nlo_vec_status nlo_fft_forward_vec(
    nlo_fft_plan* plan,
    const nlo_vec_buffer* input,
    nlo_vec_buffer* output
);

nlo_vec_status nlo_fft_inverse_vec(
    nlo_fft_plan* plan,
    const nlo_vec_buffer* input,
    nlo_vec_buffer* output
);

#ifdef __cplusplus
}
#endif
