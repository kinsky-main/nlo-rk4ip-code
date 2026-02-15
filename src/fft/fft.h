/**
 * @file fft.h
 * @brief Backend-aware FFT API operating on vector backend buffers.
 */
#pragma once

#include "backend/vector_backend.h"
#include "fft/fft_backend.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nlo_fft_plan nlo_fft_plan;

/**
 * @brief Create an FFT plan using backend-default implementation selection.
 */
nlo_vec_status nlo_fft_plan_create(
    nlo_vector_backend* backend,
    size_t signal_size,
    nlo_fft_plan** out_plan
);

/**
 * @brief Create an FFT plan with an explicit runtime FFT implementation request.
 */
nlo_vec_status nlo_fft_plan_create_with_backend(
    nlo_vector_backend* backend,
    size_t signal_size,
    nlo_fft_backend_type fft_backend,
    nlo_fft_plan** out_plan
);

/**
 * @brief Destroy a previously created FFT plan.
 */
void nlo_fft_plan_destroy(nlo_fft_plan* plan);

/**
 * @brief Execute a forward FFT on backend buffers.
 */
nlo_vec_status nlo_fft_forward_vec(
    nlo_fft_plan* plan,
    const nlo_vec_buffer* input,
    nlo_vec_buffer* output
);

/**
 * @brief Execute an inverse FFT on backend buffers.
 */
nlo_vec_status nlo_fft_inverse_vec(
    nlo_fft_plan* plan,
    const nlo_vec_buffer* input,
    nlo_vec_buffer* output
);

#ifdef __cplusplus
}
#endif
