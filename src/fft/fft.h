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
 * @brief Explicit FFT dimensionality descriptor.
 */
typedef struct {
    size_t rank;
    size_t dims[3];
} nlo_fft_shape;

/**
 * @brief Create an FFT plan using backend-default implementation selection.
 *
 * @param backend Active vector backend.
 * @param signal_size Flattened 1D signal length.
 * @param out_plan Destination FFT plan handle.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_fft_plan_create(
    nlo_vector_backend* backend,
    size_t signal_size,
    nlo_fft_plan** out_plan
);

/**
 * @brief Create an FFT plan with an explicit runtime FFT implementation request.
 *
 * @param backend Active vector backend.
 * @param signal_size Flattened 1D signal length.
 * @param fft_backend Requested FFT backend implementation.
 * @param out_plan Destination FFT plan handle.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_fft_plan_create_with_backend(
    nlo_vector_backend* backend,
    size_t signal_size,
    nlo_fft_backend_type fft_backend,
    nlo_fft_plan** out_plan
);

/**
 * @brief Create an FFT plan for an explicit shape (rank 1-3).
 *
 * @param backend Active vector backend.
 * @param shape FFT rank/dimensions descriptor.
 * @param fft_backend Requested FFT backend implementation.
 * @param out_plan Destination FFT plan handle.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_fft_plan_create_shaped_with_backend(
    nlo_vector_backend* backend,
    const nlo_fft_shape* shape,
    nlo_fft_backend_type fft_backend,
    nlo_fft_plan** out_plan
);

/**
 * @brief Destroy a previously created FFT plan.
 *
 * @param plan Plan handle to destroy (NULL is allowed).
 */
void nlo_fft_plan_destroy(nlo_fft_plan* plan);

/**
 * @brief Execute a forward FFT on backend buffers.
 *
 * @param plan FFT plan handle.
 * @param input Input vector.
 * @param output Output vector.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_fft_forward_vec(
    nlo_fft_plan* plan,
    const nlo_vec_buffer* input,
    nlo_vec_buffer* output
);

/**
 * @brief Execute an inverse FFT on backend buffers.
 *
 * @param plan FFT plan handle.
 * @param input Input vector.
 * @param output Output vector.
 * @return nlo_vec_status Operation status.
 */
nlo_vec_status nlo_fft_inverse_vec(
    nlo_fft_plan* plan,
    const nlo_vec_buffer* input,
    nlo_vec_buffer* output
);

#ifdef __cplusplus
}
#endif
