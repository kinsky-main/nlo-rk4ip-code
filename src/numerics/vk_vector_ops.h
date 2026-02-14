/**
 * @file vk_vector_ops.h
 * @dir src/numerics
 * @brief Internal Vulkan vector operation backend.
 */
#pragma once

#include "backend/vector_backend_internal.h"

#ifdef NLO_ENABLE_VECTOR_BACKEND_VULKAN
nlo_vec_status nlo_vk_backend_init(nlo_vector_backend* backend, const nlo_vk_backend_config* config);
void nlo_vk_backend_shutdown(nlo_vector_backend* backend);

nlo_vec_status nlo_vk_buffer_create(nlo_vector_backend* backend, nlo_vec_buffer* buffer);
void nlo_vk_buffer_destroy(nlo_vector_backend* backend, nlo_vec_buffer* buffer);

nlo_vec_status nlo_vk_upload(nlo_vector_backend* backend,
                             nlo_vec_buffer* buffer,
                             const void* data,
                             size_t bytes);
nlo_vec_status nlo_vk_download(nlo_vector_backend* backend,
                               const nlo_vec_buffer* buffer,
                               void* data,
                               size_t bytes);

nlo_vec_status nlo_vk_op_real_fill(nlo_vector_backend* backend, nlo_vec_buffer* dst, double value);
nlo_vec_status nlo_vk_op_real_copy(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src);
nlo_vec_status nlo_vk_op_real_mul_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src);
nlo_vec_status nlo_vk_op_complex_fill(nlo_vector_backend* backend, nlo_vec_buffer* dst, nlo_complex value);
nlo_vec_status nlo_vk_op_complex_copy(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src);
nlo_vec_status nlo_vk_op_complex_magnitude_squared(nlo_vector_backend* backend,
                                                   const nlo_vec_buffer* src,
                                                   nlo_vec_buffer* dst);
nlo_vec_status nlo_vk_op_complex_scalar_mul_inplace(nlo_vector_backend* backend,
                                                    nlo_vec_buffer* dst,
                                                    nlo_complex alpha);
nlo_vec_status nlo_vk_op_complex_mul_inplace(nlo_vector_backend* backend,
                                             nlo_vec_buffer* dst,
                                             const nlo_vec_buffer* src);
nlo_vec_status nlo_vk_op_complex_add_inplace(nlo_vector_backend* backend,
                                             nlo_vec_buffer* dst,
                                             const nlo_vec_buffer* src);
#endif


