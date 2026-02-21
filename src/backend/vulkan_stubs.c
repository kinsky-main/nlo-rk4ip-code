/**
 * @file vulkan_stubs.c
 * @dir src/backend
 * @brief Vulkan-disabled stubs for backend/FFT integration points.
 */

#include "backend/vk_auto_context.h"
#include "numerics/vk_vector_ops.h"
#include <string.h>

static nlo_vec_status nlo_vk_unavailable_status(void)
{
    return NLO_VEC_STATUS_UNSUPPORTED;
}

int nlo_vk_auto_context_init(
    nlo_vk_auto_context* context,
    char* reason,
    size_t reason_capacity
)
{
    if (context != NULL) {
        memset(context, 0, sizeof(*context));
    }
    if (reason != NULL && reason_capacity > 0u) {
        const char* text = "Vulkan backend is disabled in this build.";
        size_t i = 0u;
        for (; i + 1u < reason_capacity && text[i] != '\0'; ++i) {
            reason[i] = text[i];
        }
        reason[i] = '\0';
    }
    return -1;
}

void nlo_vk_auto_context_destroy(nlo_vk_auto_context* context)
{
    if (context != NULL) {
        memset(context, 0, sizeof(*context));
    }
}

nlo_vec_status nlo_vk_backend_init(nlo_vector_backend* backend, const nlo_vk_backend_config* config)
{
    (void)backend;
    (void)config;
    return nlo_vk_unavailable_status();
}

void nlo_vk_backend_shutdown(nlo_vector_backend* backend)
{
    (void)backend;
}

nlo_vec_status nlo_vk_simulation_phase_begin(nlo_vector_backend* backend)
{
    (void)backend;
    return nlo_vk_unavailable_status();
}

nlo_vec_status nlo_vk_simulation_phase_flush(nlo_vector_backend* backend)
{
    (void)backend;
    return nlo_vk_unavailable_status();
}

nlo_vec_status nlo_vk_simulation_phase_command_buffer(
    nlo_vector_backend* backend,
    VkCommandBuffer* out_command_buffer
)
{
    (void)backend;
    if (out_command_buffer != NULL) {
        *out_command_buffer = VK_NULL_HANDLE;
    }
    return nlo_vk_unavailable_status();
}

void nlo_vk_simulation_phase_mark_commands(nlo_vector_backend* backend)
{
    (void)backend;
}

nlo_vec_status nlo_vk_buffer_create(nlo_vector_backend* backend, nlo_vec_buffer* buffer)
{
    (void)backend;
    (void)buffer;
    return nlo_vk_unavailable_status();
}

void nlo_vk_buffer_destroy(nlo_vector_backend* backend, nlo_vec_buffer* buffer)
{
    (void)backend;
    (void)buffer;
}

nlo_vec_status nlo_vk_upload(
    nlo_vector_backend* backend,
    nlo_vec_buffer* buffer,
    const void* data,
    size_t bytes
)
{
    (void)backend;
    (void)buffer;
    (void)data;
    (void)bytes;
    return nlo_vk_unavailable_status();
}

nlo_vec_status nlo_vk_download(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* buffer,
    void* data,
    size_t bytes
)
{
    (void)backend;
    (void)buffer;
    (void)data;
    (void)bytes;
    return nlo_vk_unavailable_status();
}

nlo_vec_status nlo_vk_op_real_fill(nlo_vector_backend* backend, nlo_vec_buffer* dst, double value)
{
    (void)backend;
    (void)dst;
    (void)value;
    return nlo_vk_unavailable_status();
}

nlo_vec_status nlo_vk_op_real_copy(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src)
{
    (void)backend;
    (void)dst;
    (void)src;
    return nlo_vk_unavailable_status();
}

nlo_vec_status nlo_vk_op_real_mul_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src)
{
    (void)backend;
    (void)dst;
    (void)src;
    return nlo_vk_unavailable_status();
}

nlo_vec_status nlo_vk_op_complex_fill(nlo_vector_backend* backend, nlo_vec_buffer* dst, nlo_complex value)
{
    (void)backend;
    (void)dst;
    (void)value;
    return nlo_vk_unavailable_status();
}

nlo_vec_status nlo_vk_op_complex_copy(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src)
{
    (void)backend;
    (void)dst;
    (void)src;
    return nlo_vk_unavailable_status();
}

nlo_vec_status nlo_vk_op_complex_magnitude_squared(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* src,
    nlo_vec_buffer* dst
)
{
    (void)backend;
    (void)src;
    (void)dst;
    return nlo_vk_unavailable_status();
}

nlo_vec_status nlo_vk_op_complex_scalar_mul_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    nlo_complex alpha
)
{
    (void)backend;
    (void)dst;
    (void)alpha;
    return nlo_vk_unavailable_status();
}

nlo_vec_status nlo_vk_op_complex_mul_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src
)
{
    (void)backend;
    (void)dst;
    (void)src;
    return nlo_vk_unavailable_status();
}

nlo_vec_status nlo_vk_op_complex_add_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src
)
{
    (void)backend;
    (void)dst;
    (void)src;
    return nlo_vk_unavailable_status();
}

nlo_vec_status nlo_vk_op_complex_exp_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst)
{
    (void)backend;
    (void)dst;
    return nlo_vk_unavailable_status();
}

nlo_vec_status nlo_vk_op_complex_real_pow_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    double exponent
)
{
    (void)backend;
    (void)dst;
    (void)exponent;
    return nlo_vk_unavailable_status();
}

nlo_vec_status nlo_vk_op_complex_relative_error(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* current,
    const nlo_vec_buffer* previous,
    double epsilon,
    double* out_error
)
{
    (void)backend;
    (void)current;
    (void)previous;
    (void)epsilon;
    if (out_error != NULL) {
        *out_error = 0.0;
    }
    return nlo_vk_unavailable_status();
}

nlo_vec_status nlo_vk_op_complex_weighted_rms_error(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* fine,
    const nlo_vec_buffer* coarse,
    double atol,
    double rtol,
    double* out_error
)
{
    (void)backend;
    (void)fine;
    (void)coarse;
    (void)atol;
    (void)rtol;
    if (out_error != NULL) {
        *out_error = 0.0;
    }
    return nlo_vk_unavailable_status();
}
