/**
 * @file vulkan_stubs.c
 * @dir src/backend
 * @brief Vulkan-disabled stubs for backend/FFT integration points.
 */

#include "backend/vk_auto_context.h"
#include "numerics/vk_vector_ops.h"
#include <string.h>

static vec_status vk_unavailable_status(void)
{
    return VEC_STATUS_UNSUPPORTED;
}

int vk_auto_context_init(
    vk_auto_context* context,
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

void vk_auto_context_destroy(vk_auto_context* context)
{
    if (context != NULL) {
        memset(context, 0, sizeof(*context));
    }
}

vec_status vk_backend_init(vector_backend* backend, const vk_backend_config* config)
{
    (void)backend;
    (void)config;
    return vk_unavailable_status();
}

void vk_backend_shutdown(vector_backend* backend)
{
    (void)backend;
}

vec_status vk_simulation_phase_begin(vector_backend* backend)
{
    (void)backend;
    return vk_unavailable_status();
}

vec_status vk_simulation_phase_flush(vector_backend* backend)
{
    (void)backend;
    return vk_unavailable_status();
}

vec_status vk_simulation_phase_command_buffer(
    vector_backend* backend,
    VkCommandBuffer* out_command_buffer
)
{
    (void)backend;
    if (out_command_buffer != NULL) {
        *out_command_buffer = VK_NULL_HANDLE;
    }
    return vk_unavailable_status();
}

void vk_simulation_phase_mark_commands(vector_backend* backend)
{
    (void)backend;
}

vec_status vk_buffer_create(vector_backend* backend, vec_buffer* buffer)
{
    (void)backend;
    (void)buffer;
    return vk_unavailable_status();
}

void vk_buffer_destroy(vector_backend* backend, vec_buffer* buffer)
{
    (void)backend;
    (void)buffer;
}

vec_status vk_upload(
    vector_backend* backend,
    vec_buffer* buffer,
    const void* data,
    size_t bytes
)
{
    (void)backend;
    (void)buffer;
    (void)data;
    (void)bytes;
    return vk_unavailable_status();
}

vec_status vk_download(
    vector_backend* backend,
    const vec_buffer* buffer,
    void* data,
    size_t bytes
)
{
    (void)backend;
    (void)buffer;
    (void)data;
    (void)bytes;
    return vk_unavailable_status();
}

vec_status vk_op_real_fill(vector_backend* backend, vec_buffer* dst, double value)
{
    (void)backend;
    (void)dst;
    (void)value;
    return vk_unavailable_status();
}

vec_status vk_op_real_copy(vector_backend* backend, vec_buffer* dst, const vec_buffer* src)
{
    (void)backend;
    (void)dst;
    (void)src;
    return vk_unavailable_status();
}

vec_status vk_op_real_mul_inplace(vector_backend* backend, vec_buffer* dst, const vec_buffer* src)
{
    (void)backend;
    (void)dst;
    (void)src;
    return vk_unavailable_status();
}

vec_status vk_op_complex_fill(vector_backend* backend, vec_buffer* dst, nlo_complex value)
{
    (void)backend;
    (void)dst;
    (void)value;
    return vk_unavailable_status();
}

vec_status vk_op_complex_copy(vector_backend* backend, vec_buffer* dst, const vec_buffer* src)
{
    (void)backend;
    (void)dst;
    (void)src;
    return vk_unavailable_status();
}

vec_status vk_op_complex_magnitude_squared(
    vector_backend* backend,
    const vec_buffer* src,
    vec_buffer* dst
)
{
    (void)backend;
    (void)src;
    (void)dst;
    return vk_unavailable_status();
}

vec_status vk_op_complex_scalar_mul_inplace(
    vector_backend* backend,
    vec_buffer* dst,
    nlo_complex alpha
)
{
    (void)backend;
    (void)dst;
    (void)alpha;
    return vk_unavailable_status();
}

vec_status vk_op_complex_mul_inplace(
    vector_backend* backend,
    vec_buffer* dst,
    const vec_buffer* src
)
{
    (void)backend;
    (void)dst;
    (void)src;
    return vk_unavailable_status();
}

vec_status vk_op_complex_add_inplace(
    vector_backend* backend,
    vec_buffer* dst,
    const vec_buffer* src
)
{
    (void)backend;
    (void)dst;
    (void)src;
    return vk_unavailable_status();
}

vec_status vk_op_complex_exp_inplace(vector_backend* backend, vec_buffer* dst)
{
    (void)backend;
    (void)dst;
    return vk_unavailable_status();
}

vec_status vk_op_complex_real_pow_inplace(
    vector_backend* backend,
    vec_buffer* dst,
    double exponent
)
{
    (void)backend;
    (void)dst;
    (void)exponent;
    return vk_unavailable_status();
}

vec_status vk_op_complex_relative_error(
    vector_backend* backend,
    const vec_buffer* current,
    const vec_buffer* previous,
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
    return vk_unavailable_status();
}

vec_status vk_op_complex_weighted_rms_error(
    vector_backend* backend,
    const vec_buffer* fine,
    const vec_buffer* coarse,
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
    return vk_unavailable_status();
}

vec_status vk_op_complex_axis_unshifted_from_delta(
    vector_backend* backend,
    vec_buffer* dst,
    double delta
)
{
    (void)backend;
    (void)dst;
    (void)delta;
    return vk_unavailable_status();
}

vec_status vk_op_complex_axis_centered_from_delta(
    vector_backend* backend,
    vec_buffer* dst,
    double delta
)
{
    (void)backend;
    (void)dst;
    (void)delta;
    return vk_unavailable_status();
}

vec_status vk_op_complex_mesh_from_axis_tfast(
    vector_backend* backend,
    vec_buffer* dst,
    const vec_buffer* axis,
    size_t nt,
    size_t ny,
    vec_mesh_axis axis_kind
)
{
    (void)backend;
    (void)dst;
    (void)axis;
    (void)nt;
    (void)ny;
    (void)axis_kind;
    return vk_unavailable_status();
}
