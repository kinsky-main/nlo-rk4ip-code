/**
 * @file fft.c
 * @brief Backend-aware FFT implementation.
 */

#include "fft/fft.h"
#include "backend/vector_backend_internal.h"
#include "numerics/vk_vector_ops.h"
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <fftw3.h>

#ifndef ENABLE_VKFFT
#define ENABLE_VKFFT 1
#endif

#ifndef VKFFT_BACKEND
#define VKFFT_BACKEND 0
#endif

#if ENABLE_VKFFT
#include <vulkan/vulkan.h>
#include <vkFFT/vkFFT.h>
#endif

struct fft_plan {
    vector_backend* backend;
    vector_backend_type backend_type;
    fft_backend_type fft_backend;
    fft_shape shape;
    size_t total_size;

    fftw_plan forward_plan;
    fftw_plan inverse_plan;
    fftw_complex* plan_in;
    fftw_complex* plan_out;
    double inverse_scale;

#if ENABLE_VKFFT
    VkFFTApplication vk_app;
    vec_buffer* vk_placeholder_vec;
    VkBuffer vk_buffer_binding;
    uint64_t vk_buffer_size;
    VkCommandPool vk_command_pool;
    VkCommandBuffer vk_command_buffer;
    VkFence vkfft_internal_fence;
    VkFence vk_submit_fence;
#endif
};

static int fft_shape_valid(const fft_shape* shape)
{
    return (shape != NULL && shape->rank > 0u && shape->rank <= 3u);
}

static int fft_checked_mul_size(size_t a, size_t b, size_t* out)
{
    if (out == NULL) {
        return -1;
    }
    if (a == 0u || b == 0u) {
        *out = 0u;
        return 0;
    }
    if (a > (SIZE_MAX / b)) {
        return -1;
    }
    *out = a * b;
    return 0;
}

#if ENABLE_VKFFT
static vec_status fft_vk_validate_submit_context(const fft_plan* plan)
{
    if (plan == NULL || plan->backend == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    if (plan->vk_submit_fence == VK_NULL_HANDLE ||
        plan->vk_command_buffer == VK_NULL_HANDLE) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    return VEC_STATUS_OK;
}
#endif

static vec_status fft_validate_io_buffers(
    const fft_plan* plan,
    const vec_buffer* input,
    const vec_buffer* output
)
{
    if (plan == NULL || input == NULL || output == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    if (input->owner != plan->backend || output->owner != plan->backend ||
        input->kind != VEC_KIND_COMPLEX64 || output->kind != VEC_KIND_COMPLEX64 ||
        input->length != plan->total_size || output->length != plan->total_size) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    return VEC_STATUS_OK;
}

#if ENABLE_VKFFT
static vec_status fft_vk_begin_commands(fft_plan* plan)
{
    vec_status validation = fft_vk_validate_submit_context(plan);
    if (validation != VEC_STATUS_OK) {
        return validation;
    }

    vector_backend* backend = plan->backend;
    if (vkWaitForFences(backend->vk.device, 1u, &plan->vk_submit_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        return VEC_STATUS_BACKEND_UNAVAILABLE;
    }
    if (vkResetFences(backend->vk.device, 1u, &plan->vk_submit_fence) != VK_SUCCESS) {
        return VEC_STATUS_BACKEND_UNAVAILABLE;
    }
    if (vkResetCommandBuffer(plan->vk_command_buffer, 0u) != VK_SUCCESS) {
        return VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    if (vkBeginCommandBuffer(plan->vk_command_buffer, &begin_info) != VK_SUCCESS) {
        return VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    return VEC_STATUS_OK;
}
#endif

#if ENABLE_VKFFT
static vec_status fft_vk_submit_commands(fft_plan* plan)
{
    vec_status validation = fft_vk_validate_submit_context(plan);
    if (validation != VEC_STATUS_OK) {
        return validation;
    }

    vector_backend* backend = plan->backend;
    if (vkEndCommandBuffer(plan->vk_command_buffer) != VK_SUCCESS) {
        return VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1u,
        .pCommandBuffers = &plan->vk_command_buffer
    };
    if (vkQueueSubmit(backend->vk.queue, 1u, &submit_info, plan->vk_submit_fence) != VK_SUCCESS) {
        return VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    if (vkWaitForFences(backend->vk.device, 1u, &plan->vk_submit_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        return VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    return VEC_STATUS_OK;
}
#endif

#if ENABLE_VKFFT
static void fft_vk_cmd_compute_barrier(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize size)
{
    VkBufferMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = buffer,
        .offset = 0u,
        .size = size
    };

    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0u,
                         0u,
                         NULL,
                         1u,
                         &barrier,
                         0u,
                         NULL);
}
#endif

#if ENABLE_VKFFT
static vec_status fft_vk_execute_inplace(
    fft_plan* plan,
    vec_buffer* target,
    int inverse
)
{
    if (plan == NULL || target == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    vector_backend* backend = plan->backend;
    VkCommandBuffer cmd = plan->vk_command_buffer;
    vec_status status = VEC_STATUS_OK;
    if (backend != NULL && backend->in_simulation) {
        status = vk_simulation_phase_command_buffer(backend, &cmd);
    } else {
        status = fft_vk_begin_commands(plan);
    }
    if (status != VEC_STATUS_OK) {
        return status;
    }

    VkBuffer buffer = target->vk_buffer;
    fft_vk_cmd_compute_barrier(cmd, buffer, (VkDeviceSize)target->bytes);

    VkFFTLaunchParams launch_params = {0};
    launch_params.commandBuffer = &cmd;
    launch_params.buffer = &buffer;

    VkFFTResult result = VkFFTAppend(&plan->vk_app, inverse ? 1 : -1, &launch_params);
    if (result != VKFFT_SUCCESS) {
        if (backend == NULL || !backend->in_simulation) {
            (void)vkEndCommandBuffer(cmd);
        }
        return VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    fft_vk_cmd_compute_barrier(cmd, buffer, (VkDeviceSize)target->bytes);
    if (backend != NULL && backend->in_simulation) {
        vk_simulation_phase_mark_commands(backend);
        return VEC_STATUS_OK;
    }
    return fft_vk_submit_commands(plan);
}
#endif

static fft_backend_type fft_resolve_backend(
    vector_backend_type backend_type,
    fft_backend_type fft_backend
)
{
    if (fft_backend != FFT_BACKEND_AUTO) {
        return fft_backend;
    }

    if (backend_type == VECTOR_BACKEND_CPU) {
        return FFT_BACKEND_FFTW;
    }
    if (backend_type == VECTOR_BACKEND_VULKAN) {
        return FFT_BACKEND_VKFFT;
    }

    return FFT_BACKEND_AUTO;
}

static int fft_compute_total_size(const fft_shape* shape, size_t* out_total)
{
    if (!fft_shape_valid(shape) || out_total == NULL) {
        return -1;
    }

    size_t total = 1u;
    for (size_t i = 0u; i < shape->rank; ++i) {
        const size_t dim = shape->dims[i];
        if (dim == 0u || fft_checked_mul_size(total, dim, &total) != 0) {
            return -1;
        }
    }

    *out_total = total;
    return 0;
}

static int fft_shape_to_fftw_dims(const fft_shape* shape, int out_dims[3])
{
    if (!fft_shape_valid(shape) || out_dims == NULL) {
        return -1;
    }

    for (size_t i = 0u; i < shape->rank; ++i) {
        if (shape->dims[i] > (size_t)INT_MAX) {
            return -1;
        }
        out_dims[i] = (int)shape->dims[i];
    }
    return 0;
}

vec_status fft_plan_create(
    vector_backend* backend,
    size_t signal_size,
    fft_plan** out_plan
)
{
    return fft_plan_create_with_backend(
        backend,
        signal_size,
        FFT_BACKEND_AUTO,
        out_plan);
}

vec_status fft_plan_create_with_backend(
    vector_backend* backend,
    size_t signal_size,
    fft_backend_type fft_backend,
    fft_plan** out_plan
)
{
    fft_shape shape = {
        .rank = 1u,
        .dims = {signal_size, 1u, 1u}
    };
    return fft_plan_create_shaped_with_backend(backend, &shape, fft_backend, out_plan);
}

vec_status fft_plan_create_shaped_with_backend(
    vector_backend* backend,
    const fft_shape* shape,
    fft_backend_type fft_backend,
    fft_plan** out_plan
)
{
    if (backend == NULL || out_plan == NULL || shape == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    size_t total_size = 0u;
    if (fft_compute_total_size(shape, &total_size) != 0) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    *out_plan = NULL;

    fft_plan* plan = (fft_plan*)calloc(1, sizeof(*plan));
    if (plan == NULL) {
        return VEC_STATUS_ALLOCATION_FAILED;
    }

    plan->backend = backend;
    plan->backend_type = vector_backend_get_type(backend);
    plan->fft_backend = fft_resolve_backend(plan->backend_type, fft_backend);
    plan->shape = *shape;
    plan->total_size = total_size;
    plan->inverse_scale = 1.0 / (double)plan->total_size;

    if (plan->fft_backend == FFT_BACKEND_FFTW) {
        if (plan->backend_type != VECTOR_BACKEND_CPU) {
            fft_plan_destroy(plan);
            return VEC_STATUS_INVALID_ARGUMENT;
        }
        int fftw_dims[3] = {0};
        if (fft_shape_to_fftw_dims(&plan->shape, fftw_dims) != 0) {
            fft_plan_destroy(plan);
            return VEC_STATUS_INVALID_ARGUMENT;
        }

        plan->plan_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * plan->total_size);
        plan->plan_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * plan->total_size);
        if (plan->plan_in == NULL || plan->plan_out == NULL) {
            fft_plan_destroy(plan);
            return VEC_STATUS_ALLOCATION_FAILED;
        }

        const unsigned flags = FFTW_ESTIMATE | FFTW_UNALIGNED;
        plan->forward_plan = fftw_plan_dft((int)plan->shape.rank,
                                           fftw_dims,
                                           plan->plan_in,
                                           plan->plan_out,
                                           FFTW_FORWARD,
                                           flags);
        plan->inverse_plan = fftw_plan_dft((int)plan->shape.rank,
                                           fftw_dims,
                                           plan->plan_in,
                                           plan->plan_out,
                                           FFTW_BACKWARD,
                                           flags);
        if (plan->forward_plan == NULL || plan->inverse_plan == NULL) {
            fft_plan_destroy(plan);
            return VEC_STATUS_BACKEND_UNAVAILABLE;
        }

        *out_plan = plan;
        return VEC_STATUS_OK;
    }

    if (plan->fft_backend == FFT_BACKEND_VKFFT) {
#if ENABLE_VKFFT
        if (plan->backend_type != VECTOR_BACKEND_VULKAN) {
            fft_plan_destroy(plan);
            return VEC_STATUS_INVALID_ARGUMENT;
        }

        vec_status status = vec_create(backend,
                                               VEC_KIND_COMPLEX64,
                                               plan->total_size,
                                               &plan->vk_placeholder_vec);
        if (status != VEC_STATUS_OK) {
            fft_plan_destroy(plan);
            return status;
        }

        plan->vk_buffer_binding = plan->vk_placeholder_vec->vk_buffer;
        plan->vk_buffer_size = (uint64_t)plan->vk_placeholder_vec->bytes;

        VkCommandPoolCreateInfo pool_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .queueFamilyIndex = backend->vk.queue_family_index,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
        };
        if (vkCreateCommandPool(backend->vk.device, &pool_info, NULL, &plan->vk_command_pool) != VK_SUCCESS) {
            fft_plan_destroy(plan);
            return VEC_STATUS_BACKEND_UNAVAILABLE;
        }

        VkCommandBufferAllocateInfo command_buffer_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = plan->vk_command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1u
        };
        if (vkAllocateCommandBuffers(backend->vk.device,
                                     &command_buffer_info,
                                     &plan->vk_command_buffer) != VK_SUCCESS) {
            fft_plan_destroy(plan);
            return VEC_STATUS_BACKEND_UNAVAILABLE;
        }

        VkFenceCreateInfo vkfft_fence_info = {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO
        };
        if (vkCreateFence(backend->vk.device, &vkfft_fence_info, NULL, &plan->vkfft_internal_fence) != VK_SUCCESS) {
            fft_plan_destroy(plan);
            return VEC_STATUS_BACKEND_UNAVAILABLE;
        }

        VkFenceCreateInfo fence_info = {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT
        };
        if (vkCreateFence(backend->vk.device, &fence_info, NULL, &plan->vk_submit_fence) != VK_SUCCESS) {
            fft_plan_destroy(plan);
            return VEC_STATUS_BACKEND_UNAVAILABLE;
        }

        VkFFTConfiguration configuration = {0};
        configuration.FFTdim = (uint32_t)plan->shape.rank;
        for (size_t i = 0u; i < plan->shape.rank; ++i) {
            const size_t src_idx = (plan->shape.rank - 1u) - i;
            configuration.size[i] = (uint64_t)plan->shape.dims[src_idx];
        }
        configuration.numberBatches = 1u;
        configuration.doublePrecision = 1u;
        configuration.normalize = 0u;

        configuration.device = &backend->vk.device;
        configuration.physicalDevice = &backend->vk.physical_device;
        configuration.queue = &backend->vk.queue;
        configuration.commandPool = &plan->vk_command_pool;
        configuration.fence = &plan->vkfft_internal_fence;
        configuration.buffer = &plan->vk_buffer_binding;
        configuration.bufferSize = &plan->vk_buffer_size;

        VkFFTResult result = initializeVkFFT(&plan->vk_app, configuration);
        if (result != VKFFT_SUCCESS) {
            fft_plan_destroy(plan);
            return VEC_STATUS_BACKEND_UNAVAILABLE;
        }

        *out_plan = plan;
        return VEC_STATUS_OK;
#else
        fft_plan_destroy(plan);
        return VEC_STATUS_UNSUPPORTED;
#endif
    }

    fft_plan_destroy(plan);
    return VEC_STATUS_UNSUPPORTED;
}

void fft_plan_destroy(fft_plan* plan)
{
    if (plan == NULL) {
        return;
    }

    if (plan->fft_backend == FFT_BACKEND_FFTW) {
        if (plan->forward_plan != NULL) {
            fftw_destroy_plan(plan->forward_plan);
            plan->forward_plan = NULL;
        }
        if (plan->inverse_plan != NULL) {
            fftw_destroy_plan(plan->inverse_plan);
            plan->inverse_plan = NULL;
        }
        if (plan->plan_in != NULL) {
            fftw_free(plan->plan_in);
            plan->plan_in = NULL;
        }
        if (plan->plan_out != NULL) {
            fftw_free(plan->plan_out);
            plan->plan_out = NULL;
        }
    }

    if (plan->fft_backend == FFT_BACKEND_VKFFT) {
#if ENABLE_VKFFT
        deleteVkFFT(&plan->vk_app);
#endif
    }
#if ENABLE_VKFFT
    if (plan->backend != NULL &&
        plan->backend->type == VECTOR_BACKEND_VULKAN &&
        plan->backend->vk.device != VK_NULL_HANDLE &&
        plan->vkfft_internal_fence != VK_NULL_HANDLE) {
        vkDestroyFence(plan->backend->vk.device, plan->vkfft_internal_fence, NULL);
        plan->vkfft_internal_fence = VK_NULL_HANDLE;
    }
    if (plan->backend != NULL &&
        plan->backend->type == VECTOR_BACKEND_VULKAN &&
        plan->backend->vk.device != VK_NULL_HANDLE &&
        plan->vk_submit_fence != VK_NULL_HANDLE) {
        vkDestroyFence(plan->backend->vk.device, plan->vk_submit_fence, NULL);
        plan->vk_submit_fence = VK_NULL_HANDLE;
    }
    if (plan->backend != NULL &&
        plan->backend->type == VECTOR_BACKEND_VULKAN &&
        plan->backend->vk.device != VK_NULL_HANDLE &&
        plan->vk_command_buffer != VK_NULL_HANDLE &&
        plan->vk_command_pool != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(plan->backend->vk.device,
                             plan->vk_command_pool,
                             1u,
                             &plan->vk_command_buffer);
        plan->vk_command_buffer = VK_NULL_HANDLE;
    }
    if (plan->backend != NULL &&
        plan->backend->type == VECTOR_BACKEND_VULKAN &&
        plan->backend->vk.device != VK_NULL_HANDLE &&
        plan->vk_command_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(plan->backend->vk.device, plan->vk_command_pool, NULL);
        plan->vk_command_pool = VK_NULL_HANDLE;
    }
    if (plan->backend != NULL && plan->vk_placeholder_vec != NULL) {
        vec_destroy(plan->backend, plan->vk_placeholder_vec);
        plan->vk_placeholder_vec = NULL;
    }
#endif

    free(plan);
}

vec_status fft_forward_vec(
    fft_plan* plan,
    const vec_buffer* input,
    vec_buffer* output
)
{
    vec_status validation = fft_validate_io_buffers(plan, input, output);
    if (validation != VEC_STATUS_OK) {
        return validation;
    }

    if (plan->fft_backend == FFT_BACKEND_FFTW) {
        const void* in_ptr = NULL;
        void* out_ptr = NULL;
        vec_status status = vec_get_const_host_ptr(plan->backend, input, &in_ptr);
        if (status != VEC_STATUS_OK) {
            return status;
        }
        status = vec_get_host_ptr(plan->backend, output, &out_ptr);
        if (status != VEC_STATUS_OK) {
            return status;
        }

        fftw_execute_dft(plan->forward_plan, (fftw_complex*)in_ptr, (fftw_complex*)out_ptr);
        return VEC_STATUS_OK;
    }

    if (plan->fft_backend == FFT_BACKEND_VKFFT) {
#if ENABLE_VKFFT
        vec_status status = VEC_STATUS_OK;
        if (input != output) {
            status = vec_complex_copy(plan->backend, output, input);
            if (status != VEC_STATUS_OK) {
                return status;
            }
        }
        return fft_vk_execute_inplace(plan, output, 0);
#else
        return VEC_STATUS_UNSUPPORTED;
#endif
    }

    return VEC_STATUS_UNSUPPORTED;
}

vec_status fft_inverse_vec(
    fft_plan* plan,
    const vec_buffer* input,
    vec_buffer* output
)
{
    vec_status validation = fft_validate_io_buffers(plan, input, output);
    if (validation != VEC_STATUS_OK) {
        return validation;
    }

    if (plan->fft_backend == FFT_BACKEND_FFTW) {
        const void* in_ptr = NULL;
        void* out_ptr = NULL;
        vec_status status = vec_get_const_host_ptr(plan->backend, input, &in_ptr);
        if (status != VEC_STATUS_OK) {
            return status;
        }
        status = vec_get_host_ptr(plan->backend, output, &out_ptr);
        if (status != VEC_STATUS_OK) {
            return status;
        }

        fftw_execute_dft(plan->inverse_plan, (fftw_complex*)in_ptr, (fftw_complex*)out_ptr);

        nlo_complex* data = (nlo_complex*)out_ptr;
        for (size_t i = 0u; i < plan->total_size; ++i) {
            RE(data[i]) *= plan->inverse_scale;
            IM(data[i]) *= plan->inverse_scale;
        }
        return VEC_STATUS_OK;
    }

    if (plan->fft_backend == FFT_BACKEND_VKFFT) {
#if ENABLE_VKFFT
        vec_status status = VEC_STATUS_OK;
        if (input != output) {
            status = vec_complex_copy(plan->backend, output, input);
            if (status != VEC_STATUS_OK) {
                return status;
            }
        }
        status = fft_vk_execute_inplace(plan, output, 1);
        if (status != VEC_STATUS_OK) {
            return status;
        }

        return vec_complex_scalar_mul_inplace(plan->backend,
                                                  output,
                                                  make(plan->inverse_scale, 0.0));
#else
        return VEC_STATUS_UNSUPPORTED;
#endif
    }

    return VEC_STATUS_UNSUPPORTED;
}
