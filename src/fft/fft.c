/**
 * @file fft.c
 * @brief Backend-aware FFT implementation.
 */

#include "fft/fft.h"
#include "backend/vector_backend_internal.h"
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(NLO_FFT_BACKEND_FFTW)
#include <fftw3.h>
#endif

#if defined(NLO_ENABLE_VECTOR_BACKEND_VULKAN) && defined(NLO_ENABLE_VKFFT)
#ifndef VKFFT_BACKEND
#define VKFFT_BACKEND 0
#endif
#include <vulkan/vulkan.h>
#include <vkFFT/vkFFT.h>
#endif

struct nlo_fft_plan {
    nlo_vector_backend* backend;
    nlo_vector_backend_type backend_type;
    size_t signal_size;

#if defined(NLO_FFT_BACKEND_FFTW)
    fftw_plan forward_plan;
    fftw_plan inverse_plan;
    fftw_complex* plan_in;
    fftw_complex* plan_out;
    double inverse_scale;
#endif

#if defined(NLO_ENABLE_VECTOR_BACKEND_VULKAN) && defined(NLO_ENABLE_VKFFT)
    VkFFTApplication vk_app;
    nlo_vec_buffer* vk_placeholder_vec;
    VkBuffer vk_buffer_binding;
    uint64_t vk_buffer_size;
#endif
};

#if defined(NLO_ENABLE_VECTOR_BACKEND_VULKAN) && defined(NLO_ENABLE_VKFFT)
static nlo_vec_status nlo_fft_vk_begin_commands(nlo_fft_plan* plan)
{
    if (plan == NULL || plan->backend == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_vector_backend* backend = plan->backend;
    if (vkWaitForFences(backend->vk.device, 1u, &backend->vk.submit_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }
    if (vkResetFences(backend->vk.device, 1u, &backend->vk.submit_fence) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }
    if (vkResetCommandBuffer(backend->vk.command_buffer, 0u) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    if (vkBeginCommandBuffer(backend->vk.command_buffer, &begin_info) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    return NLO_VEC_STATUS_OK;
}

static nlo_vec_status nlo_fft_vk_submit_commands(nlo_fft_plan* plan)
{
    if (plan == NULL || plan->backend == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_vector_backend* backend = plan->backend;
    if (vkEndCommandBuffer(backend->vk.command_buffer) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1u,
        .pCommandBuffers = &backend->vk.command_buffer
    };
    if (vkQueueSubmit(backend->vk.queue, 1u, &submit_info, backend->vk.submit_fence) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    if (vkWaitForFences(backend->vk.device, 1u, &backend->vk.submit_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    return NLO_VEC_STATUS_OK;
}

static void nlo_fft_vk_cmd_compute_barrier(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize size)
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

static nlo_vec_status nlo_fft_vk_execute_inplace(nlo_fft_plan* plan,
                                                  nlo_vec_buffer* target,
                                                  int inverse)
{
    if (plan == NULL || target == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_vec_status status = nlo_fft_vk_begin_commands(plan);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    VkCommandBuffer cmd = plan->backend->vk.command_buffer;
    VkBuffer buffer = target->vk_buffer;
    nlo_fft_vk_cmd_compute_barrier(cmd, buffer, (VkDeviceSize)target->bytes);

    VkFFTLaunchParams launch_params = {0};
    launch_params.commandBuffer = &cmd;
    launch_params.buffer = &buffer;

    VkFFTResult result = VkFFTAppend(&plan->vk_app, inverse ? 1 : -1, &launch_params);
    if (result != VKFFT_SUCCESS) {
        (void)vkEndCommandBuffer(cmd);
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    nlo_fft_vk_cmd_compute_barrier(cmd, buffer, (VkDeviceSize)target->bytes);
    return nlo_fft_vk_submit_commands(plan);
}
#endif

nlo_vec_status nlo_fft_plan_create(nlo_vector_backend* backend,
                                   size_t signal_size,
                                   nlo_fft_plan** out_plan)
{
    if (backend == NULL || out_plan == NULL || signal_size == 0u) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    *out_plan = NULL;

    nlo_fft_plan* plan = (nlo_fft_plan*)calloc(1, sizeof(*plan));
    if (plan == NULL) {
        return NLO_VEC_STATUS_ALLOCATION_FAILED;
    }

    plan->backend = backend;
    plan->backend_type = nlo_vector_backend_get_type(backend);
    plan->signal_size = signal_size;

    if (plan->backend_type == NLO_VECTOR_BACKEND_CPU) {
#if defined(NLO_FFT_BACKEND_FFTW)
        if (signal_size > (size_t)INT_MAX) {
            free(plan);
            return NLO_VEC_STATUS_INVALID_ARGUMENT;
        }

        plan->plan_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * signal_size);
        plan->plan_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * signal_size);
        if (plan->plan_in == NULL || plan->plan_out == NULL) {
            nlo_fft_plan_destroy(plan);
            return NLO_VEC_STATUS_ALLOCATION_FAILED;
        }

        const unsigned flags = FFTW_ESTIMATE | FFTW_UNALIGNED;
        plan->forward_plan = fftw_plan_dft_1d((int)signal_size,
                                              plan->plan_in,
                                              plan->plan_out,
                                              FFTW_FORWARD,
                                              flags);
        plan->inverse_plan = fftw_plan_dft_1d((int)signal_size,
                                              plan->plan_in,
                                              plan->plan_out,
                                              FFTW_BACKWARD,
                                              flags);
        if (plan->forward_plan == NULL || plan->inverse_plan == NULL) {
            nlo_fft_plan_destroy(plan);
            return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
        }

        plan->inverse_scale = 1.0 / (double)signal_size;
        *out_plan = plan;
        return NLO_VEC_STATUS_OK;
#else
        free(plan);
        return NLO_VEC_STATUS_UNSUPPORTED;
#endif
    }

#if defined(NLO_ENABLE_VECTOR_BACKEND_VULKAN) && defined(NLO_ENABLE_VKFFT)
    if (plan->backend_type == NLO_VECTOR_BACKEND_VULKAN) {
        nlo_vec_status status = nlo_vec_create(backend,
                                               NLO_VEC_KIND_COMPLEX64,
                                               signal_size,
                                               &plan->vk_placeholder_vec);
        if (status != NLO_VEC_STATUS_OK) {
            nlo_fft_plan_destroy(plan);
            return status;
        }

        plan->vk_buffer_binding = plan->vk_placeholder_vec->vk_buffer;
        plan->vk_buffer_size = (uint64_t)plan->vk_placeholder_vec->bytes;

        VkFFTConfiguration configuration = {0};
        configuration.FFTdim = 1u;
        configuration.size[0] = (uint64_t)signal_size;
        configuration.numberBatches = 1u;
        configuration.doublePrecision = 1u;
        configuration.normalize = 1u;

        configuration.device = &backend->vk.device;
        configuration.physicalDevice = &backend->vk.physical_device;
        configuration.queue = &backend->vk.queue;
        configuration.commandPool = &backend->vk.command_pool;
        configuration.fence = &backend->vk.submit_fence;
        configuration.buffer = &plan->vk_buffer_binding;
        configuration.bufferSize = &plan->vk_buffer_size;

        VkFFTResult result = initializeVkFFT(&plan->vk_app, configuration);
        if (result != VKFFT_SUCCESS) {
            nlo_fft_plan_destroy(plan);
            return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
        }

        *out_plan = plan;
        return NLO_VEC_STATUS_OK;
    }
#else
    (void)signal_size;
#endif

    nlo_fft_plan_destroy(plan);
    return NLO_VEC_STATUS_UNSUPPORTED;
}

void nlo_fft_plan_destroy(nlo_fft_plan* plan)
{
    if (plan == NULL) {
        return;
    }

#if defined(NLO_FFT_BACKEND_FFTW)
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
#endif

#if defined(NLO_ENABLE_VECTOR_BACKEND_VULKAN) && defined(NLO_ENABLE_VKFFT)
    if (plan->backend_type == NLO_VECTOR_BACKEND_VULKAN) {
        deleteVkFFT(&plan->vk_app);
    }
    if (plan->backend != NULL && plan->vk_placeholder_vec != NULL) {
        nlo_vec_destroy(plan->backend, plan->vk_placeholder_vec);
        plan->vk_placeholder_vec = NULL;
    }
#endif

    free(plan);
}

nlo_vec_status nlo_fft_forward_vec(nlo_fft_plan* plan,
                                   const nlo_vec_buffer* input,
                                   nlo_vec_buffer* output)
{
    if (plan == NULL || input == NULL || output == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (input->owner != plan->backend || output->owner != plan->backend ||
        input->kind != NLO_VEC_KIND_COMPLEX64 || output->kind != NLO_VEC_KIND_COMPLEX64 ||
        input->length != plan->signal_size || output->length != plan->signal_size) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    if (plan->backend_type == NLO_VECTOR_BACKEND_CPU) {
#if defined(NLO_FFT_BACKEND_FFTW)
        const void* in_ptr = NULL;
        void* out_ptr = NULL;
        nlo_vec_status status = nlo_vec_get_const_host_ptr(plan->backend, input, &in_ptr);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
        status = nlo_vec_get_host_ptr(plan->backend, output, &out_ptr);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        fftw_execute_dft(plan->forward_plan, (fftw_complex*)in_ptr, (fftw_complex*)out_ptr);
        return NLO_VEC_STATUS_OK;
#else
        return NLO_VEC_STATUS_UNSUPPORTED;
#endif
    }

#if defined(NLO_ENABLE_VECTOR_BACKEND_VULKAN) && defined(NLO_ENABLE_VKFFT)
    if (plan->backend_type == NLO_VECTOR_BACKEND_VULKAN) {
        nlo_vec_status status = NLO_VEC_STATUS_OK;
        if (input != output) {
            status = nlo_vec_complex_copy(plan->backend, output, input);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
        }
        return nlo_fft_vk_execute_inplace(plan, output, 0);
    }
#endif

    return NLO_VEC_STATUS_UNSUPPORTED;
}

nlo_vec_status nlo_fft_inverse_vec(nlo_fft_plan* plan,
                                   const nlo_vec_buffer* input,
                                   nlo_vec_buffer* output)
{
    if (plan == NULL || input == NULL || output == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (input->owner != plan->backend || output->owner != plan->backend ||
        input->kind != NLO_VEC_KIND_COMPLEX64 || output->kind != NLO_VEC_KIND_COMPLEX64 ||
        input->length != plan->signal_size || output->length != plan->signal_size) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    if (plan->backend_type == NLO_VECTOR_BACKEND_CPU) {
#if defined(NLO_FFT_BACKEND_FFTW)
        const void* in_ptr = NULL;
        void* out_ptr = NULL;
        nlo_vec_status status = nlo_vec_get_const_host_ptr(plan->backend, input, &in_ptr);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
        status = nlo_vec_get_host_ptr(plan->backend, output, &out_ptr);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        fftw_execute_dft(plan->inverse_plan, (fftw_complex*)in_ptr, (fftw_complex*)out_ptr);

        nlo_complex* data = (nlo_complex*)out_ptr;
        for (size_t i = 0; i < plan->signal_size; ++i) {
            NLO_RE(data[i]) *= plan->inverse_scale;
            NLO_IM(data[i]) *= plan->inverse_scale;
        }
        return NLO_VEC_STATUS_OK;
#else
        return NLO_VEC_STATUS_UNSUPPORTED;
#endif
    }

#if defined(NLO_ENABLE_VECTOR_BACKEND_VULKAN) && defined(NLO_ENABLE_VKFFT)
    if (plan->backend_type == NLO_VECTOR_BACKEND_VULKAN) {
        nlo_vec_status status = NLO_VEC_STATUS_OK;
        if (input != output) {
            status = nlo_vec_complex_copy(plan->backend, output, input);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
        }
        return nlo_fft_vk_execute_inplace(plan, output, 1);
    }
#endif

    return NLO_VEC_STATUS_UNSUPPORTED;
}
