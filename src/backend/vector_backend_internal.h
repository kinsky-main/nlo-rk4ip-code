/**
 * @file vector_backend_internal.h
 * @dir src/backend
 * @brief Internal backend structures and helpers shared across implementation units.
 */
#pragma once

#include "backend/vector_backend.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef NLO_ENABLE_VECTOR_BACKEND_VULKAN
#include <vulkan/vulkan.h>

enum {
    NLO_VK_LOCAL_SIZE_X = 64u,
    NLO_VK_DEFAULT_STAGING_BYTES = 8u * 1024u * 1024u
};

typedef enum {
    NLO_VK_KERNEL_REAL_FILL = 0,
    NLO_VK_KERNEL_REAL_MUL_INPLACE = 1,
    NLO_VK_KERNEL_COMPLEX_FILL = 2,
    NLO_VK_KERNEL_COMPLEX_SCALAR_MUL_INPLACE = 3,
    NLO_VK_KERNEL_COMPLEX_ADD_INPLACE = 4,
    NLO_VK_KERNEL_COMPLEX_MUL_INPLACE = 5,
    NLO_VK_KERNEL_COMPLEX_MAGNITUDE_SQUARED = 6,
    NLO_VK_KERNEL_COMPLEX_EXP_INPLACE = 7,
    NLO_VK_KERNEL_COMPLEX_RELATIVE_ERROR_REDUCE = 8,
    NLO_VK_KERNEL_REAL_MAX_REDUCE = 9,
    NLO_VK_KERNEL_COUNT = 10
} nlo_vk_kernel_id;

typedef struct {
    uint32_t count;
    uint32_t pad;
    double scalar0;
    double scalar1;
} nlo_vk_push_constants;

typedef struct {
    VkPipeline pipeline;
} nlo_vk_kernel;

typedef struct {
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family_index;
    bool shader_float64_supported;
    VkPhysicalDeviceLimits limits;

    VkCommandPool command_pool;
    bool owns_command_pool;
    VkCommandBuffer command_buffer;
    VkFence submit_fence;

    VkDescriptorSetLayout descriptor_set_layout;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSet descriptor_set;
    VkPipelineLayout pipeline_layout;
    VkPipelineCache pipeline_cache;
    nlo_vk_kernel kernels[NLO_VK_KERNEL_COUNT];

    VkBuffer staging_buffer;
    VkDeviceMemory staging_memory;
    void* staging_mapped_ptr;
    VkDeviceSize staging_capacity;

    VkBuffer reduction_buffer_a;
    VkDeviceMemory reduction_memory_a;
    VkBuffer reduction_buffer_b;
    VkDeviceMemory reduction_memory_b;
    VkDeviceSize reduction_capacity;

    VkDeviceSize max_kernel_chunk_bytes;
} nlo_vk_backend;
#endif

struct nlo_vector_backend {
    nlo_vector_backend_type type;
    bool in_simulation;
#ifdef NLO_ENABLE_VECTOR_BACKEND_VULKAN
    nlo_vk_backend vk;
#endif
};

struct nlo_vec_buffer {
    nlo_vector_backend* owner;
    nlo_vec_kind kind;
    size_t length;
    size_t bytes;
    void* host_ptr;
#ifdef NLO_ENABLE_VECTOR_BACKEND_VULKAN
    VkBuffer vk_buffer;
    VkDeviceMemory vk_memory;
#endif
};

size_t nlo_vec_element_size(nlo_vec_kind kind);
nlo_vec_status nlo_vec_validate_backend(const nlo_vector_backend* backend);
nlo_vec_status nlo_vec_validate_buffer(const nlo_vector_backend* backend,
                                      const nlo_vec_buffer* buffer,
                                      nlo_vec_kind kind);
nlo_vec_status nlo_vec_validate_pair(const nlo_vector_backend* backend,
                                    const nlo_vec_buffer* a,
                                    const nlo_vec_buffer* b,
                                    nlo_vec_kind kind);


