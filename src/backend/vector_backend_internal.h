/**
 * @file vector_backend_internal.h
 * @dir src/backend
 * @brief Internal backend structures and helpers shared across implementation units.
 */
#pragma once

#include "backend/vector_backend.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

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
    NLO_VK_KERNEL_COMPLEX_REAL_POW_INPLACE = 8,
    NLO_VK_KERNEL_COMPLEX_RELATIVE_ERROR_REDUCE = 9,
    NLO_VK_KERNEL_REAL_MAX_REDUCE = 10,
    NLO_VK_KERNEL_COMPLEX_WEIGHTED_RMS_REDUCE = 11,
    NLO_VK_KERNEL_PAIR_SUM_REDUCE = 12,
    NLO_VK_KERNEL_COUNT = 13
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
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family_index;
    bool owns_instance;
    bool owns_device;
    bool shader_float64_supported;
    VkPhysicalDeviceLimits limits;
    VkPhysicalDeviceType device_type;
    uint64_t device_local_bytes;
    char device_name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];

    VkCommandPool command_pool;
    bool owns_command_pool;
    VkCommandBuffer command_buffer;
    VkFence submit_fence;

    VkDescriptorSetLayout descriptor_set_layout;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSet* descriptor_sets;
    uint32_t descriptor_set_count;
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
    bool simulation_phase_recording;
    bool simulation_phase_has_commands;
    uint32_t simulation_descriptor_set_cursor;
} nlo_vk_backend;

struct nlo_vector_backend {
    nlo_vector_backend_type type;
    bool in_simulation;
    nlo_vk_backend vk;
};

struct nlo_vec_buffer {
    nlo_vector_backend* owner;
    nlo_vec_kind kind;
    size_t length;
    size_t bytes;
    void* host_ptr;
    VkBuffer vk_buffer;
    VkDeviceMemory vk_memory;
};

size_t nlo_vec_element_size(nlo_vec_kind kind);
nlo_vec_status nlo_vec_validate_backend(const nlo_vector_backend* backend);
nlo_vec_status nlo_vec_validate_buffer(
    const nlo_vector_backend* backend,
    const nlo_vec_buffer* buffer,
    nlo_vec_kind kind
);
nlo_vec_status nlo_vec_validate_pair(
    const nlo_vector_backend* backend,
    const nlo_vec_buffer* a,
    const nlo_vec_buffer* b,
    nlo_vec_kind kind
);


