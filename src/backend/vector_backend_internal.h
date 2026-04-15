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

#ifndef ENABLE_VULKAN_BACKEND
#define ENABLE_VULKAN_BACKEND 1
#endif

#if ENABLE_VULKAN_BACKEND
#include <vulkan/vulkan.h>
#else
typedef void* VkInstance;
typedef void* VkPhysicalDevice;
typedef void* VkDevice;
typedef void* VkQueue;
typedef void* VkCommandPool;
typedef void* VkCommandBuffer;
typedef void* VkFence;
typedef void* VkDescriptorSetLayout;
typedef void* VkDescriptorPool;
typedef void* VkDescriptorSet;
typedef void* VkPipelineLayout;
typedef void* VkPipelineCache;
typedef void* VkPipeline;
typedef void* VkBuffer;
typedef void* VkDeviceMemory;
typedef uint64_t VkDeviceSize;
typedef int VkPhysicalDeviceType;
typedef struct {
    uint32_t maxComputeWorkGroupCount[3];
    uint64_t maxStorageBufferRange;
} VkPhysicalDeviceLimits;
#ifndef VK_MAX_PHYSICAL_DEVICE_NAME_SIZE
#define VK_MAX_PHYSICAL_DEVICE_NAME_SIZE 256
#endif
#ifndef VK_NULL_HANDLE
#define VK_NULL_HANDLE ((void*)0)
#endif
#endif

enum {
    /** Vulkan compute local size used by kernels in this backend. */
    VK_LOCAL_SIZE_X = 64u,
    /** Default staging-buffer size used for host/device transfers. */
    VK_DEFAULT_STAGING_BYTES = 8u * 1024u * 1024u
};

/**
 * @brief Internal kernel identifiers mapped to compiled Vulkan pipelines.
 */
typedef enum {
    VK_KERNEL_REAL_FILL = 0,
    VK_KERNEL_REAL_MUL_INPLACE = 1,
    VK_KERNEL_COMPLEX_FILL = 2,
    VK_KERNEL_COMPLEX_SCALAR_MUL_INPLACE = 3,
    VK_KERNEL_COMPLEX_ADD_INPLACE = 4,
    VK_KERNEL_COMPLEX_MUL_INPLACE = 5,
    VK_KERNEL_COMPLEX_MAGNITUDE_SQUARED = 6,
    VK_KERNEL_COMPLEX_EXP_INPLACE = 7,
    VK_KERNEL_COMPLEX_REAL_POW_INPLACE = 8,
    VK_KERNEL_COMPLEX_RELATIVE_ERROR_REDUCE = 9,
    VK_KERNEL_REAL_MAX_REDUCE = 10,
    VK_KERNEL_COMPLEX_WEIGHTED_RMS_REDUCE = 11,
    VK_KERNEL_PAIR_SUM_REDUCE = 12,
    VK_KERNEL_COMPLEX_AXIS_UNSHIFTED_FROM_DELTA = 13,
    VK_KERNEL_COMPLEX_AXIS_CENTERED_FROM_DELTA = 14,
    VK_KERNEL_COMPLEX_MESH_FROM_AXIS_TFAST_T = 15,
    VK_KERNEL_COMPLEX_MESH_FROM_AXIS_TFAST_Y = 16,
    VK_KERNEL_COMPLEX_MESH_FROM_AXIS_TFAST_X = 17,
    VK_KERNEL_COUNT = 18
} vk_kernel_id;

/**
 * @brief Common push-constant payload shared by vector kernels.
 */
typedef struct {
    uint32_t count;
    uint32_t pad;
    double scalar0;
    double scalar1;
} vk_push_constants;

/**
 * @brief Cached Vulkan pipeline wrapper for one compute kernel.
 */
typedef struct {
    VkPipeline pipeline;
} vk_kernel;

/**
 * @brief Internal Vulkan backend runtime state and resources.
 */
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
    vk_kernel kernels[VK_KERNEL_COUNT];

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
} vk_backend;

/**
 * @brief Concrete backend instance shared across CPU/Vulkan paths.
 */
struct vector_backend {
    vector_backend_type type;
    bool in_simulation;
    vk_backend vk;
};

/**
 * @brief Opaque buffer storage used by backend operations.
 */
struct vec_buffer {
    vector_backend* owner;
    vec_kind kind;
    size_t length;
    size_t bytes;
    void* host_ptr;
    VkBuffer vk_buffer;
    VkDeviceMemory vk_memory;
};

/**
 * @brief Return byte size of one element for a logical vector kind.
 *
 * @param kind Vector element kind.
 * @return size_t Element size in bytes.
 */
size_t vec_element_size(vec_kind kind);

/**
 * @brief Validate backend handle and internal invariants.
 *
 * @param backend Backend handle to validate.
 * @return vec_status Validation status.
 */
vec_status vec_validate_backend(const vector_backend* backend);

/**
 * @brief Validate one buffer against backend ownership and expected kind.
 *
 * @param backend Expected owner backend.
 * @param buffer Buffer to validate.
 * @param kind Expected element kind.
 * @return vec_status Validation status.
 */
vec_status vec_validate_buffer(
    const vector_backend* backend,
    const vec_buffer* buffer,
    vec_kind kind
);

/**
 * @brief Validate two buffers for paired operations.
 *
 * @param backend Expected owner backend.
 * @param a First buffer.
 * @param b Second buffer.
 * @param kind Expected element kind for both buffers.
 * @return vec_status Validation status.
 */
vec_status vec_validate_pair(
    const vector_backend* backend,
    const vec_buffer* a,
    const vec_buffer* b,
    vec_kind kind
);


