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

#ifndef NLO_ENABLE_VULKAN_BACKEND
#define NLO_ENABLE_VULKAN_BACKEND 1
#endif

#if NLO_ENABLE_VULKAN_BACKEND
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
typedef void* VkQueryPool;
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
    NLO_VK_LOCAL_SIZE_X = 64u,
    /** Default staging-buffer size used for host/device transfers. */
    NLO_VK_DEFAULT_STAGING_BYTES = 8u * 1024u * 1024u,
    /** Total descriptor bindings reserved in the shared compute set layout. */
    NLO_VK_DESCRIPTOR_BINDING_COUNT = 11u
};

typedef enum {
    NLO_VK_BINDING_DST = 0,
    NLO_VK_BINDING_FIELD = 1,
    NLO_VK_BINDING_W = 2,
    NLO_VK_BINDING_WT = 3,
    NLO_VK_BINDING_KX = 4,
    NLO_VK_BINDING_KY = 5,
    NLO_VK_BINDING_T = 6,
    NLO_VK_BINDING_X = 7,
    NLO_VK_BINDING_Y = 8,
    NLO_VK_BINDING_D = 9,
    NLO_VK_BINDING_V = 10
} nlo_vk_descriptor_binding;

/**
 * @brief Internal kernel identifiers mapped to compiled Vulkan pipelines.
 */
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
    NLO_VK_KERNEL_COMPLEX_AXIS_UNSHIFTED_FROM_DELTA = 13,
    NLO_VK_KERNEL_COMPLEX_AXIS_CENTERED_FROM_DELTA = 14,
    NLO_VK_KERNEL_COMPLEX_MESH_FROM_AXIS_TFAST_T = 15,
    NLO_VK_KERNEL_COMPLEX_MESH_FROM_AXIS_TFAST_Y = 16,
    NLO_VK_KERNEL_COMPLEX_MESH_FROM_AXIS_TFAST_X = 17,
    NLO_VK_KERNEL_COMPLEX_AXPY_INPLACE_REAL = 18,
    NLO_VK_KERNEL_COMPLEX_AFFINE_COMB2_REAL = 19,
    NLO_VK_KERNEL_COMPLEX_AFFINE_COMB3_REAL = 20,
    NLO_VK_KERNEL_COMPLEX_AFFINE_COMB4_REAL = 21,
    NLO_VK_KERNEL_COMPLEX_EMBEDDED_ERROR_PAIR_REAL = 22,
    NLO_VK_KERNEL_COMPLEX_LERP = 23,
    NLO_VK_KERNEL_COUNT = 24
} nlo_vk_kernel_id;

/**
 * @brief Common push-constant payload shared by vector kernels.
 */
typedef struct {
    uint32_t count;
    uint32_t pad;
    double scalar0;
    double scalar1;
    double scalar2;
    double scalar3;
} nlo_vk_push_constants;

/**
 * @brief Cached Vulkan pipeline wrapper for one compute kernel.
 */
typedef struct {
    VkPipeline pipeline;
} nlo_vk_kernel;

typedef struct nlo_vk_operator_jit_entry nlo_vk_operator_jit_entry;

typedef struct {
    uint32_t event_id;
    uint32_t start_query;
    uint32_t end_query;
} nlo_vk_timestamp_span;

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
    VkQueryPool timestamp_query_pool;
    nlo_vk_timestamp_span* timestamp_spans;
    uint32_t timestamp_query_capacity;
    uint32_t timestamp_query_cursor;
    uint32_t timestamp_span_capacity;
    uint32_t timestamp_span_count;
    uint32_t timestamp_valid_bits;
    double timestamp_period_ns;
    bool timestamp_queries_supported;

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
    bool submit_pending;
    uint32_t simulation_descriptor_set_cursor;
    nlo_vk_operator_jit_entry* operator_jit_entries;
} nlo_vk_backend;

/**
 * @brief Internal CUDA backend scaffold state for future device/runtime ownership.
 */
typedef struct {
    int initialized;
    int device_ordinal;
    uint32_t active_device_count;
    int peer_access_enabled;
} nlo_cuda_backend;

/**
 * @brief Concrete backend instance shared across CPU, Vulkan, and CUDA paths.
 */
struct nlo_vector_backend {
    nlo_vector_backend_type type;
    bool in_simulation;
    nlo_vk_backend vk;
    nlo_cuda_backend cuda;
};

/**
 * @brief Opaque buffer storage used by backend operations.
 */
struct nlo_vec_buffer {
    nlo_vector_backend* owner;
    nlo_vec_kind kind;
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
size_t nlo_vec_element_size(nlo_vec_kind kind);

/**
 * @brief Validate backend handle and internal invariants.
 *
 * @param backend Backend handle to validate.
 * @return nlo_vec_status Validation status.
 */
nlo_vec_status nlo_vec_validate_backend(const nlo_vector_backend* backend);

/**
 * @brief Validate one buffer against backend ownership and expected kind.
 *
 * @param backend Expected owner backend.
 * @param buffer Buffer to validate.
 * @param kind Expected element kind.
 * @return nlo_vec_status Validation status.
 */
nlo_vec_status nlo_vec_validate_buffer(
    const nlo_vector_backend* backend,
    const nlo_vec_buffer* buffer,
    nlo_vec_kind kind
);

/**
 * @brief Validate two buffers for paired operations.
 *
 * @param backend Expected owner backend.
 * @param a First buffer.
 * @param b Second buffer.
 * @param kind Expected element kind for both buffers.
 * @return nlo_vec_status Validation status.
 */
nlo_vec_status nlo_vec_validate_pair(
    const nlo_vector_backend* backend,
    const nlo_vec_buffer* a,
    const nlo_vec_buffer* b,
    nlo_vec_kind kind
);

nlo_vec_status nlo_vec_complex_axpy_inplace_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src,
    double alpha
);

nlo_vec_status nlo_vec_complex_affine_comb2_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* a,
    double alpha,
    const nlo_vec_buffer* b,
    double beta
);

nlo_vec_status nlo_vec_complex_affine_comb3_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* a,
    double alpha,
    const nlo_vec_buffer* b,
    double beta,
    const nlo_vec_buffer* c,
    double gamma
);

nlo_vec_status nlo_vec_complex_affine_comb4_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* a,
    double alpha,
    const nlo_vec_buffer* b,
    double beta,
    const nlo_vec_buffer* c,
    double gamma,
    const nlo_vec_buffer* d,
    double delta
);

nlo_vec_status nlo_vec_complex_embedded_error_pair_real(
    nlo_vector_backend* backend,
    nlo_vec_buffer* fine_out,
    nlo_vec_buffer* coarse_out,
    const nlo_vec_buffer* base,
    const nlo_vec_buffer* stage_k4,
    double fine_k4_coeff,
    double coarse_k4_coeff,
    const nlo_vec_buffer* stage_k5,
    double coarse_k5_coeff
);

nlo_vec_status nlo_vec_complex_lerp(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* a,
    const nlo_vec_buffer* b,
    double alpha
);


