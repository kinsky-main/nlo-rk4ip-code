/**
 * @file vulkan_bench_context.h
 * @brief Vulkan setup helpers for benchmark executables.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef NLO_ENABLE_VECTOR_BACKEND_VULKAN
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family_index;
    VkCommandPool command_pool;
} nlo_bench_vk_context;

/**
 * @brief Create a minimal Vulkan context for compute benchmarks.
 *
 * @param context Output context.
 * @param reason Optional output buffer with skip/error reason text.
 * @param reason_capacity Size of reason buffer.
 * @return int 0 on success, non-zero on failure.
 */
int nlo_bench_vk_context_init(
    nlo_bench_vk_context* context,
    char* reason,
    size_t reason_capacity
);

/**
 * @brief Destroy all Vulkan resources created by nlo_bench_vk_context_init.
 *
 * @param context Context to destroy.
 */
void nlo_bench_vk_context_destroy(nlo_bench_vk_context* context);

#ifdef __cplusplus
}
#endif
#endif
